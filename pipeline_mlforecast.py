from kfp import compiler, dsl
from kfp.dsl import Output, Model, Metrics, Artifact


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "lightgbm>=4.3.0",
        "mlforecast>=0.13.0",
        "google-cloud-bigquery>=3.25.0",
        "google-cloud-storage>=2.18.0",
        "pyarrow>=17.0.0",
        "db-dtypes>=1.2.0",
    ],
)
def train_single_model_component(
    project_id: str,
    bq_table: str,
    prediction_bq_table: str,
    forecast_freq: str,
    horizon: int,
    lags: list[int],
    date_features: list[str],
    model_display_name: str,
    model_type: str,
    model_artifact: Output[Model],
    eval_metrics: Output[Metrics],
    eval_predictions: Output[Artifact],
) -> str:
    import json
    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from google.cloud import bigquery
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("train_single_model_component")
    logger.info("Starting training component for model_type=%s", model_type)

    def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.abs(y_true).sum()
        if denom == 0:
            return float("nan")
        return float(np.abs(y_true - y_pred).sum() / denom)

    client = bigquery.Client(project=project_id)
    logger.info("Running BigQuery training query for table=%s", bq_table)
    query = f"""
        SELECT *
        FROM `{bq_table}`
        WHERE unique_id IS NOT NULL
          AND ds IS NOT NULL
          AND y IS NOT NULL
        ORDER BY unique_id, ds
    """
    df = client.query(query).to_dataframe()
    logger.info("Fetched %d rows from BigQuery", len(df))

    if df.empty:
        raise ValueError("No rows returned from BigQuery.")

    required_cols = {"unique_id", "ds", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["unique_id", "ds", "y"]).sort_values(["unique_id", "ds"])
    logger.info("Rows after type coercion and null filtering: %d", len(df))

    counts = df.groupby("unique_id").size()
    df = df[df["unique_id"].isin(counts[counts > horizon].index)]
    logger.info(
        "Rows after short-series filtering: %d (series_count=%d)",
        len(df),
        df["unique_id"].nunique(),
    )

    if df.empty:
        raise ValueError("No series left after filtering short series.")

    train_parts = []
    valid_parts = []

    for _, g in df.groupby("unique_id", sort=False):
        train_parts.append(g.iloc[:-horizon])
        valid_parts.append(g.iloc[-horizon:])

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    logger.info("Train rows=%d, validation rows=%d", len(train_df), len(valid_df))

    if model_type == "lgbm":
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "et":
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logger.info("Initialized model for model_type=%s", model_type)

    allowed_date_features = [
        "year",
        "month",
        "week",
        "day",
        "dayofweek",
        "dayofyear",
        "quarter",
    ]
    parsed_date_features = [x for x in date_features if x in allowed_date_features]

    fcst = MLForecast(
        models={model_type: model},
        freq=forecast_freq,
        lags=lags,
        date_features=parsed_date_features,
        num_threads=1,
    )

    logger.info(
        "Fitting MLForecast with freq=%s, horizon=%d, lags=%s, date_features=%s",
        forecast_freq,
        horizon,
        lags,
        parsed_date_features,
    )

    fcst.fit(
        train_df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )

    preds = fcst.predict(h=horizon)
    logger.info("Generated predictions with %d rows", len(preds))

    if model_type not in preds.columns:
        raise ValueError(
            f"Expected prediction column '{model_type}'. Got columns: {preds.columns.tolist()}"
        )

    merged = valid_df.merge(
        preds[["unique_id", "ds", model_type]],
        on=["unique_id", "ds"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlapping rows between validation set and predictions.")
    logger.info("Merged validation/prediction rows=%d", len(merged))

    merged = merged.rename(columns={model_type: "prediction"})

    y_true = merged["y"].to_numpy(dtype=float)
    y_pred = merged["prediction"].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    metric_wmape = wmape(y_true, y_pred)

    eval_metrics.log_metric("rmse", rmse)
    eval_metrics.log_metric("mae", mae)
    eval_metrics.log_metric("wmape", metric_wmape)
    logger.info("Metrics: rmse=%.6f, mae=%.6f, wmape=%.6f", rmse, mae, metric_wmape)

    merged["run_ts"] = pd.Timestamp.utcnow()
    merged["model_display_name"] = model_display_name
    merged["model_type"] = model_type
    merged["source_table"] = bq_table
    merged["forecast_freq"] = forecast_freq
    merged["horizon"] = horizon

    merged["unique_id"] = merged["unique_id"].astype(str)
    merged["y"] = pd.to_numeric(merged["y"], errors="coerce")
    merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce")
    merged["model_display_name"] = merged["model_display_name"].astype(str)
    merged["model_type"] = merged["model_type"].astype(str)
    merged["source_table"] = merged["source_table"].astype(str)
    merged["forecast_freq"] = merged["forecast_freq"].astype(str)
    merged["horizon"] = merged["horizon"].astype("int64")

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(
        merged[
            [
                "unique_id",
                "ds",
                "y",
                "prediction",
                "run_ts",
                "model_display_name",
                "model_type",
                "source_table",
                "forecast_freq",
                "horizon",
            ]
        ],
        prediction_bq_table,
        job_config=job_config,
    ).result()
    logger.info(
        "Wrote %d prediction rows to %s",
        len(merged),
        prediction_bq_table,
    )

    pred_path = Path(eval_predictions.path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(pred_path, index=False)
    logger.info("Saved evaluation predictions artifact at %s", pred_path)

    model_dir = Path(model_artifact.path)
    model_dir.mkdir(parents=True, exist_ok=True)
    fcst.save(model_dir)
    logger.info("Saved model artifact at %s", model_dir)

    metadata = {
        "model_display_name": model_display_name,
        "model_type": model_type,
        "source_table": bq_table,
        "prediction_bq_table": prediction_bq_table,
        "forecast_freq": forecast_freq,
        "horizon": horizon,
        "lags": lags,
        "date_features": parsed_date_features,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "series_count": int(df["unique_id"].nunique()),
        "rmse": rmse,
        "mae": mae,
        "wmape": metric_wmape,
        "model_uri": model_artifact.uri,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Completed training component for model_type=%s", model_type)

    return json.dumps(metadata)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.0",
        "google-cloud-bigquery>=3.25.0",
        "pyarrow>=17.0.0",
        "db-dtypes>=1.2.0",
    ],
)
def select_champion_component(
    project_id: str,
    champion_bq_table: str,
    lgbm_result: str,
    rf_result: str,
    et_result: str,
    metric_name: str = "wmape",
) -> str:
    import json
    import logging

    import pandas as pd
    from google.cloud import bigquery

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("select_champion_component")

    results = [
        json.loads(lgbm_result),
        json.loads(rf_result),
        json.loads(et_result),
    ]
    logger.info("Loaded %d model results for champion selection", len(results))

    valid = [r for r in results if metric_name in r and r[metric_name] is not None]
    if not valid:
        raise ValueError(f"No valid results found for metric '{metric_name}'.")
    logger.info("Found %d valid results using metric=%s", len(valid), metric_name)

    champion = min(valid, key=lambda r: r[metric_name])
    logger.info(
        "Champion selected: model_type=%s, %s=%.6f",
        champion["model_type"],
        metric_name,
        champion[metric_name],
    )

    champion_summary = {
        "champion_model_type": champion["model_type"],
        "champion_metric_name": metric_name,
        "champion_metric_value": champion[metric_name],
        "champion_model_uri": champion["model_uri"],
        "project_id": project_id,
        "champion_bq_table": champion_bq_table,
        "run_ts": pd.Timestamp.utcnow().isoformat(),
        "all_results": results,
    }

    champion_row = {
        "run_ts": pd.Timestamp.utcnow(),
        "champion_model_type": champion["model_type"],
        "champion_metric_name": metric_name,
        "champion_metric_value": champion[metric_name],
        "champion_model_uri": champion["model_uri"],
        "model_display_name": champion.get("model_display_name"),
        "source_table": champion.get("source_table"),
        "prediction_bq_table": champion.get("prediction_bq_table"),
        "forecast_freq": champion.get("forecast_freq"),
        "horizon": champion.get("horizon"),
    }

    client = bigquery.Client(project=project_id)
    champion_df = pd.DataFrame([champion_row])
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(
        champion_df,
        champion_bq_table,
        job_config=job_config,
    ).result()
    logger.info("Wrote champion summary to %s", champion_bq_table)

    print(json.dumps(champion_summary, indent=2))
    return json.dumps(champion_summary)


@dsl.pipeline(
    name="nixtla-mlforecast-parallel-pipeline",
    description="Train multiple MLForecast models in parallel on Vertex AI Pipelines and select champion",
)
def mlforecast_parallel_pipeline(
    project_id: str,
    bq_table: str,
    prediction_bq_table: str,
    champion_bq_table: str,
    forecast_freq: str = "D",
    horizon: int = 2,
    lags: list[int] = [1, 2, 3],
    date_features: list[str] = ["dayofweek", "month"],
    model_display_name: str = "mlforecast",
    champion_metric: str = "wmape",
):
    lgbm_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="lgbm",
    )
    lgbm_task.set_display_name("train-lgbm")

    rf_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="rf",
    )
    rf_task.set_display_name("train-rf")

    et_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="et",
    )
    et_task.set_display_name("train-et")

    champion_task = select_champion_component(
        project_id=project_id,
        champion_bq_table=champion_bq_table,
        lgbm_result=lgbm_task.outputs["Output"],
        rf_result=rf_task.outputs["Output"],
        et_result=et_task.outputs["Output"],
        metric_name=champion_metric,
    )
    champion_task.set_display_name("select-champion")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlforecast_parallel_pipeline,
        package_path="nixtla_mlforecast_parallel_pipeline.yaml",
    )