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
) -> None:
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from google.cloud import bigquery
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

    def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.abs(y_true).sum()
        if denom == 0:
            return float("nan")
        return float(np.abs(y_true - y_pred).sum() / denom)

    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT *
        FROM `{bq_table}`
        WHERE unique_id IS NOT NULL
          AND ds IS NOT NULL
          AND y IS NOT NULL
        ORDER BY unique_id, ds
    """
    df = client.query(query).to_dataframe()

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

    counts = df.groupby("unique_id").size()
    df = df[df["unique_id"].isin(counts[counts > horizon].index)]

    if df.empty:
        raise ValueError("No series left after filtering short series.")

    train_parts = []
    valid_parts = []

    for _, g in df.groupby("unique_id", sort=False):
        train_parts.append(g.iloc[:-horizon])
        valid_parts.append(g.iloc[-horizon:])

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)

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

    fcst.fit(
        train_df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )

    preds = fcst.predict(h=horizon)

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
        print("Validation sample:")
        print(valid_df.head())
        print("Prediction sample:")
        print(preds.head())
        raise ValueError("No overlapping rows between validation set and predictions.")

    y_true = merged["y"].to_numpy(dtype=float)
    y_pred = merged[model_type].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    metric_wmape = wmape(y_true, y_pred)

    eval_metrics.log_metric("rmse", rmse)
    eval_metrics.log_metric("mae", mae)
    eval_metrics.log_metric("wmape", metric_wmape)

    merged["run_ts"] = pd.Timestamp.utcnow()
    merged["model_display_name"] = model_display_name
    merged["model_type"] = model_type
    merged["source_table"] = bq_table
    merged["forecast_freq"] = forecast_freq
    merged["horizon"] = horizon

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(
        merged,
        prediction_bq_table,
        job_config=job_config,
    ).result()

    pred_path = Path(eval_predictions.path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(pred_path, index=False)

    model_dir = Path(model_artifact.path)
    model_dir.mkdir(parents=True, exist_ok=True)
    fcst.save(model_dir)

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
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


@dsl.pipeline(
    name="nixtla-mlforecast-parallel-pipeline",
    description="Train multiple MLForecast models in parallel on Vertex AI Pipelines",
)
def mlforecast_parallel_pipeline(
    project_id: str,
    bq_table: str,
    prediction_bq_table: str,
    forecast_freq: str = "D",
    horizon: int = 2,
    lags: list[int] = [1, 2, 3],
    date_features: list[str] = ["dayofweek", "month"],
    model_display_name: str = "mlforecast",
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


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlforecast_parallel_pipeline,
        package_path="nixtla_mlforecast_parallel_pipeline.yaml",
    )