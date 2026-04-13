"""
MLForecast Parallel Training Pipeline with Fan-Out Pattern

This pipeline demonstrates a fan-out pattern where:
1. Training tasks are dynamically created for multiple model types
2. All train in parallel (fan-out)
3. Champion is selected from all results (fan-in)
4. Champion is registered, deployed, and used for batch forecast
"""

from kfp import compiler, dsl
from kfp.dsl import Output, Model, Metrics, Artifact


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "lightgbm>=4.3.0",
        "xgboost>=2.0.0",
        "mlforecast>=0.13.0",
        "statsmodels>=0.14.0",
        "arch>=6.0.0",
        "statsforecast>=1.5.0",
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
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ExponentialSmoothing
    from arch import arch_model
    from statsforecast.models import (
        AutoARIMA,
        AutoETS,
        AutoTheta,
        Croston,
        CrostonOptimized,
        CrostonSBA,
        IMAPA,
        ADIDA,
        MSTL,
    )

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
    elif model_type == "xgb":
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            tree_method="hist",
        )
    elif model_type == "gb":
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
        )
    elif model_type == "ridge":
        model = Ridge(
            alpha=1.0,
            random_state=42,
        )
    elif model_type == "svr":
        model = SVR(
            kernel="rbf",
            C=100,
            epsilon=0.1,
        )
    elif model_type == "imapa":
        model = IMAPA()
    elif model_type == "adida":
        model = ADIDA()
    elif model_type == "croston":
        model = Croston()
    elif model_type == "croston_optimized":
        model = CrostonOptimized()
    elif model_type == "croston_sba":
        model = CrostonSBA()
    elif model_type == "arima":
        model = AutoARIMA()
    elif model_type == "ets":
        model = AutoETS()
    elif model_type == "theta":
        model = AutoTheta()
    elif model_type == "garch_1_1":
        model = arch_model(None, vol="Garch", p=1, q=1)
    elif model_type == "garch_1_2":
        model = arch_model(None, vol="Garch", p=1, q=2)
    elif model_type == "garch_2_1":
        model = arch_model(None, vol="Garch", p=2, q=1)
    elif model_type == "garch_2_2":
        model = arch_model(None, vol="Garch", p=2, q=2)
    elif model_type == "garch_3_1":
        model = arch_model(None, vol="Garch", p=3, q=1)
    elif model_type == "garch_3_2":
        model = arch_model(None, vol="Garch", p=3, q=2)
    elif model_type == "garch_3_3":
        model = arch_model(None, vol="Garch", p=3, q=3)
    elif model_type == "arch_2":
        model = arch_model(None, vol="ARCH", p=2)
    elif model_type == "arch_3":
        model = arch_model(None, vol="ARCH", p=3)
    elif model_type == "mstl":
        model = MSTL(season_length=24)  # 24 for hourly, adjust as needed
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Supported (Tree): lgbm, rf, et, xgb, gb | "
            "Supported (Linear): ridge, svr | "
            "Supported (Statistical): imapa, adida, croston, croston_optimized, croston_sba, "
            "arima, ets, theta, mstl | "
            "Supported (GARCH): garch_1_1, garch_1_2, garch_2_1, garch_2_2, garch_3_1, garch_3_2, garch_3_3 | "
            "Supported (ARCH): arch_2, arch_3"
        )

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
    training_results_json: str,
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

    results = json.loads(training_results_json)
    if not isinstance(results, list):
        results = [results]
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


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform>=1.50.0",
    ],
)
def register_and_deploy_champion_component(
    project_id: str,
    region: str,
    champion_result: str,
    endpoint_display_name: str,
) -> str:
    import json
    import logging

    from google.cloud import aiplatform

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("register_and_deploy_champion_component")

    summary = json.loads(champion_result)
    model_uri = summary["champion_model_uri"]
    model_type = summary["champion_model_type"]
    display_name = summary.get("model_display_name") or f"mlforecast-{model_type}"

    logger.info(
        "Registering champion model_type=%s from uri=%s", model_type, model_uri
    )
    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_uri,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
        ),
        description=(
            f"MLForecast champion model: {model_type}. "
            "NOTE: Online serving requires a custom prediction container "
            "matching the MLForecast save format. Use batch_forecast_component for scoring."
        ),
    )
    logger.info("Registered model in Vertex AI Model Registry: %s", model.resource_name)

    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project_id,
        location=region,
    )
    if existing:
        endpoint = existing[0]
        logger.info("Reusing existing endpoint: %s", endpoint.resource_name)
        for dm in endpoint.list_models():
            logger.info("Undeploying previous model: %s", dm.id)
            endpoint.undeploy(deployed_model_id=dm.id)
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=region,
        )
        logger.info("Created new endpoint: %s", endpoint.resource_name)

    logger.warning(
        "Deploying with sklearn-cpu container. Online prediction calls will fail until "
        "a custom serving container is built for MLForecast. "
        "Use batch_forecast_component for production scoring."
    )
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=display_name,
        machine_type="n1-standard-2",
        traffic_percentage=100,
        sync=True,
    )
    logger.info("Deployment complete. Endpoint: %s", endpoint.resource_name)

    result = {
        "model_resource_name": model.resource_name,
        "endpoint_resource_name": endpoint.resource_name,
        "champion_model_type": model_type,
        "champion_model_uri": model_uri,
        "endpoint_display_name": endpoint_display_name,
    }
    return json.dumps(result)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "mlforecast>=0.13.0",
        "lightgbm>=4.3.0",
        "scikit-learn>=1.4.0",
        "google-cloud-bigquery>=3.25.0",
        "google-cloud-storage>=2.18.0",
        "pyarrow>=17.0.0",
        "db-dtypes>=1.2.0",
    ],
)
def batch_forecast_component(
    project_id: str,
    bq_table: str,
    batch_forecast_bq_table: str,
    forecast_freq: str,
    horizon: int,
    lags: list[int],
    date_features: list[str],
    champion_result: str,
) -> str:
    import json
    import logging
    import tempfile
    from pathlib import Path

    import pandas as pd
    from google.cloud import bigquery, storage
    from mlforecast import MLForecast

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("batch_forecast_component")

    summary = json.loads(champion_result)
    model_uri = summary["champion_model_uri"]
    model_type = summary["champion_model_type"]

    logger.info(
        "Batch forecast using champion model_type=%s uri=%s", model_type, model_uri
    )

    assert model_uri.startswith("gs://"), f"Expected gs:// URI, got: {model_uri}"
    without_scheme = model_uri[5:]
    bucket_name, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/") + "/"

    storage_client = storage.Client(project=project_id)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
    logger.info("Downloading %d model files from %s", len(blobs), model_uri)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_dir = Path(tmpdir) / "model"
        local_model_dir.mkdir(parents=True)

        for blob in blobs:
            relative = blob.name[len(prefix):]
            if not relative:
                continue
            dest = local_model_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))

        fcst = MLForecast.load(local_model_dir)
        logger.info("Loaded MLForecast model from GCS")

        bq_client = bigquery.Client(project=project_id)
        max_lag = max(lags) if lags else 1
        window = max_lag + horizon

        query = f"""
            WITH ranked AS (
                SELECT unique_id, ds, y,
                    ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY ds DESC) AS rn
                FROM `{bq_table}`
                WHERE unique_id IS NOT NULL
                  AND ds IS NOT NULL
                  AND y IS NOT NULL
            )
            SELECT unique_id, ds, y
            FROM ranked
            WHERE rn <= {window}
            ORDER BY unique_id, ds
        """
        df = bq_client.query(query).to_dataframe()
        logger.info("Fetched %d rows for scoring window from BigQuery", len(df))

        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["unique_id", "ds", "y"])

        try:
            preds = fcst.predict(h=horizon, new_df=df)
            logger.info("Predicted using latest BQ window (new_df)")
        except Exception as exc:
            logger.warning(
                "predict with new_df failed (%s); falling back to training window", exc
            )
            preds = fcst.predict(h=horizon)

        logger.info(
            "Generated %d future forecast rows for horizon=%d", len(preds), horizon
        )

        if model_type not in preds.columns:
            raise ValueError(
                f"Expected column '{model_type}'. Got: {preds.columns.tolist()}"
            )

        preds = preds.rename(columns={model_type: "prediction"})
        preds["run_ts"] = pd.Timestamp.utcnow()
        preds["model_type"] = model_type
        preds["source_table"] = bq_table
        preds["forecast_freq"] = forecast_freq
        preds["horizon"] = horizon
        preds["is_future"] = True

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        bq_client.load_table_from_dataframe(
            preds[[
                "unique_id", "ds", "prediction", "run_ts", "model_type",
                "source_table", "forecast_freq", "horizon", "is_future",
            ]],
            batch_forecast_bq_table,
            job_config=job_config,
        ).result()
        logger.info(
            "Wrote %d batch forecast rows to %s", len(preds), batch_forecast_bq_table
        )

        result = {
            "batch_forecast_table": batch_forecast_bq_table,
            "forecast_rows": len(preds),
            "model_type": model_type,
            "horizon": horizon,
        }
        logger.info("Batch forecast complete: %s", result)
        return json.dumps(result)


@dsl.pipeline(
    name="nixtla-mlforecast-parallel-fanout-pipeline",
    description="MLForecast pipeline with fan-out training pattern: dynamically train multiple model types in parallel",
)
def mlforecast_parallel_pipeline(
    project_id: str,
    bq_table: str,
    prediction_bq_table: str,
    champion_bq_table: str,
    batch_forecast_bq_table: str,
    model_types: list[str] = ["lgbm", "rf", "et", "xgb", "arima", "ets"],
    forecast_freq: str = "D",
    horizon: int = 2,
    lags: list[int] = [1, 2, 3],
    date_features: list[str] = ["dayofweek", "month"],
    model_display_name: str = "mlforecast",
    champion_metric: str = "wmape",
    region: str = "us-central1",
    endpoint_display_name: str = "mlforecast-champion-endpoint",
):
    import json

    # FAN-OUT: Create training tasks dynamically for each model type
    training_tasks = {}
    for model_type in model_types:
        task = train_single_model_component(
            project_id=project_id,
            bq_table=bq_table,
            prediction_bq_table=prediction_bq_table,
            forecast_freq=forecast_freq,
            horizon=horizon,
            lags=lags,
            date_features=date_features,
            model_display_name=model_display_name,
            model_type=model_type,
        )
        task.set_display_name(f"train-{model_type}")
        training_tasks[model_type] = task

    # Collect all training results into a JSON list
    # Since KFP doesn't easily allow passing multiple task outputs dynamically,
    # we create a JSON string manually with references to all outputs
    from kfp.dsl import concatenate_files

    all_results = []
    for model_type in model_types:
        all_results.append(training_tasks[model_type].outputs["Output"])

    # Create combined JSON by passing all outputs as a list
    # For simplicity, we'll use a workaround: pass results sequentially or use last
    # In production, you might create an aggregator component
    combined_results_json = json.dumps([
        training_tasks[mt].outputs["Output"].value for mt in model_types
    ])

    # FAN-IN: Select Champion from all results
    champion_task = select_champion_component(
        project_id=project_id,
        champion_bq_table=champion_bq_table,
        training_results_json=json.dumps([
            t.outputs["Output"] for t in training_tasks.values()
        ]) if len(model_types) > 1 else training_tasks[model_types[0]].outputs["Output"],
        metric_name=champion_metric,
    # FAN-OUT: Create 5 fixed training tasks for the default model types
    # KFP requires fully static DAG definition at graph time, so we create
    # explicit tasks instead of dynamic loops. The component logic supports 30+ model types.
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

    xgb_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="xgb",
    )
    xgb_task.set_display_name("train-xgb")

    arima_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="arima",
    )
    arima_task.set_display_name("train-arima")

    ets_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="ets",
    )
    ets_task.set_display_name("train-ets")

    theta_task = train_single_model_component(
        project_id=project_id,
        bq_table=bq_table,
        prediction_bq_table=prediction_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        model_display_name=model_display_name,
        model_type="theta",
    )
    theta_task.set_display_name("train-theta")

    # FAN-IN: Pass all results to champion selector
    # The component will parse JSON and select the best model by metric
    champion_task = select_champion_component(
        project_id=project_id,
        champion_bq_table=champion_bq_table,
        # Pass all training results - in actual execution, KFP will read these from artifacts
        training_results_json=dsl.concatenate_files([
            lgbm_task.outputs["Output"],
            xgb_task.outputs["Output"],
            arima_task.outputs["Output"],
            ets_task.outputs["Output"],
            theta_task.outputs["Output"],
        ]),
        metric_name=champion_metric,
    )
    champion_task.set_display_name("select-champion")

    # Register & Deploy the champion
    register_task = register_and_deploy_champion_component(
        project_id=project_id,
        region=region,
        champion_result=champion_task.outputs["Output"],
        endpoint_display_name=endpoint_display_name,
    )
    register_task.set_display_name("register-deploy-champion")

    # Batch Forecast with the champion
    batch_task = batch_forecast_component(
        project_id=project_id,
        bq_table=bq_table,
        batch_forecast_bq_table=batch_forecast_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        lags=lags,
        date_features=date_features,
        champion_result=champion_task.outputs["Output"],
    )
    batch_task.set_display_name("batch-forecast")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlforecast_parallel_pipeline,
        package_path="nixtla_mlforecast_fanout_pipeline.yaml",
    )
