import logging
import sys


PROJECT_ID = "dazzling-seat-366014"
REGION = "us-central1"
PIPELINE_ROOT = "gs://dazzling-seat-366014-vertex-pipelines"
SERVICE_ACCOUNT = "278930531128-compute@developer.gserviceaccount.com"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_pipeline")


PIPELINE_MODE = sys.argv[1] if len(sys.argv) > 1 else "simple"
logger.info("Using pipeline mode: %s", PIPELINE_MODE)


PARAMETER_VALUES = {
    "project_id": PROJECT_ID,
    "bq_table": "forecasting.sales_daily",
    "prediction_bq_table": "forecasting.sales_predictions",
    "champion_bq_table": "forecasting.sales_champion",
    "batch_forecast_bq_table": "forecasting.sales_batch_forecasts",
    "model_types": ["lgbm", "xgb", "rf", "et", "ridge"],
    "region": REGION,
    "endpoint_display_name": "mlforecast-champion-endpoint",
    "forecast_freq": "D",
    "horizon": 2,
    "lags": [1, 2, 3],
    "date_features": ["dayofweek", "month"],
    "model_display_name": "sales-mlforecast-v3",
    "champion_metric": "wmape",
}


def run_vertex_pipeline(template_path: str, display_name: str, include_model_types: bool) -> None:
    from google.cloud import aiplatform

    parameter_values = dict(PARAMETER_VALUES)
    if not include_model_types:
        parameter_values.pop("model_types", None)

    logger.info(
        "Initializing Vertex AI client (project=%s, region=%s, pipeline_root=%s)",
        PROJECT_ID,
        REGION,
        PIPELINE_ROOT,
    )

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=PIPELINE_ROOT,
    )

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=template_path,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=parameter_values,
    )

    logger.info(
        "Created pipeline job (display_name=%s, template=%s)",
        display_name,
        template_path,
    )
    logger.info("Submitting pipeline with params: %s", parameter_values)

    job.submit(
        service_account=SERVICE_ACCOUNT,
        enable_preflight_validations=True,
    )
    logger.info("Submission accepted. Resource name: %s", job.resource_name)

    logger.info("Waiting for pipeline completion...")
    job.wait()
    logger.info("Pipeline finished. State: %s", job.state)


def run_local_pipeline() -> None:
    from local_pipeline_runner import run_local_pipeline as execute_local_pipeline

    logger.info("Running local pipeline execution path")
    result = execute_local_pipeline(
        project_id=PARAMETER_VALUES["project_id"],
        bq_table=PARAMETER_VALUES["bq_table"],
        prediction_bq_table=PARAMETER_VALUES["prediction_bq_table"],
        champion_bq_table=PARAMETER_VALUES["champion_bq_table"],
        batch_forecast_bq_table=PARAMETER_VALUES["batch_forecast_bq_table"],
        model_display_name=f"{PARAMETER_VALUES['model_display_name']}-local",
        forecast_freq=PARAMETER_VALUES["forecast_freq"],
        horizon=PARAMETER_VALUES["horizon"],
        lags=PARAMETER_VALUES["lags"],
        date_features=PARAMETER_VALUES["date_features"],
        model_types=PARAMETER_VALUES["model_types"],
    )
    logger.info("Local pipeline champion: %s", result["champion"])


def main() -> None:
    try:
        if PIPELINE_MODE == "simple":
            run_vertex_pipeline(
                template_path="nixtla_mlforecast_simple_fanout_pipeline.yaml",
                display_name="nixtla-mlforecast-simple-run",
                include_model_types=False,
            )
            return

        if PIPELINE_MODE == "original":
            run_vertex_pipeline(
                template_path="nixtla_mlforecast_parallel_pipeline.yaml",
                display_name="nixtla-mlforecast-original-run",
                include_model_types=False,
            )
            return

        if PIPELINE_MODE == "fanout":
            run_vertex_pipeline(
                template_path="nixtla_mlforecast_fanout_pipeline.yaml",
                display_name="nixtla-mlforecast-fanout-run",
                include_model_types=True,
            )
            return

        if PIPELINE_MODE == "local":
            run_local_pipeline()
            return

        raise ValueError("Unsupported mode. Use one of: simple, original, fanout, local")
    except Exception:
        logger.exception("Pipeline execution failed")
        raise


if __name__ == "__main__":
    main()