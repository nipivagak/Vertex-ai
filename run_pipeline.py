import logging

from google.cloud import aiplatform

PROJECT_ID = "dazzling-seat-366014"
REGION = "us-central1"
PIPELINE_ROOT = "gs://dazzling-seat-366014-vertex-pipelines"
SERVICE_ACCOUNT = "278930531128-compute@developer.gserviceaccount.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_pipeline")

PARAMETER_VALUES = {
    "project_id": PROJECT_ID,
    "bq_table": "dazzling-seat-366014.forecasting.sales_daily",
    "prediction_bq_table": "dazzling-seat-366014.forecasting.sales_predictions",
    "champion_bq_table": "dazzling-seat-366014.forecasting.sales_champion",
    "forecast_freq": "D",
    "horizon": 2,
    "lags": [1, 2, 3],
    "date_features": ["dayofweek", "month"],
    "model_display_name": "sales-mlforecast-v3",
    "champion_metric": "wmape",
}

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
    display_name="nixtla-mlforecast-parallel-run",
    template_path="nixtla_mlforecast_parallel_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    parameter_values=PARAMETER_VALUES,
)

logger.info(
    "Created pipeline job (display_name=%s, template=%s)",
    "nixtla-mlforecast-parallel-run",
    "nixtla_mlforecast_parallel_pipeline.yaml",
)
logger.info("Submitting pipeline with params: %s", PARAMETER_VALUES)

try:
    job.submit(
        service_account=SERVICE_ACCOUNT,
        enable_preflight_validations=True,
    )
    logger.info("Submission accepted. Resource name: %s", job.resource_name)

    logger.info("Waiting for pipeline completion...")
    job.wait()
    logger.info("Pipeline finished. State: %s", job.state)
except Exception:
    logger.exception("Pipeline execution failed")
    raise