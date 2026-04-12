from google.cloud import aiplatform

PROJECT_ID = "dazzling-seat-366014"
REGION = "us-central1"
PIPELINE_ROOT = "gs://dazzling-seat-366014-vertex-pipelines"
SERVICE_ACCOUNT = "278930531128-compute@developer.gserviceaccount.com"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINE_ROOT,
)

job = aiplatform.PipelineJob(
    display_name="nixtla-mlforecast-parallel-run",
    template_path="nixtla_mlforecast_parallel_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "bq_table": "dazzling-seat-366014.forecasting.sales_daily",
        "prediction_bq_table": "dazzling-seat-366014.forecasting.sales_predictions",
        "forecast_freq": "D",
        "horizon": 2,
        "lags": [1, 2, 3],
        "date_features": ["dayofweek", "month"],
        "model_display_name": "sales-mlforecast-v2",
    },
)

job.submit(
    service_account=SERVICE_ACCOUNT,
    enable_preflight_validations=True,
)

job.wait()