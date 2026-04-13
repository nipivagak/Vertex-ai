"""
setup_bq_tables.py
------------------
Creates required BigQuery tables for the MLForecast pipeline
if they do not already exist. Safe to run multiple times.

Usage:
    python3 setup_bq_tables.py
"""

import logging

from google.cloud import bigquery
from google.cloud.exceptions import Conflict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("setup_bq_tables")

PROJECT_ID = "dazzling-seat-366014"
DATASET_ID = "forecasting"

TABLES = {
    "sales_predictions": [
        bigquery.SchemaField("unique_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ds", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("y", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("prediction", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("run_ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("model_display_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("model_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_table", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("forecast_freq", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("horizon", "INT64", mode="NULLABLE"),
    ],
    "sales_champion": [
        bigquery.SchemaField("run_ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("champion_model_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("champion_metric_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("champion_metric_value", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("champion_model_uri", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("model_display_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_table", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("prediction_bq_table", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("forecast_freq", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("horizon", "INT64", mode="NULLABLE"),
    ],
    "sales_batch_forecasts": [
        bigquery.SchemaField("unique_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ds", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("prediction", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("run_ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("model_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_table", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("forecast_freq", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("horizon", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("is_future", "BOOL", mode="NULLABLE"),
    ],
}


def create_dataset_if_missing(client: bigquery.Client, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{dataset_id}")
    dataset_ref.location = "US"
    try:
        client.create_dataset(dataset_ref, exists_ok=True)
        logger.info("Dataset %s.%s is ready", PROJECT_ID, dataset_id)
    except Conflict:
        logger.info("Dataset %s.%s already exists", PROJECT_ID, dataset_id)


def create_table_if_missing(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    schema: list,
) -> None:
    full_table_id = f"{PROJECT_ID}.{dataset_id}.{table_id}"
    table = bigquery.Table(full_table_id, schema=schema)
    try:
        client.create_table(table, exists_ok=True)
        logger.info("Table %s is ready", full_table_id)
    except Conflict:
        logger.info("Table %s already exists", full_table_id)


def main() -> None:
    client = bigquery.Client(project=PROJECT_ID)
    logger.info("Connected to BigQuery project=%s", PROJECT_ID)

    create_dataset_if_missing(client, DATASET_ID)

    for table_id, schema in TABLES.items():
        create_table_if_missing(client, DATASET_ID, table_id, schema)

    logger.info("All tables are ready. You can now run the pipeline.")


if __name__ == "__main__":
    main()
