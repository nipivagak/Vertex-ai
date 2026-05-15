"""
Vertex AI Pipeline (KFP) wrapper for Nixtla Ray forecasting pipeline.

Compiles to YAML and submits to Vertex AI Pipelines.

Usage:
    # Compile to YAML
    python nixtla_vertex_ray_pipeline.py

    # Or submit directly
    python nixtla_vertex_ray_pipeline.py --submit \
      --project_id YOUR_PROJECT \
      --region us-central1 \
      --display_name my-forecast-run
"""

import argparse
import os
import subprocess
import sys
from typing import Optional

from kfp import dsl, compiler
from kfp.dsl import Input, Output, Artifact


@dsl.container_component
def run_ray_forecast(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    uid_col: str,
    date_col: str,
    target_col: str,
    sys_id_value: str,
    output_dataset: str,
    forecast_table: str,
    metrics_table: str,
    champions_table: str,
    run_logs_table: str,
    cluster_name: str = "nixtla-forecast-ray-cluster",
    region: str = "us-central1",
    horizon: int = 12,
    season_length: int = 12,
    validation_horizon: int = 12,
    worker_node_count: int = 4,
    head_machine_type: str = "n1-standard-16",
    worker_machine_type: str = "n1-standard-16",
    head_custom_image: str = "",
    worker_custom_image: str = "",
    delete_cluster_after_run: bool = False,
) -> dsl.ContainerSpec:
    """
    Submits Ray forecasting job to Vertex AI Ray cluster.
    
    Args:
        project_id: GCP project ID
        bq_dataset: Source BigQuery dataset
        bq_table: Source BigQuery table
        uid_col: Unique ID column name
        date_col: Date/timestamp column name
        target_col: Target value column name
        sys_id_value: System/partition identifier
        output_dataset: Output BigQuery dataset
        forecast_table: Output forecast table name
        metrics_table: Output metrics table name
        champions_table: Output champions table name
        run_logs_table: Output run logs table name
        cluster_name: Ray cluster name (reused if exists)
        region: GCP region for Ray cluster
        horizon: Forecast horizon in periods
        season_length: Seasonal period
        validation_horizon: Validation horizon
        worker_node_count: Number of Ray worker nodes
        head_machine_type: Machine type for head node
        worker_machine_type: Machine type for workers
        head_custom_image: Artifact Registry image URI for head node
        worker_custom_image: Artifact Registry image URI for worker nodes
        delete_cluster_after_run: Delete cluster after job completes
    """
    return dsl.ContainerSpec(
        image="python:3.11",
        command=[
            "sh",
            "-c",
        ],
        args=[
            f"""
pip install -q \
    google-cloud-aiplatform \
    google-cloud-bigquery \
    ray[client] \
    pandas \
    pyarrow \
    db-dtypes \
    statsforecast \
    mlforecast \
    utilsforecast \
    lightgbm \
    xgboost \
    scikit-learn

HEAD_IMAGE_ARG=""
if [ -n "{head_custom_image}" ]; then
    HEAD_IMAGE_ARG="--head_custom_image {head_custom_image}"
fi

WORKER_IMAGE_ARG=""
if [ -n "{worker_custom_image}" ]; then
    WORKER_IMAGE_ARG="--worker_custom_image {worker_custom_image}"
fi

python run_ray_pipeline.py \
    --project_id {project_id} \
    --region {region} \
    --cluster_name {cluster_name} \
    --reuse_existing_cluster \
    --head_machine_type {head_machine_type} \
    --worker_machine_type {worker_machine_type} \
    --worker_node_count {worker_node_count} \
    --bq_dataset {bq_dataset} \
    --bq_table {bq_table} \
    --uid_col {uid_col} \
    --date_col {date_col} \
    --target_col {target_col} \
    --sys_id_value {sys_id_value} \
    --output_dataset {output_dataset} \
    --forecast_table {forecast_table} \
    --metrics_table {metrics_table} \
    --champions_table {champions_table} \
    --run_logs_table {run_logs_table} \
    --horizon {horizon} \
    --season_length {season_length} \
    --validation_horizon {validation_horizon} \
    $HEAD_IMAGE_ARG \
    $WORKER_IMAGE_ARG \
    {'--delete_cluster_after_run' if delete_cluster_after_run else ''}
""",
        ],
    )


@dsl.pipeline(
    name="nixtla-ray-forecast-pipeline",
    description="Nixtla MLForecast + StatsForecast on Vertex AI Ray cluster",
)
def nixtla_forecast_pipeline(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    uid_col: str,
    date_col: str,
    target_col: str,
    sys_id_value: str,
    output_dataset: str,
    forecast_table: str,
    metrics_table: str,
    champions_table: str,
    run_logs_table: str,
    cluster_name: str = "nixtla-forecast-ray-cluster",
    region: str = "us-central1",
    horizon: int = 12,
    season_length: int = 12,
    validation_horizon: int = 12,
    worker_node_count: int = 4,
    head_machine_type: str = "n1-standard-16",
    worker_machine_type: str = "n1-standard-16",
    head_custom_image: str = "",
    worker_custom_image: str = "",
    delete_cluster_after_run: bool = False,
):
    """Pipeline DAG for Nixtla forecasting."""
    run_ray_forecast(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        uid_col=uid_col,
        date_col=date_col,
        target_col=target_col,
        sys_id_value=sys_id_value,
        output_dataset=output_dataset,
        forecast_table=forecast_table,
        metrics_table=metrics_table,
        champions_table=champions_table,
        run_logs_table=run_logs_table,
        cluster_name=cluster_name,
        region=region,
        horizon=horizon,
        season_length=season_length,
        validation_horizon=validation_horizon,
        worker_node_count=worker_node_count,
        head_machine_type=head_machine_type,
        worker_machine_type=worker_machine_type,
        head_custom_image=head_custom_image,
        worker_custom_image=worker_custom_image,
        delete_cluster_after_run=delete_cluster_after_run,
    )


def compile_pipeline(output_file: str = "nixtla_vertex_ray_pipeline.yaml"):
    """Compile pipeline to YAML."""
    compiler.Compiler().compile(
        pipeline_func=nixtla_forecast_pipeline,
        package_path=output_file,
    )
    print(f"✓ Pipeline compiled to {output_file}")


def submit_pipeline(
    project_id: str,
    region: str,
    display_name: str,
    pipeline_yaml: str = "nixtla_vertex_ray_pipeline.yaml",
    enable_caching: bool = False,
):
    """Submit compiled pipeline to Vertex AI."""
    if not os.path.exists(pipeline_yaml):
        print(f"✗ Pipeline YAML not found: {pipeline_yaml}")
        print(f"  Run: python nixtla_vertex_ray_pipeline.py")
        sys.exit(1)

    cmd = [
        "gcloud",
        "ai",
        "pipelines",
        "runs",
        "submit",
        "--project",
        project_id,
        "--region",
        region,
        "--pipeline-root",
        f"gs://{project_id}-vertex-pipelines",
        "--display-name",
        display_name,
        "--template-path",
        pipeline_yaml,
    ]

    if not enable_caching:
        cmd.append("--disable-cache")

    print(f"Submitting pipeline to {project_id}/{region}...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and submit Nixtla Ray forecasting pipeline to Vertex AI",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Compile pipeline to YAML (default: True)",
    )
    parser.add_argument(
        "--output",
        default="nixtla_vertex_ray_pipeline.yaml",
        help="Output YAML filename (default: nixtla_vertex_ray_pipeline.yaml)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit compiled pipeline to Vertex AI Pipelines",
    )
    parser.add_argument(
        "--project_id",
        default="dazzling-seat-366014",
        help="GCP project ID",
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="GCP region",
    )
    parser.add_argument(
        "--display_name",
        help="Display name for pipeline run",
    )
    parser.add_argument(
        "--enable_caching",
        action="store_true",
        help="Enable pipeline caching (default: False)",
    )

    args = parser.parse_args()

    # Always compile first
    compile_pipeline(args.output)

    # Optionally submit
    if args.submit:
        if not args.display_name:
            import datetime
            args.display_name = f"nixtla-forecast-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        submit_pipeline(
            project_id=args.project_id,
            region=args.region,
            display_name=args.display_name,
            pipeline_yaml=args.output,
            enable_caching=args.enable_caching,
        )
