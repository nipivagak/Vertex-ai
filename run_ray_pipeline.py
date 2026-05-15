import argparse
import datetime
import logging
import shlex
import time

import google.auth
import google.auth.transport.requests
from ray.job_submission import JobStatus, JobSubmissionClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_ray_pipeline")

TERMINAL_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Vertex Ray forecasting pipeline.")

    parser.add_argument("--project_id", default="dazzling-seat-366014")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--staging_bucket", default="gs://dazzling-seat-366014-vertex-pipelines")
    parser.add_argument("--service_account", default="278930531128-compute@developer.gserviceaccount.com")
    parser.add_argument("--network", default=None)

    parser.add_argument("--cluster_name", default="nixtla-forecast-ray-cluster")
    parser.add_argument(
        "--reuse_existing_cluster",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--delete_cluster_after_run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--head_machine_type", default="n1-standard-16")
    parser.add_argument("--worker_machine_type", default="n1-standard-16")
    parser.add_argument("--head_custom_image", default=None)
    parser.add_argument("--worker_custom_image", default=None)
    parser.add_argument("--worker_node_count", type=int, default=4)
    parser.add_argument("--boot_disk_size_gb", type=int, default=200)
    parser.add_argument("--ray_version", default="2.47")
    parser.add_argument("--python_version", default="3.11")

    parser.add_argument("--bq_dataset", required=True)
    parser.add_argument("--bq_table", required=True)
    parser.add_argument("--uid_col", required=True)
    parser.add_argument("--date_col", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--sys_id_value", required=True)
    parser.add_argument("--uid_delimiter", default="_")
    parser.add_argument("--freq", default="MS")
    parser.add_argument("--season_length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--validation_horizon", type=int, default=12)
    parser.add_argument("--min_history_for_full_audit", type=int, default=24)
    parser.add_argument("--max_series", type=int, default=0)
    parser.add_argument("--ml_lags", default="1,12")
    parser.add_argument("--ml_audit_shards", type=int, default=0)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--forecast_table", required=True)
    parser.add_argument("--metrics_table", required=True)
    parser.add_argument("--champions_table", required=True)
    parser.add_argument("--run_logs_table", required=True)
    parser.add_argument("--working_dir", default="ray_job")
    parser.add_argument("--script_name", default="train_nixtla_ray_cluster.py")
    parser.add_argument("--token_refresh_minutes", type=int, default=45)

    return parser.parse_args()


def ensure_cluster(args):
    from google.cloud.aiplatform import vertex_ray
    from google.cloud.aiplatform.vertex_ray import Resources

    def find_existing_cluster(name: str):
        for cluster in vertex_ray.list_ray_clusters():
            if cluster.cluster_resource_name.rsplit("/", 1)[-1] == name:
                return cluster.cluster_resource_name
        return None

    head_node = Resources(
        machine_type=args.head_machine_type,
        node_count=1,
        boot_disk_size_gb=args.boot_disk_size_gb,
        custom_image=args.head_custom_image,
    )
    worker_nodes = [
        Resources(
            machine_type=args.worker_machine_type,
            node_count=args.worker_node_count,
            boot_disk_size_gb=args.boot_disk_size_gb,
            custom_image=args.worker_custom_image,
        )
    ]

    cluster_resource_name = None
    if args.reuse_existing_cluster:
        cluster_resource_name = find_existing_cluster(args.cluster_name)
        if cluster_resource_name:
            logger.info("Reusing existing cluster: %s", cluster_resource_name)

    if cluster_resource_name is None:
        logger.info("Creating new cluster: %s", args.cluster_name)
        cluster_resource_name = vertex_ray.create_ray_cluster(
            head_node_type=head_node,
            worker_node_types=worker_nodes,
            python_version=args.python_version,
            ray_version=args.ray_version,
            cluster_name=args.cluster_name,
            network=args.network,
            service_account=args.service_account,
        )

    logger.info("cluster_resource_name=%s", cluster_resource_name)
    return cluster_resource_name


def get_ray_client(cluster_resource_name: str) -> JobSubmissionClient:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    return JobSubmissionClient(
        f"vertex_ray://{cluster_resource_name}",
        headers={"Authorization": f"Bearer {credentials.token}"},
    )


def build_entrypoint(args):
    q = shlex.quote
    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    submission_id = f"nixtla-{args.sys_id_value.lower().replace('_', '-')}-{run_ts}"
    entrypoint = (
        f"python {q(args.script_name)}"
        f" --project_id {q(args.project_id)}"
        f" --bq_dataset {q(args.bq_dataset)}"
        f" --bq_table {q(args.bq_table)}"
        f" --uid_col {q(args.uid_col)}"
        f" --date_col {q(args.date_col)}"
        f" --target_col {q(args.target_col)}"
        f" --sys_id_value {q(args.sys_id_value)}"
        f" --uid_delimiter {q(args.uid_delimiter)}"
        f" --freq {q(args.freq)}"
        f" --season_length {args.season_length}"
        f" --horizon {args.horizon}"
        f" --validation_horizon {args.validation_horizon}"
        f" --min_history_for_full_audit {args.min_history_for_full_audit}"
        f" --max_series {args.max_series}"
        f" --ml_lags {q(args.ml_lags)}"
        f" --ml_audit_shards {args.ml_audit_shards}"
        f" --output_dataset {q(args.output_dataset)}"
        f" --forecast_table {q(args.forecast_table)}"
        f" --metrics_table {q(args.metrics_table)}"
        f" --champions_table {q(args.champions_table)}"
        f" --run_logs_table {q(args.run_logs_table)}"
    )
    return submission_id, entrypoint


def submit_job(args, cluster_resource_name: str):
    client = get_ray_client(cluster_resource_name)
    submission_id, entrypoint = build_entrypoint(args)

    job_id = client.submit_job(
        entrypoint=entrypoint,
        submission_id=submission_id,
        runtime_env={
            "working_dir": args.working_dir,
            "pip": [
                "immutabledict",
                "google-cloud-aiplatform",
                "google-cloud-bigquery",
                "pandas",
                "pyarrow",
                "db-dtypes",
                "statsforecast",
                "mlforecast",
                "utilsforecast",
                "lightgbm",
                "xgboost",
                "scikit-learn",
            ],
        },
    )
    logger.info("Submitted Ray job: %s", job_id)
    return client, job_id


def tail_job_logs(
    client: JobSubmissionClient,
    job_id: str,
    cluster_resource_name: str,
    refresh_minutes: int,
):
    last_logs = ""
    last_refresh = time.time()
    refresh_interval_seconds = max(1, refresh_minutes) * 60

    while True:
        if time.time() - last_refresh > refresh_interval_seconds:
            client = get_ray_client(cluster_resource_name)
            last_refresh = time.time()

        status = client.get_job_status(job_id)
        logs = client.get_job_logs(job_id)
        new_logs = logs[len(last_logs):]
        if new_logs:
            print(new_logs, end="")
            last_logs = logs
        if status in TERMINAL_STATUSES:
            print(f"\n[done] status={status}")
            return status
        time.sleep(5)


def maybe_delete_cluster(args, cluster_resource_name: str):
    if not args.delete_cluster_after_run:
        logger.info("Skipping cluster teardown: %s", cluster_resource_name)
        return

    from google.cloud.aiplatform import vertex_ray

    vertex_ray.delete_ray_cluster(cluster_resource_name)
    logger.info("Deleted cluster: %s", cluster_resource_name)


def main():
    args = parse_args()
    cluster_resource_name = ensure_cluster(args)
    client, job_id = submit_job(args, cluster_resource_name)
    status = tail_job_logs(client, job_id, cluster_resource_name, args.token_refresh_minutes)
    logger.info("Ray job finished with status: %s", status)
    maybe_delete_cluster(args, cluster_resource_name)


if __name__ == "__main__":
    main()
