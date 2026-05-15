import argparse
import datetime
import shlex

import google.auth
import google.auth.transport.requests
from ray.job_submission import JobSubmissionClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_resource_name", required=True)
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--bq_dataset", required=True)
    parser.add_argument("--bq_table", required=True)
    parser.add_argument("--uid_col", required=True)
    parser.add_argument("--date_col", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--sys_id_value", required=True)
    parser.add_argument("--uid_delimiter", required=True)
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
    return parser.parse_args()


def get_client(cluster_resource_name: str) -> JobSubmissionClient:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    return JobSubmissionClient(
        f"vertex_ray://{cluster_resource_name}",
        headers={"Authorization": f"Bearer {credentials.token}"},
    )


def main():
    args = parse_args()
    client = get_client(args.cluster_resource_name)

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
    print(job_id)


if __name__ == "__main__":
    main()
