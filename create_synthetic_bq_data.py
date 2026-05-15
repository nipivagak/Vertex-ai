#!/usr/bin/env python3
"""Create synthetic forecasting data in BigQuery.

This script creates a table with the required columns for the notebook pipeline:
- unique_id (STRING): combination of sys_id and account_id
- date (DATE)
- target (FLOAT64)

It also includes sys_id and account_id as helper columns.

Defaults are chosen to produce about 1M rows over 2 years:
- 21 sys_ids
- 66 account_ids
- daily data for 730 days (~1,011,780 rows)
"""

from __future__ import annotations

import argparse
import datetime as dt
import re

from google.cloud import bigquery


def _validate_identifier(name: str, kind: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Invalid {kind} '{name}'. Use only letters, numbers, and underscores.")
    return name


def _parse_args() -> argparse.Namespace:
    today = dt.date.today()
    default_end = today - dt.timedelta(days=1)
    default_start = default_end - dt.timedelta(days=729)

    parser = argparse.ArgumentParser(description="Create synthetic BigQuery table for forecasting.")
    parser.add_argument("--project-id", required=True, help="GCP project id")
    parser.add_argument("--dataset-id", default="forecasting", help="BigQuery dataset id")
    parser.add_argument("--table-name", default="sales_daily_synth", help="BigQuery table name")
    parser.add_argument("--region", default="US", help="BigQuery dataset region (e.g. US, us-central1)")
    parser.add_argument("--start-date", default=str(default_start), help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=str(default_end), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--num-accounts",
        type=int,
        default=66,
        help="Number of account_ids per sys_id (default 66 gives ~1M rows over 2 years)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_id = _validate_identifier(args.dataset_id, "dataset_id")
    table_name = _validate_identifier(args.table_name, "table_name")

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")
    if args.num_accounts < 1:
        raise ValueError("num-accounts must be >= 1")

    num_days = (end_date - start_date).days + 1
    sys_count = 21
    expected_rows = sys_count * args.num_accounts * num_days

    client = bigquery.Client(project=args.project_id)

    dataset_sql = (
        f"CREATE SCHEMA IF NOT EXISTS `{args.project_id}.{dataset_id}` "
        f"OPTIONS(location='{args.region}')"
    )
    client.query(dataset_sql).result()

    table_fqn = f"{args.project_id}.{dataset_id}.{table_name}"
    create_sql = f"""
    CREATE OR REPLACE TABLE `{table_fqn}` AS
    WITH
      sys AS (
        SELECT sys_id
        FROM UNNEST(GENERATE_ARRAY(1, 21)) AS sys_id
      ),
      accounts AS (
        SELECT account_id
        FROM UNNEST(GENERATE_ARRAY(1, @num_accounts)) AS account_id
      ),
      dates AS (
        SELECT d AS date
        FROM UNNEST(GENERATE_DATE_ARRAY(@start_date, @end_date, INTERVAL 1 DAY)) AS d
      )
    SELECT
      sys_id,
      account_id,
      FORMAT('SYS%02d_ACC%04d', sys_id, account_id) AS unique_id,
      date,
      ROUND(
        GREATEST(
          0,
          100
          + 2.5 * sys_id
          + 1.1 * MOD(account_id, 17)
          + 0.03 * DATE_DIFF(date, @start_date, DAY)
          + 18 * COS(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM date) / 365.25)
          + 6 * COS(2 * ACOS(-1) * EXTRACT(DAYOFWEEK FROM date) / 7)
          + RAND() * 8
        ),
        2
      ) AS target
    FROM sys
    CROSS JOIN accounts
    CROSS JOIN dates
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("num_accounts", "INT64", args.num_accounts),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )

    client.query(create_sql, job_config=job_config).result()

    count_sql = f"SELECT COUNT(*) AS row_count FROM `{table_fqn}`"
    row_count = next(client.query(count_sql).result())["row_count"]

    print("Synthetic table created successfully")
    print(f"Table: {table_fqn}")
    print(f"Date range: {start_date} to {end_date} ({num_days} days)")
    print(f"sys_ids: {sys_count}, account_ids per sys_id: {args.num_accounts}")
    print(f"Expected rows: {expected_rows:,}")
    print(f"Actual rows:   {row_count:,}")


if __name__ == "__main__":
    main()
