#!/usr/bin/env python3
"""Vertex AI Custom Training Script — MLForecast ensemble.

Reads training data from BigQuery, fits XGB / LGBM / Ridge / RF / ET models
via MLForecast, evaluates each on a hold-out set, selects the champion by WMAPE,
writes predictions + champion metadata back to BigQuery, and saves the champion
model to the AIP_MODEL_DIR directory provided by Vertex AI.

Environment variables (injected automatically by Vertex AI):
    AIP_MODEL_DIR  — GCS URI to write the champion model artifacts.

Training arguments (passed via --args in CustomJob / from_local_script):
    --project-id          GCP project id
    --bq-table            Fully-qualified source table  (project.dataset.table)
    --prediction-bq-table Fully-qualified predictions table
    --champion-bq-table   Fully-qualified champion metadata table
    --model-display-name  Display name prefix for saved model artifacts
    --forecast-freq       Pandas/MLForecast freq string (e.g. "D", "MS")
    --horizon             Number of steps to forecast  (int)
    --lags                Comma-separated lag integers (e.g. "1,2,3")
    --date-features       Comma-separated feature names (e.g. "dayofweek,month")
    --model-types         Comma-separated model keys   (e.g. "lgbm,xgb,rf,et,ridge")
    --champion-metric     Metric to minimise for champion selection (default: wmape)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def build_model(model_type: str):
    if model_type == "lgbm":
        return LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=-1,
        )
    if model_type == "xgb":
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            eval_metric="rmse", verbosity=0,
        )
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )
    if model_type == "et":
        return ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unsupported model_type: {model_type}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--bq-table", required=True)
    parser.add_argument("--prediction-bq-table", required=True)
    parser.add_argument("--champion-bq-table", required=True)
    parser.add_argument("--model-display-name", default="sales-mlforecast")
    parser.add_argument("--forecast-freq", default="D")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--lags", default="1,2,3")
    parser.add_argument("--date-features", default="dayofweek,month")
    parser.add_argument("--model-types", default="lgbm,xgb,rf,et,ridge")
    parser.add_argument("--champion-metric", default="wmape")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    date_features = [x.strip() for x in args.date_features.split(",") if x.strip()]
    model_types = [x.strip() for x in args.model_types.split(",") if x.strip()]

    allowed_date_features = {"year", "month", "week", "day", "dayofweek", "dayofyear", "quarter"}
    date_features = [f for f in date_features if f in allowed_date_features]

    logger.info(
        "Config: project=%s bq_table=%s freq=%s horizon=%d lags=%s "
        "date_features=%s model_types=%s champion_metric=%s",
        args.project_id, args.bq_table, args.forecast_freq, args.horizon,
        lags, date_features, model_types, args.champion_metric,
    )

    # ------------------------------------------------------------------
    # 1. Load data from BigQuery
    # ------------------------------------------------------------------
    client = bigquery.Client(project=args.project_id)
    query = f"""
        SELECT unique_id, ds, y
        FROM `{args.bq_table}`
        WHERE unique_id IS NOT NULL
          AND ds IS NOT NULL
          AND y IS NOT NULL
        ORDER BY unique_id, ds
    """
    logger.info("Fetching training data from %s", args.bq_table)
    df = client.query(query).to_dataframe()
    logger.info("Fetched %d rows", len(df))

    if df.empty:
        raise ValueError("No rows returned from BigQuery.")

    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["unique_id", "ds", "y"]).sort_values(["unique_id", "ds"])
    logger.info("Rows after cleaning: %d", len(df))

    counts = df.groupby("unique_id").size()
    df = df[df["unique_id"].isin(counts[counts > args.horizon].index)]
    logger.info(
        "Rows after short-series filter: %d (series=%d)",
        len(df), df["unique_id"].nunique(),
    )
    if df.empty:
        raise ValueError("No series left after filtering short series.")

    # ------------------------------------------------------------------
    # 2. Train / validation split (hold-out last `horizon` steps)
    # ------------------------------------------------------------------
    train_parts, valid_parts = [], []
    for _, g in df.groupby("unique_id", sort=False):
        train_parts.append(g.iloc[:-args.horizon])
        valid_parts.append(g.iloc[-args.horizon:])
    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    logger.info("Train rows=%d, validation rows=%d", len(train_df), len(valid_df))

    # ------------------------------------------------------------------
    # 3. Train each model and collect metrics
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    all_predictions: list[pd.DataFrame] = []

    for model_type in model_types:
        logger.info("Training model_type=%s", model_type)
        model = build_model(model_type)

        fcst = MLForecast(
            models={model_type: model},
            freq=args.forecast_freq,
            lags=lags,
            date_features=date_features,
            num_threads=1,
        )
        fcst.fit(train_df, id_col="unique_id", time_col="ds", target_col="y")
        preds = fcst.predict(h=args.horizon)

        merged = valid_df.merge(
            preds[["unique_id", "ds", model_type]],
            on=["unique_id", "ds"],
            how="inner",
        ).rename(columns={model_type: "prediction"})

        if merged.empty:
            logger.warning("No overlapping rows for model_type=%s — skipping", model_type)
            continue

        y_true = merged["y"].to_numpy(dtype=float)
        y_pred = merged["prediction"].to_numpy(dtype=float)

        metrics = {
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "wmape": wmape(y_true, y_pred),
        }
        results[model_type] = {"fcst": fcst, "metrics": metrics, "preds": merged}
        logger.info("model_type=%s metrics=%s", model_type, metrics)

        merged = merged.copy()
        merged["run_ts"] = pd.Timestamp.utcnow()
        merged["model_display_name"] = args.model_display_name
        merged["model_type"] = model_type
        merged["source_table"] = args.bq_table
        merged["forecast_freq"] = args.forecast_freq
        merged["horizon"] = args.horizon
        all_predictions.append(merged)

    if not results:
        raise ValueError("All models failed to produce overlapping predictions.")

    # ------------------------------------------------------------------
    # 4. Select champion
    # ------------------------------------------------------------------
    champion_type = min(
        results,
        key=lambda m: results[m]["metrics"].get(args.champion_metric, float("inf")),
    )
    champion_metrics = results[champion_type]["metrics"]
    champion_fcst = results[champion_type]["fcst"]
    logger.info(
        "Champion: model_type=%s %s=%.6f",
        champion_type, args.champion_metric, champion_metrics[args.champion_metric],
    )

    # ------------------------------------------------------------------
    # 5. Write predictions to BigQuery
    # ------------------------------------------------------------------
    preds_df = pd.concat(all_predictions, ignore_index=True)
    for col in ["unique_id", "model_display_name", "model_type", "source_table", "forecast_freq"]:
        preds_df[col] = preds_df[col].astype(str)
    preds_df["y"] = pd.to_numeric(preds_df["y"], errors="coerce")
    preds_df["prediction"] = pd.to_numeric(preds_df["prediction"], errors="coerce")
    preds_df["horizon"] = preds_df["horizon"].astype("int64")

    client.load_table_from_dataframe(
        preds_df[[
            "unique_id", "ds", "y", "prediction", "run_ts",
            "model_display_name", "model_type", "source_table", "forecast_freq", "horizon",
        ]],
        args.prediction_bq_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    logger.info("Wrote %d prediction rows to %s", len(preds_df), args.prediction_bq_table)

    # ------------------------------------------------------------------
    # 6. Write champion metadata to BigQuery
    # ------------------------------------------------------------------
    champion_row = pd.DataFrame([{
        "model_display_name": args.model_display_name,
        "model_type": champion_type,
        "source_table": args.bq_table,
        "prediction_bq_table": args.prediction_bq_table,
        "forecast_freq": args.forecast_freq,
        "horizon": args.horizon,
        "lags": json.dumps(lags),
        "date_features": json.dumps(date_features),
        "series_count": int(df["unique_id"].nunique()),
        "train_rows": int(len(train_df)),
        "rmse": champion_metrics["rmse"],
        "mae": champion_metrics["mae"],
        "wmape": champion_metrics["wmape"],
        "run_ts": pd.Timestamp.utcnow(),
    }])
    client.load_table_from_dataframe(
        champion_row, args.champion_bq_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    logger.info("Wrote champion metadata to %s", args.champion_bq_table)

    # ------------------------------------------------------------------
    # 7. Save champion model to AIP_MODEL_DIR
    # ------------------------------------------------------------------
    model_dir = Path(os.environ.get("AIP_MODEL_DIR", "model_output"))
    if str(model_dir).startswith("gs://"):
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "model"
            local_dir.mkdir()
            champion_fcst.save(local_dir)
            metadata = {
                **champion_metrics,
                "model_type": champion_type,
                "model_display_name": args.model_display_name,
                "champion_metric": args.champion_metric,
            }
            with open(local_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", str(local_dir) + "/", str(model_dir) + "/"],
                check=True,
            )
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        champion_fcst.save(model_dir)
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    **champion_metrics,
                    "model_type": champion_type,
                    "model_display_name": args.model_display_name,
                    "champion_metric": args.champion_metric,
                },
                f, indent=2,
            )

    logger.info("Champion model saved to %s", model_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
