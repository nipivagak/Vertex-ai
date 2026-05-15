import argparse
import json
import re
import time

import ray
import numpy as np
import pandas as pd
from google.cloud import bigquery
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from statsforecast import StatsForecast  # pyright: ignore[reportMissingImports]
from statsforecast.models import (  # pyright: ignore[reportMissingImports]
    ADIDA,
    ARCH,
    AutoARIMA,
    AutoETS,
    AutoTheta,
    CrostonOptimized,
    GARCH,
    MSTL,
    Naive,
    SeasonalNaive,
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse
from xgboost import XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def build_query(project_id, dataset, table):
    return f"""
    WITH base AS (
      SELECT
        CAST({{uid_col}} AS STRING) AS unique_id,
        DATE({{date_col}}) AS ds,
        CAST({{target_col}} AS FLOAT64) AS y
      FROM `{project_id}.{dataset}.{table}`
      WHERE {{uid_col}} IS NOT NULL AND {{date_col}} IS NOT NULL AND {{target_col}} IS NOT NULL
    ),
    parsed AS (
      SELECT SPLIT(unique_id, @uid_delimiter)[SAFE_OFFSET(0)] AS sys_id, unique_id, ds, y
      FROM base
    )
    SELECT unique_id, ds, y, sys_id
    FROM parsed WHERE sys_id = @sys_id_value
    ORDER BY unique_id, ds
    """


def read_data(args):
    client = bigquery.Client(project=args.project_id)
    query = build_query(args.project_id, args.bq_dataset, args.bq_table).format(
        uid_col=args.uid_col,
        date_col=args.date_col,
        target_col=args.target_col,
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("uid_delimiter", "STRING", args.uid_delimiter),
            bigquery.ScalarQueryParameter("sys_id_value", "STRING", args.sys_id_value),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=True)
    if df.empty:
        raise ValueError(f"No rows returned for sys_id={args.sys_id_value}")
    df["ds"] = pd.to_datetime(df["ds"])
    if args.max_series and args.max_series > 0:
        uids = df["unique_id"].unique()[: args.max_series]
        df = df[df["unique_id"].isin(uids)]
        print(f"[max_series] Limiting to {len(uids)} series for testing.")
    return df


def aggregate_monthly(df):
    df = df.copy()
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
    return (
        df.groupby(["unique_id", "ds"], as_index=False)["y"]
        .sum()
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def split_ids_by_history(df, min_history):
    counts = df.groupby("unique_id").size()
    return counts[counts >= min_history].index.tolist(), counts[counts < min_history].index.tolist()


def split_train_valid(df, validation_horizon):
    train_parts, valid_parts = [], []
    for _, grp in df.groupby("unique_id", sort=False):
        grp = grp.sort_values("ds").reset_index(drop=True)
        if len(grp) <= validation_horizon:
            continue
        train_parts.append(grp.iloc[:-validation_horizon].copy())
        valid_parts.append(grp.iloc[-validation_horizon:].copy())
    if not train_parts or not valid_parts:
        raise ValueError("Not enough history to build train/validation split.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def build_stats_models(season_length):
    return [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
        MSTL(season_length=[season_length], trend_forecaster=AutoARIMA(season_length=1)),
        ARCH(),
        GARCH(),
        ADIDA(),
        CrostonOptimized(),
        Naive(),
        SeasonalNaive(season_length=season_length),
    ]


@ray.remote
def _stats_audit_uid_remote(uid_records, horizon, freq, season_length):
    try:
        df = pd.DataFrame(uid_records)
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        sf = StatsForecast(models=build_stats_models(season_length), freq=freq, n_jobs=1)
        preds = sf.forecast(df=df, h=horizon)
        preds["ds"] = pd.to_datetime(preds["ds"])
        return preds.to_dict(orient="records")
    except Exception as e:
        uid = uid_records[0].get("unique_id") if uid_records else "<unknown>"
        print(f"[stats_audit_error] uid={uid} error={e!r}")
        return []


def run_stats_audit(train_df, horizon, freq, season_length):
    futures = [
        _stats_audit_uid_remote.remote(grp.to_dict(orient="records"), horizon, freq, season_length)
        for _, grp in train_df.groupby("unique_id", sort=False)
    ]
    results = ray.get(futures)
    rows = [row for batch in results for row in batch]
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds"])
    out = pd.DataFrame(rows)
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_mlforecast(lags, freq, num_threads=-1):
    models = {
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        ),
    }
    lag_transforms = {
        lags[0]: [RollingMean(window_size=3), RollingMean(window_size=6), RollingMean(window_size=12)],
        lags[-1]: [RollingMean(window_size=3)],
    }
    return MLForecast(
        models=models,
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=["month", "quarter", "year"],
        num_threads=num_threads,
    )


@ray.remote
def _ml_audit_shard_remote(shard_records, horizon, lags, freq):
    try:
        shard_df = pd.DataFrame(shard_records)
        shard_df["ds"] = pd.to_datetime(shard_df["ds"])
        shard_df = shard_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        fcst = build_mlforecast(lags, freq, num_threads=1)
        fcst.fit(shard_df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
        preds = fcst.predict(horizon)
        preds["ds"] = pd.to_datetime(preds["ds"])
        return preds.to_dict(orient="records")
    except Exception as e:
        print(f"[ml_audit_error] shard_failed error={e!r}")
        return []


def _balanced_uid_shards(train_df, n_shards):
    counts = train_df.groupby("unique_id").size().sort_values(ascending=False)
    shards = [[] for _ in range(max(1, n_shards))]
    loads = [0 for _ in range(max(1, n_shards))]
    for uid, cnt in counts.items():
        idx = int(np.argmin(loads))
        shards[idx].append(uid)
        loads[idx] += int(cnt)
    return [s for s in shards if s]


def run_ml_audit_distributed(train_df, horizon, lags, freq, n_shards):
    n_unique = int(train_df["unique_id"].nunique())
    n_shards = max(1, min(int(n_shards), n_unique))
    uid_shards = _balanced_uid_shards(train_df, n_shards)
    futures = []
    shard_row_counts = []
    for shard_uids in uid_shards:
        shard_df = train_df[train_df["unique_id"].isin(shard_uids)].copy()
        shard_row_counts.append(int(len(shard_df)))
        futures.append(
            _ml_audit_shard_remote.remote(
                shard_df.to_dict(orient="records"),
                horizon,
                lags,
                freq,
            )
        )

    print(f"[ml_audit] shard_row_counts={shard_row_counts}")
    results = ray.get(futures)
    rows = [row for batch in results for row in batch]
    if not rows:
        raise ValueError("Distributed ML audit returned no predictions.")
    out = pd.DataFrame(rows)
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def evaluate_predictions(valid_df, preds):
    merged = valid_df.merge(preds, on=["unique_id", "ds"], how="inner")
    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    if not model_cols:
        raise ValueError("No forecast columns found for evaluation.")
    metric_df = evaluate(merged, metrics=[rmse], models=model_cols, id_col="unique_id", target_col="y")
    metric_df = metric_df[metric_df["metric"] == "rmse"].drop(columns=["metric"])
    return (
        metric_df.melt(id_vars=["unique_id"], var_name="model", value_name="rmse")
        .sort_values(["unique_id", "rmse", "model"])
        .reset_index(drop=True)
    )


def build_champions(stats_metrics, ml_metrics, fallback_ids, sys_id_value):
    all_metrics = pd.concat([stats_metrics, ml_metrics], ignore_index=True)
    best = (
        all_metrics.sort_values(["unique_id", "rmse", "model"])
        .groupby("unique_id", as_index=False)
        .first()
        .rename(columns={"model": "best_model_name"})
    )
    best["sys_id"] = sys_id_value
    if fallback_ids:
        fb = pd.DataFrame(
            {
                "unique_id": fallback_ids,
                "best_model_name": "Naive_Fallback",
                "rmse": np.nan,
                "sys_id": sys_id_value,
            }
        )
        best = pd.concat([best, fb], ignore_index=True)
    return best[["sys_id", "unique_id", "best_model_name", "rmse"]].sort_values(["unique_id"]).reset_index(drop=True)


def normalize_model_name(model_name):
    if not isinstance(model_name, str):
        return model_name
    name = model_name.strip()
    if not name:
        return name

    compact = name.replace(" ", "")
    if compact.startswith("ARCH("):
        return "ARCH"
    if compact.startswith("GARCH("):
        return "GARCH"

    aliases = {
        "NaiveFallback": "Naive_Fallback",
    }
    return aliases.get(name, name)


def fit_single_stats_model(model_name, uid_df, horizon, freq, season_length):
    resolved_model_name = normalize_model_name(model_name)
    model_map = {
        "AutoARIMA": AutoARIMA(season_length=season_length),
        "AutoETS": AutoETS(season_length=season_length),
        "AutoTheta": AutoTheta(season_length=season_length),
        "MSTL": MSTL(season_length=[season_length], trend_forecaster=AutoARIMA(season_length=1)),
        "ARCH": ARCH(),
        "GARCH": GARCH(),
        "ADIDA": ADIDA(),
        "CrostonOptimized": CrostonOptimized(),
        "Naive": Naive(),
        "SeasonalNaive": SeasonalNaive(season_length=season_length),
        "Naive_Fallback": Naive(),
    }
    if resolved_model_name not in model_map:
        supported = sorted(model_map.keys())
        raise ValueError(
            f"Unsupported stats model: {model_name} (resolved={resolved_model_name}); supported={supported}"
        )
    sf = StatsForecast(models=[model_map[resolved_model_name]], freq=freq, n_jobs=1)
    pred = sf.forecast(df=uid_df, h=horizon)
    pred["ds"] = pd.to_datetime(pred["ds"])
    forecast_col = [c for c in pred.columns if c not in {"unique_id", "ds"}][0]
    return pred.rename(columns={forecast_col: "Forecast"})[["unique_id", "ds", "Forecast"]]


def fit_single_ml_model(model_name, uid_df, horizon, lags, freq):
    if model_name == "XGBoost":
        models = {
            "fcst_temp": XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=1,
            )
        }
    elif model_name == "LightGBM":
        models = {
            "fcst_temp": LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            )
        }
    else:
        raise ValueError(f"Unsupported ML model: {model_name}")
    fcst = MLForecast(
        models=models,
        freq=freq,
        lags=lags,
        lag_transforms={
            lags[0]: [RollingMean(window_size=3), RollingMean(window_size=6), RollingMean(window_size=12)],
            lags[-1]: [RollingMean(window_size=3)],
        },
        date_features=["month", "quarter", "year"],
        num_threads=1,
    )
    fcst.fit(uid_df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    pred = fcst.predict(horizon)
    pred["ds"] = pd.to_datetime(pred["ds"])
    return pred.rename(columns={"fcst_temp": "Forecast"})[["unique_id", "ds", "Forecast"]]


@ray.remote
def _forecast_uid_remote(uid, grp_records, best_model, horizon, freq, season_length, sys_id_value, lags):
    ml_models = {"XGBoost", "LightGBM"}
    try:
        grp = pd.DataFrame(grp_records)
        grp["ds"] = pd.to_datetime(grp["ds"])
        grp = grp.sort_values("ds").reset_index(drop=True)
        resolved_model = normalize_model_name(best_model)
        if resolved_model in ml_models:
            pred = fit_single_ml_model(resolved_model, grp, horizon, lags, freq)
        else:
            pred = fit_single_stats_model(resolved_model, grp, horizon, freq, season_length)
        pred["sys_id"] = sys_id_value
        pred["Best_model_name"] = best_model
        return pred[["sys_id", "unique_id", "ds", "Forecast", "Best_model_name"]].to_dict(orient="records")
    except Exception as e:
        print(f"[forecast_error] uid={uid} model={best_model} error={e!r}")
        return []


def refit_and_forecast(full_df, champions_df, horizon, freq, season_length, sys_id_value, lags):
    champ_map = dict(zip(champions_df["unique_id"], champions_df["best_model_name"]))
    futures = []
    for uid, grp in full_df.groupby("unique_id", sort=False):
        best_model = champ_map.get(uid)
        if best_model is None:
            continue
        futures.append(
            _forecast_uid_remote.remote(
                uid,
                grp.to_dict(orient="records"),
                best_model,
                horizon,
                freq,
                season_length,
                sys_id_value,
                lags,
            )
        )
    results = ray.get(futures)
    rows = [row for batch in results for row in batch]
    if not rows:
        return pd.DataFrame(columns=["sys_id", "unique_id", "ds", "Forecast", "Best_model_name", "created_at"])
    out = pd.DataFrame(rows)
    out["ds"] = pd.to_datetime(out["ds"])
    out["created_at"] = utc_now()
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_metrics_output(stats_metrics, ml_metrics, champions_df):
    all_metrics = pd.concat([stats_metrics, ml_metrics], ignore_index=True)
    if all_metrics.empty:
        return pd.DataFrame()
    wide = all_metrics.pivot_table(index="unique_id", columns="model", values="rmse", aggfunc="first")
    wide.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", f"{col}_RMSE") for col in wide.columns]
    wide = wide.reset_index()
    best = champions_df[["unique_id", "best_model_name", "rmse"]].rename(
        columns={"best_model_name": "Best_model_name", "rmse": "Best_model_RMSE"}
    )
    wide = wide.merge(best, on="unique_id", how="left")
    wide["created_at"] = utc_now()
    return wide


def write_bq(df, table_fqn, project_id, write_disposition="WRITE_APPEND"):
    client = bigquery.Client(project=project_id)
    df = df.drop(columns=["sys_id"], errors="ignore")
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    if write_disposition == "WRITE_APPEND":
        job_config.schema_update_options = [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    client.load_table_from_dataframe(df, table_fqn, job_config=job_config).result()


def write_run_logs(df, table_fqn, project_id):
    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job_config.schema_update_options = [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    client.load_table_from_dataframe(df, table_fqn, job_config=job_config).result()


def main():
    args = parse_args()
    ml_lags = [int(x) for x in args.ml_lags.split(",")]

    run_started_at = utc_now()

    ray.init(address="auto", ignore_reinit_error=True, logging_level="WARNING")
    cluster_resources = ray.cluster_resources()
    print(f"[ray] cluster_resources={cluster_resources}")
    total_cpus = int(cluster_resources.get("CPU", 1))
    print(f"[ray] total_cpus_in_cluster={total_cpus}")

    t0 = time.perf_counter()
    raw_df = read_data(args)
    full_df = aggregate_monthly(raw_df)
    data_read_seconds = round(time.perf_counter() - t0, 3)

    eligible_ids, fallback_ids = split_ids_by_history(full_df, args.min_history_for_full_audit)
    eligible_df = full_df[full_df["unique_id"].isin(eligible_ids)].copy()

    stats_metrics = pd.DataFrame(columns=["unique_id", "model", "rmse"])
    ml_metrics = pd.DataFrame(columns=["unique_id", "model", "rmse"])
    ml_shards_used = 0
    stats_audit_seconds = 0.0
    ml_audit_seconds = 0.0

    if not eligible_df.empty:
        train_df, valid_df = split_train_valid(eligible_df, args.validation_horizon)

        n_series = eligible_df["unique_id"].nunique()
        print(f"[audit] running stats audit via Ray ({n_series} series x 10 models, ~{total_cpus} parallel slots)")
        t_stats = time.perf_counter()
        stats_preds = run_stats_audit(train_df, args.validation_horizon, args.freq, args.season_length)
        stats_audit_seconds = round(time.perf_counter() - t_stats, 3)

        auto_shards = max(1, min(total_cpus // 2 if total_cpus > 1 else 1, n_series))
        requested_ml_shards = args.ml_audit_shards if args.ml_audit_shards and args.ml_audit_shards > 0 else auto_shards
        ml_shards_used = max(1, min(int(requested_ml_shards), int(n_series)))
        print(f"[audit] running distributed ml audit via Ray (requested_shards={requested_ml_shards}, used_shards={ml_shards_used})")
        t_ml = time.perf_counter()
        ml_preds = run_ml_audit_distributed(train_df, args.validation_horizon, ml_lags, args.freq, ml_shards_used)
        ml_audit_seconds = round(time.perf_counter() - t_ml, 3)

        stats_metrics = evaluate_predictions(valid_df, stats_preds)
        ml_metrics = evaluate_predictions(valid_df, ml_preds)

    champions_df = build_champions(stats_metrics, ml_metrics, fallback_ids, args.sys_id_value)
    metrics_df = build_metrics_output(stats_metrics, ml_metrics, champions_df)

    forecast_df = refit_and_forecast(
        full_df,
        champions_df,
        args.horizon,
        args.freq,
        args.season_length,
        args.sys_id_value,
        ml_lags,
    )

    forecast_table_fqn = f"{args.project_id}.{args.output_dataset}.{args.forecast_table}"
    metrics_table_fqn = f"{args.project_id}.{args.output_dataset}.{args.metrics_table}"
    champions_table_fqn = f"{args.project_id}.{args.output_dataset}.{args.champions_table}"
    run_logs_table_fqn = f"{args.project_id}.{args.output_dataset}.{args.run_logs_table}"

    t_bq = time.perf_counter()
    if not metrics_df.empty:
        write_bq(metrics_df, metrics_table_fqn, args.project_id, write_disposition="WRITE_TRUNCATE")
    write_bq(champions_df, champions_table_fqn, args.project_id)
    write_bq(forecast_df, forecast_table_fqn, args.project_id)
    bq_write_seconds = round(time.perf_counter() - t_bq, 3)

    run_finished_at = utc_now()

    summary = {
        "sys_id": args.sys_id_value,
        "max_series": args.max_series,
        "series_count": int(full_df["unique_id"].nunique()),
        "eligible_ids_for_full_audit": len(eligible_ids),
        "fallback_ids": len(fallback_ids),
        "ml_audit_shards_used": int(ml_shards_used),
        "metrics_rows": int(len(metrics_df)),
        "champion_rows": int(len(champions_df)),
        "forecast_rows": int(len(forecast_df)),
        "data_read_seconds": data_read_seconds,
        "stats_audit_seconds": stats_audit_seconds,
        "ml_audit_seconds": ml_audit_seconds,
        "bq_write_seconds": bq_write_seconds,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "run_duration_seconds": round((run_finished_at - run_started_at).total_seconds(), 3),
    }

    run_logs_df = pd.DataFrame([summary])
    run_logs_df["created_at"] = utc_now()
    write_run_logs(run_logs_df, run_logs_table_fqn, args.project_id)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
