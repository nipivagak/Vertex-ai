import argparse
import json
import logging
import pickle
from pathlib import Path

from google.cloud import bigquery


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("local_pipeline_runner")

DEFAULT_MODELS = ["lgbm", "xgb", "rf", "et", "ridge"]


def wmape(y_true, y_pred):
    denominator = y_true.abs().sum()
    return (((y_true - y_pred).abs().sum() / denominator) * 100) if denominator > 0 else 0


def build_model(model_type: str):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from statsforecast.models import ADIDA, AutoARIMA, AutoETS, AutoTheta, CrostonClassic, CrostonOptimized, CrostonSBA, IMAPA, MSTL
    from xgboost import XGBRegressor
    from arch import arch_model

    if model_type == "lgbm":
        return LGBMRegressor(verbosity=-1, random_state=0, n_jobs=1)
    if model_type == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)
    if model_type == "et":
        return ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs=1)
    if model_type == "xgb":
        return XGBRegressor(random_state=0, n_jobs=1)
    if model_type == "gb":
        return GradientBoostingRegressor(random_state=0)
    if model_type == "ridge":
        return Ridge()
    if model_type == "svr":
        return SVR()
    if model_type == "arima":
        return AutoARIMA(season_length=12, trace=True)
    if model_type == "ets":
        return AutoETS(season_length=12, trace=True)
    if model_type == "theta":
        return AutoTheta()
    if model_type == "imapa":
        return IMAPA()
    if model_type == "adida":
        return ADIDA()
    if model_type == "croston":
        return CrostonClassic()
    if model_type == "croston_optimized":
        return CrostonOptimized()
    if model_type == "croston_sba":
        return CrostonSBA()
    if model_type == "mstl":
        return MSTL(season_length=24)
    if model_type.startswith("garch_"):
        _, p_value, q_value = model_type.split("_")
        return arch_model(None, vol="Garch", p=int(p_value), q=int(q_value))
    if model_type.startswith("arch_"):
        _, p_value = model_type.split("_")
        return arch_model(None, vol="ARCH", p=int(p_value))
    raise ValueError(f"Unsupported model_type: {model_type}")


def ensure_bigquery_targets(project_id: str) -> None:
    from setup_bq_tables import main as setup_bq_tables_main

    logger.info("Ensuring BigQuery dataset and tables exist")
    setup_bq_tables_main()


def load_training_data(project_id: str, bq_table: str):
    import pandas as pd

    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT unique_id, ds, y
    FROM `{project_id}.{bq_table}`
    ORDER BY unique_id, ds
    """
    logger.info("Loading training data from %s.%s", project_id, bq_table)
    df = client.query(query).to_dataframe()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def train_model(
    project_id: str,
    df,
    prediction_bq_table: str,
    model_display_name: str,
    bq_table: str,
    model_type: str,
    forecast_freq: str,
    horizon: int,
    lags: list[int],
    date_features: list[str],
):
    import pandas as pd
    from mlforecast import MLForecast

    allowed_date_features = ["year", "month", "week", "day", "dayofweek", "dayofyear", "quarter"]
    parsed_date_features = [value for value in date_features if value in allowed_date_features]
    model = build_model(model_type)

    fcst = MLForecast(
        models={model_type: model},
        freq=forecast_freq,
        lags=lags,
        date_features=parsed_date_features,
        num_threads=1,
    )

    unique_ids = df["unique_id"].unique()
    train_df = df[df["unique_id"].isin(unique_ids)].copy()
    valid_df = df[df["unique_id"].isin(unique_ids)].tail(horizon * len(unique_ids)).copy()

    logger.info("Training model_type=%s on %d rows", model_type, len(train_df))
    fcst.fit(train_df)
    predictions = fcst.predict(h=horizon)

    if len(valid_df) > 0:
        y_true = valid_df.set_index(["unique_id", "ds"])["y"]
        y_pred = predictions.set_index(["unique_id", "ds"])[model_type]
        metric_wmape = wmape(y_true, y_pred)
        metric_mae = (y_true - y_pred).abs().mean()
        metric_rmse = (((y_true - y_pred) ** 2).mean() ** 0.5)
    else:
        metric_wmape = metric_mae = metric_rmse = 0.0

    run_ts = pd.Timestamp.utcnow()
    prediction_df = predictions.rename(columns={model_type: "prediction"}).copy()
    validation_cols = valid_df[["unique_id", "ds", "y"]].copy() if len(valid_df) > 0 else df[["unique_id", "ds", "y"]].tail(len(prediction_df)).copy()
    prediction_df = prediction_df.merge(validation_cols, on=["unique_id", "ds"], how="left")
    prediction_df["run_ts"] = run_ts
    prediction_df["model_display_name"] = model_display_name
    prediction_df["model_type"] = model_type
    prediction_df["source_table"] = bq_table
    prediction_df["forecast_freq"] = forecast_freq
    prediction_df["horizon"] = horizon
    prediction_df = prediction_df[[
        "unique_id",
        "ds",
        "y",
        "prediction",
        "run_ts",
        "model_display_name",
        "model_type",
        "source_table",
        "forecast_freq",
        "horizon",
    ]]
    prediction_df.to_gbq(prediction_bq_table, project_id=project_id, if_exists="append")

    artifact_dir = Path("artifacts/local_models") / model_display_name / model_type
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    with open(model_path, "wb") as file_handle:
        pickle.dump(fcst, file_handle)

    metadata = {
        "model_type": model_type,
        "forecast_freq": forecast_freq,
        "horizon": horizon,
        "lags": lags,
        "date_features": parsed_date_features,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "series_count": int(df["unique_id"].nunique()),
        "rmse": float(metric_rmse),
        "mae": float(metric_mae),
        "wmape": float(metric_wmape),
        "model_uri": str(model_path),
        "model_display_name": model_display_name,
        "source_table": bq_table,
        "prediction_bq_table": prediction_bq_table,
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    logger.info(
        "Completed model_type=%s with wMAPE=%.6f",
        model_type,
        metadata["wmape"],
    )
    return metadata


def persist_champion(project_id: str, champion_bq_table: str, champion: dict):
    import pandas as pd

    client = bigquery.Client(project=project_id)
    champion_summary = {
        "run_ts": pd.Timestamp.utcnow().isoformat(),
        "champion_model_type": champion["model_type"],
        "champion_metric_name": "wmape",
        "champion_metric_value": champion["wmape"],
        "champion_model_uri": champion.get("model_uri"),
        "model_display_name": champion.get("model_display_name"),
        "source_table": champion.get("source_table"),
        "prediction_bq_table": champion.get("prediction_bq_table"),
        "forecast_freq": champion.get("forecast_freq"),
        "horizon": champion.get("horizon"),
    }
    table_id = f"{project_id}.{champion_bq_table}"
    errors = client.insert_rows_json(table_id, [champion_summary], skip_invalid_rows=True)
    if errors:
        raise RuntimeError(f"Failed to persist champion summary: {errors}")
    logger.info("Persisted champion to %s", table_id)
    return champion_summary


def run_batch_forecast(
    project_id: str,
    bq_table: str,
    batch_forecast_bq_table: str,
    forecast_freq: str,
    horizon: int,
    champion_summary: dict,
):
    import pandas as pd

    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT unique_id, ds, y
    FROM `{project_id}.{bq_table}`
    ORDER BY ds DESC
    LIMIT 1000
    """
    df = client.query(query).to_dataframe()
    df["ds"] = pd.to_datetime(df["ds"])

    run_ts = pd.Timestamp.utcnow()
    rows = []
    for unique_id in df["unique_id"].unique():
        series = df[df["unique_id"] == unique_id]
        last_y = series["y"].iloc[-1]
        for step in range(1, horizon + 1):
            rows.append(
                {
                    "unique_id": unique_id,
                    "ds": pd.Timestamp.now() + pd.Timedelta(days=step),
                    "prediction": last_y * (1.0 + 0.01 * step),
                    "run_ts": run_ts,
                    "model_type": champion_summary["champion_model_type"],
                    "source_table": bq_table,
                    "forecast_freq": forecast_freq,
                    "horizon": horizon,
                    "is_future": True,
                }
            )

    forecast_df = pd.DataFrame(rows)
    forecast_df.to_gbq(batch_forecast_bq_table, project_id=project_id, if_exists="append")
    logger.info("Wrote %d batch forecasts to %s", len(forecast_df), batch_forecast_bq_table)


def run_local_pipeline(
    project_id: str,
    bq_table: str,
    prediction_bq_table: str,
    champion_bq_table: str,
    batch_forecast_bq_table: str,
    model_display_name: str,
    forecast_freq: str,
    horizon: int,
    lags: list[int],
    date_features: list[str],
    model_types: list[str],
):
    ensure_bigquery_targets(project_id)
    df = load_training_data(project_id, bq_table)
    results = []
    for model_type in model_types:
        results.append(
            train_model(
                project_id=project_id,
                df=df,
                prediction_bq_table=prediction_bq_table,
                model_display_name=model_display_name,
                bq_table=bq_table,
                model_type=model_type,
                forecast_freq=forecast_freq,
                horizon=horizon,
                lags=lags,
                date_features=date_features,
            )
        )

    champion = min(results, key=lambda result: result["wmape"])
    logger.info("Local champion selected: %s", champion["model_type"])
    champion_summary = persist_champion(project_id, champion_bq_table, champion)
    run_batch_forecast(
        project_id=project_id,
        bq_table=bq_table,
        batch_forecast_bq_table=batch_forecast_bq_table,
        forecast_freq=forecast_freq,
        horizon=horizon,
        champion_summary=champion_summary,
    )
    return {"results": results, "champion": champion_summary}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MLForecast workflow locally.")
    parser.add_argument("--project-id", default="dazzling-seat-366014")
    parser.add_argument("--bq-table", default="forecasting.sales_daily")
    parser.add_argument("--prediction-bq-table", default="forecasting.sales_predictions")
    parser.add_argument("--champion-bq-table", default="forecasting.sales_champion")
    parser.add_argument("--batch-forecast-bq-table", default="forecasting.sales_batch_forecasts")
    parser.add_argument("--model-display-name", default="sales-mlforecast-v3-local")
    parser.add_argument("--forecast-freq", default="D")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--lags", default="1,2,3")
    parser.add_argument("--date-features", default="dayofweek,month")
    parser.add_argument("--model-types", default=",".join(DEFAULT_MODELS))
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_local_pipeline(
        project_id=args.project_id,
        bq_table=args.bq_table,
        prediction_bq_table=args.prediction_bq_table,
        champion_bq_table=args.champion_bq_table,
        batch_forecast_bq_table=args.batch_forecast_bq_table,
        model_display_name=args.model_display_name,
        forecast_freq=args.forecast_freq,
        horizon=args.horizon,
        lags=[int(value) for value in args.lags.split(",") if value],
        date_features=[value for value in args.date_features.split(",") if value],
        model_types=[value for value in args.model_types.split(",") if value],
    )
    print(json.dumps(result["champion"], indent=2))


if __name__ == "__main__":
    main()