#!/usr/bin/env python
"""
Inference & Evaluation Script for Individual Station Models

Usage (example):
    python inference.py \
        --test_data_path data/new_trips.csv \
        --feature_scaler_path models/feature_scaler.joblib \
        --arrivals_model_path models/lightgbm_arrivals.joblib \
        --departures_model_path models/lightgbm_departures.joblib \
        --output_dir inference_results

The script performs:
1. Optional merge with existing historical raw trips to preserve lag features
2. Feature engineering (re-uses FeatureEngineer from indv.feature_engineering)
3. Applies saved StandardScaler
4. Loads LightGBM models for arrivals & departures
5. Generates predictions and computes regression metrics (RMSE, MAE, R², MAPE)
6. Saves predictions & metrics to the chosen output directory
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports from the same folder (relative import works when executed from repo root)
from feature_engineering import FeatureEngineer, load_and_process_data

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_raw_data(path: str) -> pl.DataFrame:
    """Load CSV or Parquet into a Polars DataFrame."""
    if path.lower().endswith(".parquet"):
        return pl.read_parquet(path)
    elif path.lower().endswith(".csv"):
        return pl.read_csv(path, try_parse_dates=True)
    else:
        # Try parquet first, fallback to csv
        try:
            return pl.read_parquet(path)
        except Exception:
            return pl.read_csv(path, try_parse_dates=True)


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute common regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def run_inference(
    test_data_path: str,
    feature_scaler_path: str,
    arrivals_model_path: str,
    departures_model_path: str,
    output_dir: str = "inference_results",
    existing_data_path: str | None = None,
    delta_t_minutes: int = 30,
    fill_nulls: bool = True,
    fill_strategy: str = "zero",
    add_long_lags: bool = True,
    lookback_intervals: int = 1,
    station_id: int | None = None,
):
    """Run feature engineering, scaling, prediction & evaluation.

    Parameters
    ----------
    test_data_path: str
        Path to new raw trips CSV/Parquet containing the evaluation window.
    feature_scaler_path: str
        Joblib file with the fitted ``StandardScaler`` used during training.
    arrivals_model_path: str
        Joblib file with the trained LightGBM model for *arrivals*.
    departures_model_path: str
        Joblib file with the trained LightGBM model for *departures*.
    output_dir: str, default "inference_results"
        Directory to store predictions and metrics.
    existing_data_path: str | None, default None
        Optional raw trips file with historical data so that long-lag features
        (e.g. 1-week lags, moving averages) are computed without leakage.
    All other parameters replicate the values used in training so that feature
    engineering remains consistent.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n==============================")
    print("MODEL INFERENCE & EVALUATION")
    print("==============================")
    print(f"Test data        : {test_data_path}")
    print(f"Existing raw data: {existing_data_path or 'N/A'}")
    print(f"Scaler           : {feature_scaler_path}")
    print(f"Arrivals model   : {arrivals_model_path}")
    print(f"Departures model : {departures_model_path}")
    print(f"Output directory : {output_dir}\n")

    # ------------------------------------------------------------------
    # 1. Load raw data (test + optional historical)
    # ------------------------------------------------------------------
    print("Loading raw trips ...")
    test_raw = _load_raw_data(test_data_path)
    print(f"  Test rows: {len(test_raw):,}")

    if existing_data_path:
        hist_raw = _load_raw_data(existing_data_path)
        print(f"  Historical rows: {len(hist_raw):,}")
        combined_raw = pl.concat([hist_raw, test_raw])
    else:
        combined_raw = test_raw

    # ------------------------------------------------------------------
    # 2. Feature engineering (re-use existing implementation)
    # ------------------------------------------------------------------
    print("\nRunning feature engineering ...")
    fe = FeatureEngineer(
        delta_t_minutes=delta_t_minutes,
        fill_nulls=fill_nulls,
        fill_strategy=fill_strategy,
        add_long_lags=add_long_lags,
        lookback_intervals=lookback_intervals,
    )
    processed_df = fe.process_data(combined_raw, station_id=station_id)

    # Filter for specific date and time range: 9/9/2024, 12:00-18:00
    target_date = "2024-09-09"
    start_hour = 12
    end_hour = 18
    
    eval_df = processed_df.filter(
        (pl.col("datetime").dt.date() == pl.date(2024, 9, 9)) &
        (pl.col("datetime").dt.hour() >= start_hour) &
        (pl.col("datetime").dt.hour() <= end_hour)
    )
    print(f"Filtered for {target_date} {start_hour}:00-{end_hour}:00")
    print(f"Processed intervals for evaluation: {len(eval_df):,}")

    # Prepare for model input (features / targets)
    X_pl, y_pl, metadata_pl = fe.prepare_for_training(
        eval_df, target_cols=["arrivals", "departures"]
    )

    # Convert to pandas (scikit-learn friendly)
    X = X_pl.to_pandas()
    y = y_pl.to_pandas()
    metadata = metadata_pl.to_pandas()

    # ------------------------------------------------------------------
    # 3. Scaling  -------------------------------------------------------
    # ------------------------------------------------------------------
    print("\nApplying feature scaling ...")
    scaler = joblib.load(feature_scaler_path)

    # Align columns with scaler expectation (if names stored)
    if hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
        missing = set(expected_cols) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing columns required by scaler: {sorted(missing)}"
            )
        # Reorder accordingly
        X = X[expected_cols]
    else:
        print("  Warning: `StandardScaler` object has no `feature_names_in_`. "
              "Assuming training & inference column order are identical.")

    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------------
    # 4. Load models & predict -----------------------------------------
    # ------------------------------------------------------------------
    print("\nLoading models & generating predictions ...")
    arrivals_model = joblib.load(arrivals_model_path)
    departures_model = joblib.load(departures_model_path)

    arrivals_pred = arrivals_model.predict(X_scaled)
    departures_pred = departures_model.predict(X_scaled)

    # ------------------------------------------------------------------
    # 5. Evaluation metrics --------------------------------------------
    # ------------------------------------------------------------------
    print("\nComputing evaluation metrics ...")
    metrics = {
        "arrivals": _compute_basic_metrics(y["arrivals"].values, arrivals_pred),
        "departures": _compute_basic_metrics(y["departures"].values, departures_pred),
    }

    # Pretty print
    for target, m in metrics.items():
        print(f"\n{target.upper()}:")
        for k, v in m.items():
            print(f"  {k.upper():<4}: {v:,.4f}")

    # ------------------------------------------------------------------
    # 6. Save outputs ---------------------------------------------------
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create arrivals CSV: datetime, station_id, pred, target
    arrivals_df = pd.DataFrame({
        'datetime': metadata['datetime'],
        'station_id': metadata['station_id'],
        'pred': arrivals_pred,
        'target': y['arrivals'].values
    })
    
    # Create departures CSV: datetime, station_id, pred, target
    departures_df = pd.DataFrame({
        'datetime': metadata['datetime'],
        'station_id': metadata['station_id'],
        'pred': departures_pred,
        'target': y['departures'].values
    })

    arrivals_path = Path(output_dir) / f"arrivals_predictions_{timestamp}.csv"
    departures_path = Path(output_dir) / f"departures_predictions_{timestamp}.csv"
    metrics_path = Path(output_dir) / f"metrics_{timestamp}.json"

    arrivals_df.to_csv(arrivals_path, index=False)
    departures_df.to_csv(departures_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved arrivals predictions   : {arrivals_path}")
    print(f"Saved departures predictions : {departures_path}")
    print(f"Saved metrics                : {metrics_path}")
    print("\nInference & evaluation complete! ✨")

    return metrics, arrivals_df, departures_df


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Run inference and evaluation on new trip data using trained models."
    )
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to new raw trips CSV or Parquet file.")
    parser.add_argument("--feature_scaler_path", type=str, required=True,
                        help="Path to saved StandardScaler joblib file.")
    parser.add_argument("--arrivals_model_path", type=str, required=True,
                        help="Path to LightGBM arrivals model joblib.")
    parser.add_argument("--departures_model_path", type=str, required=True,
                        help="Path to LightGBM departures model joblib.")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory where predictions & metrics will be stored.")
    parser.add_argument("--existing_data_path", type=str, default=None,
                        help="Optional raw trips file with historical data for lag computation.")
    parser.add_argument("--station_id", type=int, default=None,
                        help="If provided, limits inference to a specific station ID.")
    # Feature engineering flags (should match training settings)
    parser.add_argument("--disable_long_lags", action="store_true",
                        help="Disable 1-week lag & moving average features.")
    parser.add_argument("--lookback_intervals", type=int, default=1,
                        help="Number of previous ΔT intervals for last_dt features (default 1).")
    parser.add_argument("--no_fill_nulls", action="store_true",
                        help="Keep nulls instead of filling (defaults to zero fill).")
    parser.add_argument("--fill_strategy", type=str, default="zero",
                        choices=["zero", "mean", "median", "forward_fill"],
                        help="Strategy to fill remaining nulls (default zero).")

    args = parser.parse_args()

    run_inference(
        test_data_path=args.test_data_path,
        feature_scaler_path=args.feature_scaler_path,
        arrivals_model_path=args.arrivals_model_path,
        departures_model_path=args.departures_model_path,
        output_dir=args.output_dir,
        existing_data_path=args.existing_data_path,
        fill_nulls=not args.no_fill_nulls,
        fill_strategy=args.fill_strategy,
        add_long_lags=not args.disable_long_lags,
        lookback_intervals=args.lookback_intervals,
        station_id=args.station_id,
    )


if __name__ == "__main__":
    _cli()
