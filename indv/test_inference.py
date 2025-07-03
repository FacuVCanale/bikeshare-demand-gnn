#!/usr/bin/env python
"""
Test Set Inference Script for Individual Station Models

This script loads the processed_data.parquet and splits it using the same 
methodology as training (last 15% as test set), then runs inference only 
on that test portion.

Usage:
    python test_inference.py \
        --processed_data_path results/processed_data.parquet \
        --feature_scaler_path results/models/feature_scaler.joblib \
        --arrivals_model_path results/models/lightgbm_arrivals.joblib \
        --departures_model_path results/models/lightgbm_departures.joblib \
        --output_dir test_inference_results
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

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute common regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def _split_processed_data(df: pl.DataFrame, test_size: float = 0.15):
    """
    Split processed data using time-based split (last 15% as test).
    Matches the training pipeline methodology.
    """
    print(f"Splitting processed data with test_size={test_size}")
    
    # Sort by datetime to ensure proper chronological split
    df_sorted = df.sort('datetime')
    
    # Calculate split point
    n_samples = len(df_sorted)
    test_start_idx = int(n_samples * (1 - test_size))
    
    train_val_df = df_sorted[:test_start_idx]
    test_df = df_sorted[test_start_idx:]
    
    print(f"  Total samples: {n_samples:,}")
    print(f"  Train+Val samples: {len(train_val_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Test date range: {test_df.select(pl.col('datetime').min()).item()} to {test_df.select(pl.col('datetime').max()).item()}")
    
    return test_df


def _prepare_features_targets(df: pl.DataFrame, target_cols: list = ['arrivals', 'departures']):
    """
    Prepare features and targets from processed dataframe.
    Mimics the prepare_for_training function from feature_engineering.
    """
    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in 
                   ['datetime', 'station_id'] + target_cols]
    
    X = df.select(feature_cols)
    y = df.select(target_cols)
    metadata = df.select(['datetime', 'station_id'])
    
    print(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} samples")
    
    return X, y, metadata


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def run_test_inference(
    processed_data_path: str,
    feature_scaler_path: str,
    arrivals_model_path: str,
    departures_model_path: str,
    output_dir: str = "test_inference_results",
    test_size: float = 0.15,
    target_date: str | None = None,
    start_hour: int | None = None,
    end_hour: int | None = None,
):
    """Run inference on test split of processed data.

    Parameters
    ----------
    processed_data_path: str
        Path to processed_data.parquet file.
    feature_scaler_path: str
        Path to saved StandardScaler joblib file.
    arrivals_model_path: str
        Path to LightGBM arrivals model joblib.
    departures_model_path: str
        Path to LightGBM departures model joblib.
    output_dir: str, default "test_inference_results"
        Directory to store predictions and metrics.
    test_size: float, default 0.15
        Fraction of data to use as test set (last 15%).
    target_date: str | None, default None
        Optional: filter for specific date (YYYY-MM-DD format).
    start_hour: int | None, default None
        Optional: filter for specific hour range start.
    end_hour: int | None, default None
        Optional: filter for specific hour range end.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n==============================")
    print("TEST SET INFERENCE")
    print("==============================")
    print(f"Processed data   : {processed_data_path}")
    print(f"Scaler           : {feature_scaler_path}")
    print(f"Arrivals model   : {arrivals_model_path}")
    print(f"Departures model : {departures_model_path}")
    print(f"Output directory : {output_dir}")
    print(f"Test size        : {test_size:.1%}\n")

    # ------------------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------------------
    print("Loading processed data...")
    processed_df = pl.read_parquet(processed_data_path)
    print(f"  Total rows: {len(processed_df):,}")
    print(f"  Columns: {len(processed_df.columns)}")
    print(f"  Date range: {processed_df.select(pl.col('datetime').min()).item()} to {processed_df.select(pl.col('datetime').max()).item()}")

    # ------------------------------------------------------------------
    # 2. Split to get test set
    # ------------------------------------------------------------------
    test_df = _split_processed_data(processed_df, test_size=test_size)

    # ------------------------------------------------------------------
    # 3. Optional filtering by date/time
    # ------------------------------------------------------------------
    if target_date or (start_hour is not None and end_hour is not None):
        print("\nApplying additional filters...")
        
        if target_date:
            # Parse target date
            year, month, day = map(int, target_date.split('-'))
            test_df = test_df.filter(
                pl.col("datetime").dt.date() == pl.date(year, month, day)
            )
            print(f"  Filtered for date: {target_date}")
        
        if start_hour is not None and end_hour is not None:
            test_df = test_df.filter(
                (pl.col("datetime").dt.hour() >= start_hour) &
                (pl.col("datetime").dt.hour() <= end_hour)
            )
            print(f"  Filtered for hours: {start_hour}:00-{end_hour}:00")
        
        print(f"  Samples after filtering: {len(test_df):,}")
        
        if len(test_df) == 0:
            raise ValueError("No data available after applying filters!")

    # ------------------------------------------------------------------
    # 4. Prepare features and targets
    # ------------------------------------------------------------------
    print("\nPreparing features and targets...")
    X_pl, y_pl, metadata_pl = _prepare_features_targets(test_df)

    # Convert to pandas (scikit-learn friendly)
    X = X_pl.to_pandas()
    y = y_pl.to_pandas()
    metadata = metadata_pl.to_pandas()

    # ------------------------------------------------------------------
    # 5. Apply scaling
    # ------------------------------------------------------------------
    print("\nApplying feature scaling...")
    scaler = joblib.load(feature_scaler_path)

    # Align columns with scaler expectation
    if hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
        missing = set(expected_cols) - set(X.columns)
        extra = set(X.columns) - set(expected_cols)
        
        if missing:
            print(f"  Missing columns: {sorted(missing)}")
            print(f"  Available columns: {sorted(X.columns)}")
            raise ValueError(f"Missing columns required by scaler: {sorted(missing)}")
        
        if extra:
            print(f"  Extra columns will be dropped: {sorted(extra)}")
        
        # Reorder accordingly
        X = X[expected_cols]
        print(f"  Aligned {len(expected_cols)} features with scaler")
    else:
        print("  Warning: StandardScaler has no feature_names_in_.")

    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------------
    # 6. Load models & predict
    # ------------------------------------------------------------------
    print("\nLoading models & generating predictions...")
    arrivals_model = joblib.load(arrivals_model_path)
    departures_model = joblib.load(departures_model_path)

    arrivals_pred = arrivals_model.predict(X_scaled)
    departures_pred = departures_model.predict(X_scaled)

    # ------------------------------------------------------------------
    # 7. Evaluation metrics
    # ------------------------------------------------------------------
    print("\nComputing evaluation metrics...")
    metrics = {
        "arrivals": _compute_basic_metrics(y["arrivals"].values, arrivals_pred),
        "departures": _compute_basic_metrics(y["departures"].values, departures_pred),
    }

    # Pretty print metrics
    for target, m in metrics.items():
        print(f"\n{target.upper()}:")
        for k, v in m.items():
            print(f"  {k.upper():<4}: {v:,.4f}")

    # ------------------------------------------------------------------
    # 8. Save outputs
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

    arrivals_path = Path(output_dir) / f"test_arrivals_predictions_{timestamp}.csv"
    departures_path = Path(output_dir) / f"test_departures_predictions_{timestamp}.csv"
    metrics_path = Path(output_dir) / f"test_metrics_{timestamp}.json"

    arrivals_df.to_csv(arrivals_path, index=False)
    departures_df.to_csv(departures_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved test arrivals predictions   : {arrivals_path}")
    print(f"Saved test departures predictions : {departures_path}")
    print(f"Saved test metrics                : {metrics_path}")
    print("\nTest set inference complete! ✨")

    return metrics, arrivals_df, departures_df


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Run inference on test split of processed data."
    )
    parser.add_argument("--processed_data_path", type=str, required=True,
                        help="Path to processed_data.parquet file.")
    parser.add_argument("--feature_scaler_path", type=str, required=True,
                        help="Path to saved StandardScaler joblib file.")
    parser.add_argument("--arrivals_model_path", type=str, required=True,
                        help="Path to LightGBM arrivals model joblib.")
    parser.add_argument("--departures_model_path", type=str, required=True,
                        help="Path to LightGBM departures model joblib.")
    parser.add_argument("--output_dir", type=str, default="test_inference_results",
                        help="Directory where predictions & metrics will be stored.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Fraction of data to use as test set (default: 0.15).")
    
    # Optional filtering arguments
    parser.add_argument("--target_date", type=str, default=None,
                        help="Optional: filter for specific date (YYYY-MM-DD).")
    parser.add_argument("--start_hour", type=int, default=None,
                        help="Optional: filter for specific hour range start.")
    parser.add_argument("--end_hour", type=int, default=None,
                        help="Optional: filter for specific hour range end.")

    args = parser.parse_args()

    run_test_inference(
        processed_data_path=args.processed_data_path,
        feature_scaler_path=args.feature_scaler_path,
        arrivals_model_path=args.arrivals_model_path,
        departures_model_path=args.departures_model_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        target_date=args.target_date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )


if __name__ == "__main__":
    _cli()