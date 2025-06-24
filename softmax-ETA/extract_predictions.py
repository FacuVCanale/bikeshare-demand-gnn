"""
Extract predictions from current pipeline run to use with fast station arrivals analysis.

This script recreates the XGBoost model from your current run and extracts predictions
to numpy files that can be used with the ultra-fast station arrivals script.

Usage:
    python extract_predictions.py --test-data data/processed/trips_test.parquet --output-dir predictions
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from train_gbdt import GBDTTrainer
from evaluation import load_and_prepare_data


def extract_xgb_predictions(output_dir: str = "predictions"):
    """
    Extract XGBoost predictions by recreating the model from your pipeline run.
    """
    print("🔮 EXTRACTING XGBOOST PREDICTIONS")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data (same as in pipeline)
    data_dir = Path("data/processed")
    train_path = data_dir / "trips_train.parquet"
    val_path = data_dir / "trips_val.parquet"
    test_path = data_dir / "trips_test.parquet"
    
    print("📂 Loading datasets...")
    train_df, val_df, test_df = load_and_prepare_data(
        str(train_path), str(val_path), str(test_path)
    )
    
    print(f"✅ Loaded test data: {len(test_df):,} trips")
    
    # Train XGBoost model (same as pipeline)
    print("🏗️  Training XGBoost model...")
    model = GBDTTrainer(model_type="xgb", use_gpu=True)
    model.fit(train_df, val_df)
    
    print("🔮 Getting predictions...")
    eta_pred, dest_pred, dest_proba = model.predict(test_df)
    
    # Save predictions
    eta_file = output_path / "eta_predictions.npy"
    dest_file = output_path / "dest_predictions.npy"
    proba_file = output_path / "dest_probabilities.npy"
    
    np.save(eta_file, eta_pred)
    np.save(dest_file, dest_pred)
    np.save(proba_file, dest_proba)
    
    print(f"💾 Saved predictions:")
    print(f"   ETA: {eta_file}")
    print(f"   Destinations: {dest_file}")
    print(f"   Probabilities: {proba_file}")
    
    # Save test data for reference
    test_file = output_path / "trips_test.parquet"
    test_df.write_parquet(test_file)
    print(f"   Test data: {test_file}")
    
    print("\n🚀 Ready for ultra-fast analysis! Run:")
    print(f"python fast_station_arrivals.py \\")
    print(f"  --test-data {test_file} \\")
    print(f"  --eta-pred {eta_file} \\")
    print(f"  --dest-pred {dest_file} \\")
    print(f"  --delta-t 1800")


def main():
    parser = argparse.ArgumentParser(
        description="Extract XGBoost predictions for fast station arrival analysis"
    )
    
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Directory to save predictions (default: predictions)')
    
    args = parser.parse_args()
    
    extract_xgb_predictions(args.output_dir)


if __name__ == "__main__":
    main() 