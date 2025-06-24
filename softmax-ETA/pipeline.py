"""
Comprehensive pipeline for ETA and destination prediction with optional station arrival analysis.

This script orchestrates the entire workflow:
1. Feature engineering and data splitting
2. Model training (baseline, XGBoost, or LightGBM)
3. Individual trip evaluation
4. Optional station arrival count analysis

Usage:
    python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model lgb
    python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model baseline --arrivals
    python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model xgb --arrivals --delta-t 1800
"""

import argparse
import sys
import subprocess
from pathlib import Path
import polars as pl
import numpy as np
from typing import Dict, Any, Tuple

# Import our modules
from evaluation import (
    evaluate_eta_prediction, 
    evaluate_destination_prediction,
    print_evaluation_results,
    create_station_arrival_targets,
    create_predicted_station_arrivals,
    evaluate_station_arrival_prediction,
    print_station_arrival_evaluation,
    load_and_prepare_data
)
from train_baseline import HistoricalBaseline
from train_gbdt import GBDTTrainer


def run_feature_engineering(trips_path: str, users_path: str, force_rebuild: bool = False) -> bool:
    """
    Run feature engineering to create train/val/test splits.
    
    Args:
        trips_path: Path to trips_with_weather.parquet
        users_path: Path to users.csv
        force_rebuild: Whether to force rebuilding even if files exist
        
    Returns:
        True if successful, False otherwise
    """
    print("🔧 FEATURE ENGINEERING")
    print("="*60)
    
    # Check if output files already exist
    data_dir = Path("data/processed")
    required_files = [
        data_dir / "trips_train.parquet",
        data_dir / "trips_val.parquet", 
        data_dir / "trips_test.parquet"
    ]
    
    if not force_rebuild and all(f.exists() for f in required_files):
        print("✅ Train/val/test splits already exist. Skipping feature engineering.")
        return True
    
    # Check if input files exist
    if not Path(trips_path).exists():
        print(f"❌ Trips file not found: {trips_path}")
        return False
    
    if not Path(users_path).exists():
        print(f"❌ Users file not found: {users_path}")
        return False
    
    print(f"📂 Input files:")
    print(f"   Trips: {trips_path}")
    print(f"   Users: {users_path}")
    
    try:
        # Modify feat-eng.py paths and run it
        print("🚀 Running feature engineering...")
        
        # Read and modify feat-eng.py to use our custom paths
        feat_eng_script = "feat-eng.py"
        
        # Create a temporary modified version
        with open(feat_eng_script, 'r') as f:
            content = f.read()
        
        # Replace the hardcoded paths
        content = content.replace(
            'TRIPS_PATH  = Path("data/processed/trips_with_weather.parquet")',
            f'TRIPS_PATH  = Path("{trips_path}")'
        )
        content = content.replace(
            'USERS_PATH  = Path("data/raw/combined/users.csv")',
            f'USERS_PATH  = Path("{users_path}")'
        )
        
        # Write temporary script
        temp_script = "feat-eng_temp.py"
        with open(temp_script, 'w') as f:
            f.write(content)
        
        # Run the modified script
        result = subprocess.run([sys.executable, temp_script], capture_output=True, text=True)
        
        # Clean up
        Path(temp_script).unlink()
        
        if result.returncode != 0:
            print(f"❌ Feature engineering failed:")
            print(result.stderr)
            return False
        
        print("✅ Feature engineering completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        return False


def train_model(model_type: str, experiment_name: str = None) -> Tuple[Any, str]:
    """
    Train the specified model type.
    
    Args:
        model_type: 'baseline', 'xgb', or 'lgb'
        experiment_name: Optional experiment name
        
    Returns:
        Tuple of (trained_model, model_name)
    """
    print(f"\n🏗️ TRAINING {model_type.upper()} MODEL")
    print("="*60)
    
    # Define data paths
    data_dir = Path("data/processed")
    train_path = data_dir / "trips_train.parquet"
    val_path = data_dir / "trips_val.parquet"
    test_path = data_dir / "trips_test.parquet"
    
    # Load data
    train_df, val_df, test_df = load_and_prepare_data(
        str(train_path), str(val_path), str(test_path)
    )
    
    if model_type == "baseline":
        model = HistoricalBaseline()
        model.fit(train_df)
        model_name = "Historical Baseline"
        
    elif model_type in ["xgb", "lgb"]:
        model = GBDTTrainer(model_type=model_type, use_gpu=True)
        model.fit(train_df, val_df)
        model_name = f"{model_type.upper()} GBDT"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"✅ {model_name} training completed!")
    return model, model_name


def evaluate_individual_trips(model, model_type: str, model_name: str) -> Tuple[Dict, Dict]:
    """
    Evaluate model on individual trip prediction tasks.
    
    Args:
        model: Trained model
        model_type: 'baseline', 'xgb', or 'lgb'
        model_name: Human-readable model name
        
    Returns:
        Tuple of (eta_metrics, dest_metrics)
    """
    print(f"\n🔍 INDIVIDUAL TRIP EVALUATION")
    print("="*60)
    
    # Load test data
    data_dir = Path("data/processed")
    test_path = data_dir / "trips_test.parquet"
    test_df = pl.read_parquet(test_path)
    
    # Get true values
    test_eta_true = test_df.select("duracion_recorrido").to_numpy().flatten()
    test_dest_true = test_df.select("id_estacion_destino").to_numpy().flatten()
    
    # Get predictions based on model type
    if model_type == "baseline":
        test_eta_pred = model.predict_eta(test_df)
        test_dest_pred, test_dest_proba = model.predict_destination(test_df)
        class_labels = sorted(model.overall_dest_frequencies.keys())
        
    elif model_type in ["xgb", "lgb"]:
        test_eta_pred, test_dest_pred, test_dest_proba = model.predict(test_df)
        class_labels = model.dest_classes
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate metrics
    eta_metrics = evaluate_eta_prediction(test_eta_true, test_eta_pred)
    dest_metrics = evaluate_destination_prediction(
        test_dest_true, test_dest_pred, test_dest_proba, class_labels
    )
    
    # Print results
    print_evaluation_results(eta_metrics, dest_metrics, model_name)
    
    return eta_metrics, dest_metrics


def evaluate_station_arrivals(
    model, 
    model_type: str, 
    model_name: str, 
    delta_t: int = 1800
) -> Dict[str, float]:
    """
    Evaluate model on station arrival count prediction.
    
    Args:
        model: Trained model
        model_type: 'baseline', 'xgb', or 'lgb'
        model_name: Human-readable model name
        delta_t: Time window in seconds for arrival counting
        
    Returns:
        Dictionary with station arrival metrics
    """
    print(f"\n🏢 STATION ARRIVAL EVALUATION")
    print("="*60)
    
    # Load test data
    data_dir = Path("data/processed")
    test_path = data_dir / "trips_test.parquet"
    test_df = pl.read_parquet(test_path)
    
    print(f"📊 Analyzing station arrivals with {delta_t/60:.1f}-minute windows...")
    
    # Create true station arrival targets
    print("\n🎯 Creating true station arrival targets...")
    arrival_targets_true, station_list = create_station_arrival_targets(
        test_df, delta_t=delta_t
    )
    
    # Get model predictions
    print(f"\n🤖 Getting model predictions...")
    if model_type == "baseline":
        eta_pred = model.predict_eta(test_df)
        dest_pred, _ = model.predict_destination(test_df)
        
    elif model_type in ["xgb", "lgb"]:
        eta_pred, dest_pred, _ = model.predict(test_df)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create predicted station arrivals
    print(f"\n📈 Creating predicted station arrivals...")
    arrival_targets_pred, _ = create_predicted_station_arrivals(
        test_df, eta_pred, dest_pred, delta_t=delta_t
    )
    
    # Evaluate station arrival predictions
    print(f"\n📊 Evaluating station arrival predictions...")
    station_metrics = evaluate_station_arrival_prediction(
        arrival_targets_true, arrival_targets_pred, station_list
    )
    
    # Print results
    print_station_arrival_evaluation(station_metrics, delta_t, model_name)
    
    return station_metrics


def save_results(
    eta_metrics: Dict, 
    dest_metrics: Dict, 
    station_metrics: Dict, 
    model_name: str,
    output_dir: str = "results"
):
    """Save all evaluation results to files."""
    print(f"\n💾 SAVING RESULTS")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Combine all metrics
    all_metrics = {
        "model": model_name,
        **{f"eta_{k}": v for k, v in eta_metrics.items()},
        **{f"dest_{k}": v for k, v in dest_metrics.items()}
    }
    
    # Add station metrics if available
    if station_metrics:
        all_metrics.update({f"station_{k}": v for k, v in station_metrics.items()})
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame([all_metrics])
    
    # Create safe filename
    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_file = output_path / f"{safe_model_name}_pipeline_results.csv"
    
    df.to_csv(output_file, index=False)
    print(f"📊 Results saved to: {output_file}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline for bike trip prediction and station arrival analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with LightGBM
  python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model lgb
  
  # Baseline model with station arrival analysis
  python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model baseline --arrivals
  
  # XGBoost with custom time window for arrivals
  python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model xgb --arrivals --delta-t 3600
        """
    )
    
    # Required arguments
    parser.add_argument('--trips-path', type=str, required=True,
                       help='Path to trips_with_weather.parquet file')
    parser.add_argument('--users-path', type=str, required=True,
                       help='Path to users.csv file')
    parser.add_argument('--model', type=str, choices=['baseline', 'xgb', 'lgb'], required=True,
                       help='Model type to train and evaluate')
    
    # Optional arguments
    parser.add_argument('--arrivals', action='store_true',
                       help='Enable station arrival analysis (default: disabled)')
    parser.add_argument('--delta-t', type=int, default=1800,
                       help='Time window for arrival analysis in seconds (default: 1800 = 30 minutes)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuilding of train/val/test splits even if they exist')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Optional experiment name for this run')
    
    args = parser.parse_args()
    
    print("🚀 BIKE TRIP PREDICTION PIPELINE")
    print("="*80)
    print(f"📂 Trips Path:       {args.trips_path}")
    print(f"👥 Users Path:       {args.users_path}")
    print(f"🤖 Model:            {args.model}")
    print(f"🏢 Station Arrivals: {'Enabled' if args.arrivals else 'Disabled'}")
    if args.arrivals:
        print(f"⏰ Delta T:          {args.delta_t} seconds ({args.delta_t/60:.1f} minutes)")
    print(f"📁 Output Dir:       {args.output_dir}")
    print("="*80)
    
    # Step 1: Feature Engineering
    if not run_feature_engineering(args.trips_path, args.users_path, args.force_rebuild):
        print("❌ Pipeline failed at feature engineering step")
        return 1
    
    # Step 2: Model Training
    try:
        model, model_name = train_model(args.model, args.experiment_name)
    except Exception as e:
        print(f"❌ Pipeline failed at model training step: {e}")
        return 1
    
    # Step 3: Individual Trip Evaluation
    try:
        eta_metrics, dest_metrics = evaluate_individual_trips(model, args.model, model_name)
    except Exception as e:
        print(f"❌ Pipeline failed at individual trip evaluation step: {e}")
        return 1
    
    # Step 4: Station Arrival Evaluation (if enabled)
    station_metrics = {}
    if args.arrivals:
        try:
            station_metrics = evaluate_station_arrivals(
                model, args.model, model_name, args.delta_t
            )
        except Exception as e:
            print(f"❌ Pipeline failed at station arrival evaluation step: {e}")
            return 1
    
    # Step 5: Save Results
    try:
        save_results(eta_metrics, dest_metrics, station_metrics, model_name, args.output_dir)
    except Exception as e:
        print(f"⚠️  Warning: Could not save results: {e}")
    
    # Final Summary
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Model: {model_name}")
    print(f"📈 ETA R²: {eta_metrics['r2']:.4f}")
    print(f"🎯 Destination Accuracy: {dest_metrics['accuracy']:.4f}")
    
    if station_metrics:
        print(f"🏢 Station Arrival R²: {station_metrics['overall_r2']:.4f}")
        print(f"🏢 Station Correlation: {station_metrics['station_correlation']:.4f}")
    
    print(f"📁 Results saved in: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 