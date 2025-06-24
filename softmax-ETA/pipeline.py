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
import logging
import datetime
import os

# Import our modules
from evaluation import (
    evaluate_eta_prediction, 
    evaluate_destination_prediction,
    print_evaluation_results,
    create_station_arrival_targets_parallel,
    create_predicted_station_arrivals_parallel,
    evaluate_station_arrival_prediction,
    print_station_arrival_evaluation,
    load_and_prepare_data
)
from train_baseline import HistoricalBaseline
from train_gbdt import GBDTTrainer


def setup_logging(log_file: str = None):
    """
    Set up logging to write to both console and file.
    
    Args:
        log_file: Path to log file. If None, only logs to console.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logging.info(f"📝 Logging to file: {log_file}")
    
    return logger


def log_print(*args, **kwargs):
    """
    Custom print function that logs to both console and file.
    This replaces regular print statements to ensure all output is captured.
    """
    # Convert all arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Use logging instead of print
    logging.info(message)


# Replace print with our logging version globally
print = log_print


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
        
        # Run the modified script and capture output
        print("   📊 Running feature engineering script...")
        result = subprocess.run(
            [sys.executable, temp_script], 
            capture_output=True, 
            text=True
        )
        
        # Log the output from the subprocess
        if result.stdout:
            print("   📋 Feature engineering output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
        
        # Clean up
        Path(temp_script).unlink()
        
        if result.returncode != 0:
            print(f"❌ Feature engineering failed:")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"   ERROR: {line}")
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


def save_predictions_immediately(
    test_df: pl.DataFrame,
    eta_pred: np.ndarray,
    dest_pred: np.ndarray,
    dest_proba: np.ndarray,
    predictions_dir: str = "predictions"
):
    """
    Save predictions immediately after calculation for use with fast analysis.
    
    Args:
        test_df: Test dataframe
        eta_pred: ETA predictions
        dest_pred: Destination predictions  
        dest_proba: Destination probabilities
        predictions_dir: Directory to save predictions
    """
    pred_path = Path(predictions_dir)
    pred_path.mkdir(exist_ok=True, parents=True)
    
    # Save predictions
    eta_file = pred_path / "eta_predictions.npy"
    dest_file = pred_path / "dest_predictions.npy" 
    proba_file = pred_path / "dest_probabilities.npy"
    test_file = pred_path / "trips_test.parquet"
    
    np.save(eta_file, eta_pred)
    np.save(dest_file, dest_pred)
    np.save(proba_file, dest_proba)
    test_df.write_parquet(test_file)
    
    print(f"💾 Predictions saved immediately:")
    print(f"   ETA: {eta_file}")
    print(f"   Destinations: {dest_file}")
    print(f"   Probabilities: {proba_file}")
    print(f"   Test data: {test_file}")
    
    return str(test_file), str(eta_file), str(dest_file)


def evaluate_individual_trips(model, model_type: str, model_name: str, predictions_dir: str = "predictions") -> Tuple[Dict, Dict, Tuple[str, str, str]]:
    """
    Evaluate model on individual trip prediction tasks.
    
    Args:
        model: Trained model
        model_type: 'baseline', 'xgb', or 'lgb'
        model_name: Human-readable model name
        predictions_dir: Directory to save predictions immediately
        
    Returns:
        Tuple of (eta_metrics, dest_metrics, prediction_files_paths)
    """
    print(f"\n🔍 INDIVIDUAL TRIP EVALUATION")
    print("="*60)
    
    # Load test data
    data_dir = Path("data/processed")
    test_path = data_dir / "trips_test.parquet"
    test_df = pl.read_parquet(test_path)
    
    print(f"📊 Loaded test data: {len(test_df):,} trips")
    
    # Get true values
    test_eta_true = test_df.select("duracion_recorrido").to_numpy().flatten()
    test_dest_true = test_df.select("id_estacion_destino").to_numpy().flatten()
    
    # Get predictions based on model type
    if model_type == "baseline":
        print("🔮 Getting baseline predictions...")
        test_eta_pred = model.predict_eta(test_df)
        test_dest_pred, test_dest_proba = model.predict_destination(test_df)
        class_labels = sorted(model.overall_dest_frequencies.keys())
        
    elif model_type in ["xgb", "lgb"]:
        print("🔮 Getting GBDT predictions...")
        test_eta_pred, test_dest_pred, test_dest_proba = model.predict(test_df)
        class_labels = model.dest_classes
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 💾 SAVE PREDICTIONS IMMEDIATELY
    print("💾 Saving predictions immediately for potential fast analysis...")
    test_file, eta_file, dest_file = save_predictions_immediately(
        test_df, test_eta_pred, test_dest_pred, test_dest_proba, predictions_dir
    )
    
    print("📊 Calculating evaluation metrics...")
    
    # Calculate metrics
    eta_metrics = evaluate_eta_prediction(test_eta_true, test_eta_pred)
    dest_metrics = evaluate_destination_prediction(
        test_dest_true, test_dest_pred, test_dest_proba, class_labels
    )
    
    # Print results
    print_evaluation_results(eta_metrics, dest_metrics, model_name)
    
    return eta_metrics, dest_metrics, (test_file, eta_file, dest_file)


def evaluate_station_arrivals_fast(
    model, 
    model_type: str, 
    model_name: str, 
    delta_t: int = 1800,
    prediction_files: Tuple[str, str, str] = None,
    not_leak: bool = False,
    output_dir: str = "results"
) -> Dict[str, float]:
    """
    Evaluate model on station arrival count prediction using ultra-fast skeleton method.
    
    Args:
        model: Trained model (not used, predictions are from files)
        model_type: 'baseline', 'xgb', or 'lgb' (not used, predictions are from files)
        model_name: Human-readable model name
        delta_t: Time window in seconds for arrival counting
        prediction_files: Tuple of (test_file, eta_file, dest_file) - required
        not_leak: Whether to prevent temporal leakage
        output_dir: Directory for results output
        
    Returns:
        Dictionary with station arrival metrics
    """
    print(f"\n🏢 ULTRA-FAST STATION ARRIVAL EVALUATION (SKELETON METHOD)")
    print("="*60)
    
    if not prediction_files:
        raise ValueError("prediction_files required for fast station arrival evaluation")
    
    test_file, eta_file, dest_file = prediction_files
    
    # Build command for fast_station_arrivals.py
    cmd = [
        sys.executable, "fast_station_arrivals.py",
        "--test-data", test_file,
        "--eta-pred", eta_file,
        "--dest-pred", dest_file,
        "--delta-t", str(delta_t),
        "--output-dir", output_dir
    ]
    
    # Add not-leak flag if specified
    if not_leak:
        cmd.append("--not-leak")
    
    print(f"🚀 Running ultra-fast skeleton-based analysis...")
    print(f"   Command: {' '.join(cmd)}")
    
    # Run the fast analysis
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Log the output
        if result.stdout:
            print("📋 Fast analysis output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Fast station arrivals analysis failed:")
        if e.stderr:
            for line in e.stderr.split('\n'):
                if line.strip():
                    print(f"   ERROR: {line}")
        raise e
    
    # Load results from the saved metrics file
    metrics_file = Path(output_dir) / "fast_station_arrivals_metrics.csv"
    if metrics_file.exists():
        import pandas as pd
        metrics_df = pd.read_csv(metrics_file)
        if len(metrics_df) > 0:
            # Convert first row to dictionary and return
            station_metrics = metrics_df.iloc[0].to_dict()
            
            print(f"✅ Station arrival analysis completed using skeleton method!")
            print(f"📊 Results loaded from: {metrics_file}")
            
            return station_metrics
    
    # Fallback - return empty dict if we can't load results
    print(f"⚠️  Warning: Could not load station arrival metrics from {metrics_file}")
    return {}


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
        print(f"📊 Including station arrival metrics in results")
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame([all_metrics])
    
    # Create safe filename
    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_file = output_path / f"{safe_model_name}_pipeline_results.csv"
    
    df.to_csv(output_file, index=False)
    print(f"📊 Results saved to: {output_file}")
    
    # Log summary of saved metrics
    print(f"📋 Saved metrics summary:")
    print(f"   ETA metrics: {len([k for k in all_metrics.keys() if k.startswith('eta_')])}")
    print(f"   Destination metrics: {len([k for k in all_metrics.keys() if k.startswith('dest_')])}")
    if station_metrics:
        print(f"   Station metrics: {len([k for k in all_metrics.keys() if k.startswith('station_')])}")


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
  
  # XGBoost with custom time window for arrivals and logging
  python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model xgb --arrivals --delta-t 3600 --log-file logs/pipeline_run.log
  
  # LightGBM with leak prevention and station arrivals
  python pipeline.py --trips-path data/processed/trips_with_weather.parquet --users-path data/raw/combined/users.csv --model lgb --arrivals --not-leak
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
    parser.add_argument('--predictions-dir', type=str, default='predictions',
                       help='Directory to save predictions immediately (default: predictions)')

    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuilding of train/val/test splits even if they exist')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Optional experiment name for this run')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to save complete log of the pipeline run (optional)')
    parser.add_argument('--not-leak', action='store_true',
                       help='Prevent temporal leakage in station arrival predictions (skip predictions when departure and arrival are in same time window)')
    
    args = parser.parse_args()
    
    # Set up logging before any output
    logger = setup_logging(args.log_file)
    
    # Log start time and arguments
    start_time = datetime.datetime.now()
    print("🚀 BIKE TRIP PREDICTION PIPELINE (OPTIMIZED)")
    print("="*80)
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Trips Path:       {args.trips_path}")
    print(f"👥 Users Path:       {args.users_path}")
    print(f"🤖 Model:            {args.model}")
    print(f"🏢 Station Arrivals: {'Enabled (ULTRA-FAST)' if args.arrivals else 'Disabled'}")
    if args.arrivals:
        print(f"⏰ Delta T:          {args.delta_t} seconds ({args.delta_t/60:.1f} minutes)")
        print(f"🚫 Prevent Leak:     {'Yes' if args.not_leak else 'No'}")
    print(f"📁 Output Dir:       {args.output_dir}")
    print(f"💾 Predictions Dir:  {args.predictions_dir}")
    if args.log_file:
        print(f"📝 Log File:         {args.log_file}")
    print("="*80)
    
    try:
        # Step 1: Feature Engineering
        print("\n🔄 STEP 1: FEATURE ENGINEERING")
        if not run_feature_engineering(args.trips_path, args.users_path, args.force_rebuild):
            print("❌ Pipeline failed at feature engineering step")
            return 1
        
        # Step 2: Model Training
        print("\n🔄 STEP 2: MODEL TRAINING")
        try:
            model, model_name = train_model(args.model, args.experiment_name)
        except Exception as e:
            print(f"❌ Pipeline failed at model training step: {e}")
            logging.exception("Model training failed with exception:")
            return 1
        
        # Step 3: Individual Trip Evaluation (with immediate prediction saving)
        print("\n🔄 STEP 3: INDIVIDUAL TRIP EVALUATION")
        try:
            eta_metrics, dest_metrics, prediction_files = evaluate_individual_trips(
                model, args.model, model_name, args.predictions_dir
            )
        except Exception as e:
            print(f"❌ Pipeline failed at individual trip evaluation step: {e}")
            logging.exception("Individual trip evaluation failed with exception:")
            return 1
        
        # Step 4: Station Arrival Evaluation (if enabled) - NOW WITH ULTRA-FAST SKELETON METHOD
        station_metrics = {}
        if args.arrivals:
            print("\n🔄 STEP 4: ULTRA-FAST STATION ARRIVAL EVALUATION")
            try:
                station_metrics = evaluate_station_arrivals_fast(
                    model, args.model, model_name, args.delta_t, prediction_files, args.not_leak, args.output_dir
                )
            except Exception as e:
                print(f"❌ Pipeline failed at station arrival evaluation step: {e}")
                logging.exception("Station arrival evaluation failed with exception:")
                return 1
        else:
            print("\n🔄 STEP 4: STATION ARRIVAL EVALUATION (SKIPPED)")
            print("   Station arrival analysis disabled (use --arrivals to enable)")
        
        # Step 5: Save Results
        print("\n🔄 STEP 5: SAVING RESULTS")
        try:
            save_results(eta_metrics, dest_metrics, station_metrics, model_name, args.output_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not save results: {e}")
            logging.exception("Results saving failed with exception:")
        
        # Final Summary
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 ULTRA-FAST OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏰ End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total duration: {duration}")
        print(f"📊 Model: {model_name}")
        print(f"📈 ETA R²: {eta_metrics['r2']:.4f}")
        print(f"🎯 Destination Accuracy: {dest_metrics['accuracy']:.4f}")
        
        if station_metrics:
            print(f"🏢 Station Arrival R²: {station_metrics['overall_r2']:.4f}")
            print(f"🏢 Station Correlation: {station_metrics['station_correlation']:.4f}")
        
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"💾 Predictions saved in: {args.predictions_dir}")
        if args.log_file:
            print(f"📝 Complete log saved to: {args.log_file}")
        
        # Show command to run standalone fast analysis
        test_file, eta_file, dest_file = prediction_files
        print(f"\n🚀 For future fast-only analysis, run:")
        print(f"python fast_station_arrivals.py \\")
        print(f"  --test-data {test_file} \\")
        print(f"  --eta-pred {eta_file} \\")
        print(f"  --dest-pred {dest_file} \\")
        print(f"  --delta-t {args.delta_t}{' \\' if args.not_leak else ''}")
        if args.not_leak:
            print(f"  --not-leak")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed with unexpected error: {e}")
        logging.exception("Pipeline failed with unexpected exception:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 