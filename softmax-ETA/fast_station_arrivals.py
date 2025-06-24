"""
Ultra-fast station arrival analysis using skeleton-based slot filling.
This approach creates empty matrices and fills slots where arrivals occur - much faster than complex chunking.

Usage:
    python fast_station_arrivals.py --test-data trips_test.parquet --eta-pred eta_predictions.npy --dest-pred dest_predictions.npy --delta-t 1800
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Tuple
import time
import pandas as pd


def create_time_station_skeleton(
    df: pl.DataFrame,
    delta_t: int = 1800
) -> Tuple[np.ndarray, np.ndarray, List[int], List]:
    """
    Create a skeleton (empty matrices) for time windows and stations.
    
    Args:
        df: Test dataframe
        delta_t: Time window in seconds
        
    Returns:
        Tuple of (time_windows, time_to_idx, station_list, station_to_idx)
    """
    print(f"🏗️  Creating time-station skeleton with {delta_t/60:.1f}-minute windows...")
    
    # Get all unique times and stations
    df_pd = df.to_pandas()
    df_pd = df_pd.sort_values("fecha_origen_recorrido")
    
    # Create time windows (round down to delta_t intervals)
    df_pd['window_start'] = (df_pd['fecha_origen_recorrido'].astype('int64') // (delta_t * 1_000_000_000)) * (delta_t * 1_000_000_000)
    df_pd['window_start'] = pd.to_datetime(df_pd['window_start'])
    
    # Get unique time windows and stations
    unique_windows = sorted(df_pd['window_start'].unique())
    unique_stations = sorted(df_pd['id_estacion_destino'].unique())
    
    # Create mapping dictionaries
    time_to_idx = {time: idx for idx, time in enumerate(unique_windows)}
    station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
    
    print(f"📊 Created skeleton: {len(unique_windows):,} time windows × {len(unique_stations)} stations")
    
    return np.array(unique_windows), time_to_idx, unique_stations, station_to_idx


def calculate_station_arrivals_skeleton(
    df: pl.DataFrame,
    eta_predictions: np.ndarray,
    dest_predictions: np.ndarray,
    dest_probabilities: np.ndarray = None,
    delta_t: int = 1800,
    use_probabilities: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Calculate station arrivals using simple skeleton approach.
    
    Args:
        df: Test dataframe with trip data
        eta_predictions: Array of predicted trip durations
        dest_predictions: Array of predicted destinations (for hard predictions)
        dest_probabilities: Array of destination probabilities (for soft predictions)
        delta_t: Time window in seconds
        use_probabilities: Whether to use probability distributions
        
    Returns:
        Tuple of (true_arrivals, pred_arrivals, station_list)
    """
    prediction_type = "probability-weighted" if use_probabilities else "hard predictions"
    print(f"🚀 Using skeleton approach for {len(df):,} trips")
    print(f"🎯 Prediction method: {prediction_type}")
    
    # Convert to pandas for easier operations
    df_pd = df.to_pandas()
    df_pd = df_pd.sort_values("fecha_origen_recorrido").reset_index(drop=True)
    
    # Add predictions
    df_pd['eta_pred'] = eta_predictions
    df_pd['dest_pred'] = dest_predictions
    
    # Calculate arrival times
    print("⏱️  Calculating arrival times...")
    df_pd['arrival_time'] = df_pd['fecha_origen_recorrido'] + pd.to_timedelta(df_pd['duracion_recorrido'], unit='s')
    df_pd['predicted_arrival_time'] = df_pd['fecha_origen_recorrido'] + pd.to_timedelta(df_pd['eta_pred'], unit='s')
    
    # Create time windows for arrivals
    df_pd['true_arrival_window'] = (df_pd['arrival_time'].astype('int64') // (delta_t * 1_000_000_000)) * (delta_t * 1_000_000_000)
    df_pd['pred_arrival_window'] = (df_pd['predicted_arrival_time'].astype('int64') // (delta_t * 1_000_000_000)) * (delta_t * 1_000_000_000)
    df_pd['true_arrival_window'] = pd.to_datetime(df_pd['true_arrival_window'])
    df_pd['pred_arrival_window'] = pd.to_datetime(df_pd['pred_arrival_window'])
    
    # Get unique windows and stations
    all_windows = sorted(set(df_pd['true_arrival_window'].unique()) | set(df_pd['pred_arrival_window'].unique()))
    station_list = sorted(df_pd['id_estacion_destino'].unique())
    
    # Create mapping dictionaries  
    time_to_idx = {time: idx for idx, time in enumerate(all_windows)}
    station_to_idx = {station: idx for idx, station in enumerate(station_list)}
    
    n_windows = len(all_windows)
    n_stations = len(station_list)
    
    print(f"🗺️  Processing {n_windows:,} time windows × {n_stations} stations")
    
    # Initialize skeleton matrices
    true_arrivals = np.zeros((n_windows, n_stations), dtype=np.int32)
    pred_arrivals = np.zeros((n_windows, n_stations), dtype=np.float32)
    
    print("🔥 Filling arrival slots...")
    start_time = time.time()
    
    # Fill true arrivals - simple slot filling
    print("   📊 Processing true arrivals...")
    for _, row in df_pd.iterrows():
        time_idx = time_to_idx.get(row['true_arrival_window'])
        station_idx = station_to_idx.get(row['id_estacion_destino'])
        
        if time_idx is not None and station_idx is not None:
            true_arrivals[time_idx, station_idx] += 1
    
    # Fill predicted arrivals
    print("   🎯 Processing predicted arrivals...")
    if use_probabilities and dest_probabilities is not None:
        # Probability-weighted approach
        print("   🎲 Using probability distributions...")
        for i, row in df_pd.iterrows():
            time_idx = time_to_idx.get(row['pred_arrival_window'])
            
            if time_idx is not None and i < len(dest_probabilities):
                trip_probs = dest_probabilities[i]
                
                # Distribute probability across stations
                for station_idx, prob in enumerate(trip_probs):
                    if station_idx < n_stations and prob > 0:
                        pred_arrivals[time_idx, station_idx] += prob
    else:
        # Hard predictions approach
        print("   🎯 Using hard predictions...")
        for _, row in df_pd.iterrows():
            time_idx = time_to_idx.get(row['pred_arrival_window'])
            station_idx = station_to_idx.get(row['dest_pred'])
            
            if time_idx is not None and station_idx is not None:
                pred_arrivals[time_idx, station_idx] += 1
    
    elapsed_time = time.time() - start_time
    print(f"⚡ Skeleton filling completed in {elapsed_time:.1f} seconds")
    print(f"📊 Average arrivals per window: True={true_arrivals.mean():.2f}, Pred={pred_arrivals.mean():.2f}")
    
    return true_arrivals, pred_arrivals, station_list


def evaluate_station_arrivals_fast(
    true_arrivals: np.ndarray,
    pred_arrivals: np.ndarray,
    station_list: List[int]
) -> Dict[str, float]:
    """Fast evaluation of station arrival predictions using vectorized operations."""
    print(f"📊 Evaluating station arrival predictions...")
    
    # Flatten for overall metrics
    true_flat = true_arrivals.flatten()
    pred_flat = pred_arrivals.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(true_flat) | np.isnan(pred_flat))
    true_clean = true_flat[mask]
    pred_clean = pred_flat[mask]
    
    if len(true_clean) == 0:
        return {"error": "No valid data for evaluation"}
    
    # Calculate metrics efficiently using vectorized operations
    mse = np.mean((true_clean - pred_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_clean - pred_clean))
    
    # R² calculation
    ss_res = np.sum((true_clean - pred_clean) ** 2)
    ss_tot = np.sum((true_clean - np.mean(true_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Total counts
    total_true = np.sum(true_arrivals)
    total_pred = np.sum(pred_arrivals)
    
    # Per-station metrics (vectorized)
    station_totals_true = np.sum(true_arrivals, axis=0)
    station_totals_pred = np.sum(pred_arrivals, axis=0)
    
    # Station correlation
    if np.std(station_totals_true) > 0 and np.std(station_totals_pred) > 0:
        station_correlation = np.corrcoef(station_totals_true, station_totals_pred)[0, 1]
    else:
        station_correlation = 0.0
    
    return {
        "overall_mse": mse,
        "overall_rmse": rmse,
        "overall_mae": mae,
        "overall_r2": r2,
        "total_arrivals_true": total_true,
        "total_arrivals_pred": total_pred,
        "station_correlation": station_correlation,
        "n_stations": len(station_list),
        "n_timepoints": true_arrivals.shape[0]
    }


def save_results_fast(
    metrics: Dict[str, float],
    true_arrivals: np.ndarray,
    pred_arrivals: np.ndarray,
    station_list: List[int],
    output_dir: str = "results"
):
    """Save results efficiently."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_file = output_path / "fast_station_arrivals_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"📊 Metrics saved to: {metrics_file}")
    
    # Save arrival matrices (compressed)
    np.savez_compressed(
        output_path / "station_arrivals_data.npz",
        true_arrivals=true_arrivals,
        pred_arrivals=pred_arrivals,
        station_list=np.array(station_list)
    )
    print(f"💾 Arrival data saved to: {output_path / 'station_arrivals_data.npz'}")


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-fast station arrival analysis using skeleton approach"
    )
    
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test dataset parquet file')
    parser.add_argument('--eta-pred', type=str, required=True,
                       help='Path to ETA predictions numpy file')
    parser.add_argument('--dest-pred', type=str, required=True,
                       help='Path to destination predictions numpy file')
    parser.add_argument('--dest-proba', type=str, default=None,
                       help='Path to destination probabilities numpy file (optional, for --use-probabilities)')
    parser.add_argument('--delta-t', type=int, default=1800,
                       help='Time window in seconds (default: 1800 = 30 minutes)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--use-probabilities', action='store_true',
                       help='Use destination probabilities instead of hard predictions (requires --dest-proba)')
    
    args = parser.parse_args()
    
    print("🚀 ULTRA-FAST STATION ARRIVAL ANALYSIS (SKELETON METHOD)")
    print("="*80)
    print(f"📂 Test data: {args.test_data}")
    print(f"🎯 ETA predictions: {args.eta_pred}")
    print(f"🎯 Dest predictions: {args.dest_pred}")
    if args.use_probabilities:
        print(f"🎯 Dest probabilities: {args.dest_proba}")
    print(f"⏰ Delta T: {args.delta_t} seconds ({args.delta_t/60:.1f} minutes)")
    print(f"🎲 Use probabilities: {'Yes' if args.use_probabilities else 'No'}")
    print("="*80)
    
    start_total = time.time()
    
    # Load data
    print("\n📂 Loading data...")
    df = pl.read_parquet(args.test_data)
    eta_pred = np.load(args.eta_pred)
    dest_pred = np.load(args.dest_pred)
    
    print(f"✅ Loaded {len(df):,} trips")
    print(f"✅ ETA predictions shape: {eta_pred.shape}")
    print(f"✅ Destination predictions shape: {dest_pred.shape}")
    
    # Load probabilities if requested
    dest_proba = None
    if args.use_probabilities:
        if args.dest_proba is None:
            # Try to find probabilities file automatically
            proba_path = args.dest_pred.replace('dest_predictions.npy', 'dest_probabilities.npy')
            if Path(proba_path).exists():
                args.dest_proba = proba_path
                print(f"📂 Auto-detected probabilities file: {proba_path}")
            else:
                raise ValueError("--use-probabilities requires --dest-proba argument or dest_probabilities.npy in same directory")
        
        dest_proba = np.load(args.dest_proba)
        print(f"✅ Destination probabilities shape: {dest_proba.shape}")
    
    # Validate shapes match
    if len(df) != len(eta_pred) or len(df) != len(dest_pred):
        raise ValueError("Data shapes don't match!")
    
    if dest_proba is not None and len(df) != len(dest_proba):
        raise ValueError("Probabilities shape doesn't match data!")
    
    # Calculate arrivals using skeleton method
    print("\n🔥 Starting skeleton-based station arrival calculation...")
    true_arrivals, pred_arrivals, station_list = calculate_station_arrivals_skeleton(
        df, eta_pred, dest_pred, dest_proba, args.delta_t, args.use_probabilities
    )
    
    # Evaluate
    print("\n📊 Evaluating results...")
    metrics = evaluate_station_arrivals_fast(true_arrivals, pred_arrivals, station_list)
    
    # Print results
    prediction_method = "Probability-weighted" if args.use_probabilities else "Hard predictions"
    print(f"\n🎉 STATION ARRIVAL ANALYSIS RESULTS ({prediction_method})")
    print("="*80)
    print(f"⏱️  Overall MSE: {metrics['overall_mse']:.2f}")
    print(f"⏱️  Overall RMSE: {metrics['overall_rmse']:.2f}")
    print(f"⏱️  Overall MAE: {metrics['overall_mae']:.2f}")
    print(f"⏱️  Overall R²: {metrics['overall_r2']:.4f}")
    print(f"📊 Total true arrivals: {metrics['total_arrivals_true']:.0f}")
    print(f"📊 Total pred arrivals: {metrics['total_arrivals_pred']:.1f}")
    print(f"🔗 Station correlation: {metrics['station_correlation']:.4f}")
    print(f"🗺️  Number of stations: {metrics['n_stations']}")
    print(f"⏰ Number of timepoints: {metrics['n_timepoints']:,}")
    
    # Save results
    print(f"\n💾 Saving results...")
    metrics['prediction_method'] = prediction_method
    save_results_fast(metrics, true_arrivals, pred_arrivals, station_list, args.output_dir)
    
    total_time = time.time() - start_total
    print(f"\n🎉 ANALYSIS COMPLETED!")
    print(f"⚡ Total execution time: {total_time:.1f} seconds")
    print(f"🚀 Processing rate: {len(df)/total_time:,.0f} trips/second")


if __name__ == "__main__":
    main() 