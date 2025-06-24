"""
Ultra-fast station arrival analysis using multiprocessing and optimized Polars operations.
Designed for high-performance machines with many cores and lots of RAM.

This script takes model predictions and ground truth to calculate station arrival counts
in parallel across time windows, leveraging all available CPU cores.

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
from functools import partial
import pandas as pd


def process_time_chunk_vectorized(
    chunk_data: Tuple[pl.DataFrame, int, List[int], int, int]
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Process a chunk of time points using vectorized operations.
    
    Args:
        chunk_data: Tuple of (df_chunk, delta_t, station_list, start_idx, end_idx)
        
    Returns:
        Tuple of (chunk_start_idx, true_arrivals_chunk, pred_arrivals_chunk)
    """
    df_chunk, delta_t, station_list, start_idx, end_idx = chunk_data
    
    n_timepoints = end_idx - start_idx
    n_stations = len(station_list)
    
    # Initialize result arrays for this chunk
    true_arrivals = np.zeros((n_timepoints, n_stations), dtype=np.int32)
    pred_arrivals = np.zeros((n_timepoints, n_stations), dtype=np.float32)
    
    # Get unique time points for this chunk
    unique_times = df_chunk.select("fecha_origen_recorrido").unique().sort("fecha_origen_recorrido")
    
    if len(unique_times) == 0:
        return start_idx, true_arrivals, pred_arrivals
    
    # Convert to pandas for faster time operations in this specific case
    df_pd = df_chunk.to_pandas()
    unique_times_pd = unique_times.to_pandas()["fecha_origen_recorrido"]
    
    station_to_idx = {station: idx for idx, station in enumerate(station_list)}
    
    # Process each time point in the chunk
    for i, current_time in enumerate(unique_times_pd):
        if i >= n_timepoints:
            break
            
        window_start = current_time
        window_end = current_time + pd.Timedelta(seconds=delta_t)
        
        # Vectorized filtering for arrivals in window
        mask_true = (
            (df_pd['arrival_time'] >= window_start) & 
            (df_pd['arrival_time'] < window_end)
        )
        
        mask_pred = (
            (df_pd['predicted_arrival_time'] >= window_start) & 
            (df_pd['predicted_arrival_time'] < window_end)
        )
        
        # Count true arrivals
        if mask_true.sum() > 0:
            true_arrivals_in_window = df_pd.loc[mask_true, 'id_estacion_destino']
            true_counts = true_arrivals_in_window.value_counts()
            
            for station, count in true_counts.items():
                if station in station_to_idx:
                    true_arrivals[i, station_to_idx[station]] = count
        
        # Count predicted arrivals
        if mask_pred.sum() > 0:
            pred_arrivals_in_window = df_pd.loc[mask_pred, 'dest_pred']
            # Filter out invalid predictions
            valid_mask = pred_arrivals_in_window.isin(station_list)
            if valid_mask.sum() > 0:
                pred_counts = pred_arrivals_in_window[valid_mask].value_counts()
                
                for station, count in pred_counts.items():
                    if station in station_to_idx:
                        pred_arrivals[i, station_to_idx[station]] = count
    
    return start_idx, true_arrivals, pred_arrivals


def calculate_station_arrivals_parallel(
    df: pl.DataFrame,
    eta_predictions: np.ndarray,
    dest_predictions: np.ndarray,
    delta_t: int = 1800,
    n_processes: int = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Calculate station arrival targets and predictions using parallel processing.
    
    Args:
        df: Test dataframe with trip data
        eta_predictions: Array of predicted trip durations
        dest_predictions: Array of predicted destinations
        delta_t: Time window in seconds
        n_processes: Number of processes (default: all cores)
        
    Returns:
        Tuple of (true_arrivals, pred_arrivals, station_list)
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"🚀 Using {n_processes} processes for parallel computation")
    print(f"📊 Processing {len(df):,} trips with {delta_t/60:.1f}-minute windows")
    
    # Add predictions to dataframe efficiently
    print("📈 Adding predictions to dataframe...")
    df_with_preds = df.with_columns([
        pl.Series("eta_pred", eta_predictions),
        pl.Series("dest_pred", dest_predictions)
    ])
    
    # Calculate arrival times efficiently using Polars
    print("⏱️  Calculating arrival times...")
    df_with_arrivals = df_with_preds.with_columns([
        # True arrival time
        (pl.col("fecha_origen_recorrido") + pl.duration(seconds=pl.col("duracion_recorrido"))).alias("arrival_time"),
        # Predicted arrival time  
        (pl.col("fecha_origen_recorrido") + pl.duration(seconds=pl.col("eta_pred"))).alias("predicted_arrival_time")
    ])
    
    # Get station list and time points
    station_list = sorted(df_with_arrivals.select("id_estacion_destino").unique().to_series().to_list())
    unique_times = df_with_arrivals.select("fecha_origen_recorrido").unique().sort("fecha_origen_recorrido").to_series()
    
    n_stations = len(station_list)
    n_timepoints = len(unique_times)
    
    print(f"🗺️  Found {n_stations} stations and {n_timepoints:,} unique time points")
    
    # Split work into chunks for parallel processing
    chunk_size = max(1000, n_timepoints // (n_processes * 4))  # Ensure reasonable chunk sizes
    chunks = []
    
    print(f"📦 Creating chunks of size ~{chunk_size:,} time points...")
    
    for start_idx in range(0, n_timepoints, chunk_size):
        end_idx = min(start_idx + chunk_size, n_timepoints)
        
        # Get time range for this chunk
        chunk_start_time = unique_times[start_idx]
        chunk_end_time = unique_times[end_idx - 1] if end_idx < n_timepoints else unique_times[-1]
        
        # Add buffer for arrival time calculations
        buffer_time = pd.Timedelta(seconds=delta_t * 2)  # 2x delta_t buffer
        extended_end_time = chunk_end_time + buffer_time
        
        # Filter dataframe for this time chunk (with buffer)
        chunk_df = df_with_arrivals.filter(
            (pl.col("fecha_origen_recorrido") >= chunk_start_time) &
            (pl.col("fecha_origen_recorrido") <= extended_end_time)
        )
        
        if len(chunk_df) > 0:
            chunk_data = (chunk_df, delta_t, station_list, start_idx, end_idx)
            chunks.append(chunk_data)
    
    print(f"📦 Created {len(chunks)} chunks for parallel processing")
    
    # Initialize result arrays
    true_arrivals_full = np.zeros((n_timepoints, n_stations), dtype=np.int32)
    pred_arrivals_full = np.zeros((n_timepoints, n_stations), dtype=np.float32)
    
    # Process chunks in parallel
    print("🔥 Starting parallel processing...")
    start_time = time.time()
    
    if len(chunks) == 1:
        # Single chunk - no need for multiprocessing overhead
        print("   Using single-threaded processing for single chunk")
        start_idx, true_chunk, pred_chunk = process_time_chunk_vectorized(chunks[0])
        chunk_size_actual = true_chunk.shape[0]
        true_arrivals_full[start_idx:start_idx + chunk_size_actual] = true_chunk
        pred_arrivals_full[start_idx:start_idx + chunk_size_actual] = pred_chunk
    else:
        # Multi-chunk - use multiprocessing
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(process_time_chunk_vectorized, chunks)
        
        # Combine results
        print("🔗 Combining results from parallel chunks...")
        for start_idx, true_chunk, pred_chunk in results:
            chunk_size_actual = true_chunk.shape[0]
            end_idx = start_idx + chunk_size_actual
            if end_idx <= n_timepoints:
                true_arrivals_full[start_idx:end_idx] = true_chunk
                pred_arrivals_full[start_idx:end_idx] = pred_chunk
    
    elapsed_time = time.time() - start_time
    print(f"⚡ Parallel processing completed in {elapsed_time:.1f} seconds")
    print(f"📊 Processed {n_timepoints:,} time points across {n_stations} stations")
    
    return true_arrivals_full, pred_arrivals_full, station_list


def evaluate_station_arrivals_fast(
    true_arrivals: np.ndarray,
    pred_arrivals: np.ndarray,
    station_list: List[int]
) -> Dict[str, float]:
    """Fast evaluation of station arrival predictions."""
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
        description="Ultra-fast station arrival analysis using multiprocessing"
    )
    
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test dataset parquet file')
    parser.add_argument('--eta-pred', type=str, required=True,
                       help='Path to ETA predictions numpy file')
    parser.add_argument('--dest-pred', type=str, required=True,
                       help='Path to destination predictions numpy file')
    parser.add_argument('--delta-t', type=int, default=1800,
                       help='Time window in seconds (default: 1800 = 30 minutes)')
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of processes (default: all available cores)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🚀 ULTRA-FAST STATION ARRIVAL ANALYSIS")
    print("="*80)
    print(f"💻 Available CPU cores: {mp.cpu_count()}")
    print(f"⚡ Using processes: {args.n_processes or mp.cpu_count()}")
    print(f"📂 Test data: {args.test_data}")
    print(f"🎯 ETA predictions: {args.eta_pred}")
    print(f"🎯 Dest predictions: {args.dest_pred}")
    print(f"⏰ Delta T: {args.delta_t} seconds ({args.delta_t/60:.1f} minutes)")
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
    
    # Validate shapes match
    if len(df) != len(eta_pred) or len(df) != len(dest_pred):
        raise ValueError("Data shapes don't match!")
    
    # Calculate arrivals
    print("\n🔥 Starting ultra-fast station arrival calculation...")
    true_arrivals, pred_arrivals, station_list = calculate_station_arrivals_parallel(
        df, eta_pred, dest_pred, args.delta_t, args.n_processes
    )
    
    # Evaluate
    print("\n📊 Evaluating results...")
    metrics = evaluate_station_arrivals_fast(true_arrivals, pred_arrivals, station_list)
    
    # Print results
    print(f"\n🎉 STATION ARRIVAL ANALYSIS RESULTS")
    print("="*80)
    print(f"⏱️  Overall MSE: {metrics['overall_mse']:.2f}")
    print(f"⏱️  Overall RMSE: {metrics['overall_rmse']:.2f}")
    print(f"⏱️  Overall MAE: {metrics['overall_mae']:.2f}")
    print(f"⏱️  Overall R²: {metrics['overall_r2']:.4f}")
    print(f"📊 Total true arrivals: {metrics['total_arrivals_true']:.0f}")
    print(f"📊 Total pred arrivals: {metrics['total_arrivals_pred']:.0f}")
    print(f"🔗 Station correlation: {metrics['station_correlation']:.4f}")
    print(f"🗺️  Number of stations: {metrics['n_stations']}")
    print(f"⏰ Number of timepoints: {metrics['n_timepoints']:,}")
    
    # Save results
    print(f"\n💾 Saving results...")
    save_results_fast(metrics, true_arrivals, pred_arrivals, station_list, args.output_dir)
    
    total_time = time.time() - start_total
    print(f"\n🎉 ANALYSIS COMPLETED!")
    print(f"⚡ Total execution time: {total_time:.1f} seconds")
    print(f"🚀 Processing rate: {metrics['n_timepoints']*metrics['n_stations']/total_time:,.0f} calculations/second")


if __name__ == "__main__":
    main() 