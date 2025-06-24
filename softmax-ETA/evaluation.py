"""
Evaluation utilities for ETA prediction and destination prediction models.
This module provides reusable functions for computing metrics across different model types.
"""

import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# Global variables for multiprocessing worker functions
_df_pd_global = None
_station_to_idx_global = None
_n_stations_global = None
_destination_col_global = None
_delta_t_global = None


def _process_time_chunk_true_worker(chunk_times):
    """
    Worker function for processing time chunks for true arrivals.
    This needs to be at module level to be pickleable by multiprocessing.
    """
    global _df_pd_global, _station_to_idx_global, _n_stations_global, _destination_col_global, _delta_t_global
    
    chunk_results = []
    
    for current_time in chunk_times:
        window_start = current_time
        window_end = current_time + pd.Timedelta(seconds=_delta_t_global)
        
        # Find arrivals in this window
        arrivals_mask = (
            (_df_pd_global['arrival_time'] >= window_start) & 
            (_df_pd_global['arrival_time'] < window_end)
        )
        
        if arrivals_mask.sum() > 0:
            arrivals_in_window = _df_pd_global.loc[arrivals_mask, _destination_col_global]
            station_counts = arrivals_in_window.value_counts()
            
            # Convert to station array
            station_array = np.zeros(_n_stations_global, dtype=np.int32)
            for station, count in station_counts.items():
                if station in _station_to_idx_global:
                    station_array[_station_to_idx_global[station]] = count
            
            chunk_results.append((current_time, station_array))
        else:
            chunk_results.append((current_time, np.zeros(_n_stations_global, dtype=np.int32)))
    
    return chunk_results


def _process_time_chunk_pred_worker(chunk_times):
    """
    Worker function for processing time chunks for predicted arrivals.
    This needs to be at module level to be pickleable by multiprocessing.
    """
    global _df_pd_global, _station_to_idx_global, _n_stations_global, _delta_t_global
    
    chunk_results = []
    
    for current_time in chunk_times:
        window_start = current_time
        window_end = current_time + pd.Timedelta(seconds=_delta_t_global)
        
        # Find predicted arrivals in this window
        pred_arrivals_mask = (
            (_df_pd_global['predicted_arrival_time'] >= window_start) & 
            (_df_pd_global['predicted_arrival_time'] < window_end)
        )
        
        if pred_arrivals_mask.sum() > 0:
            pred_arrivals_in_window = _df_pd_global.loc[pred_arrivals_mask, 'dest_pred']
            # Filter out predictions not in our station list
            valid_preds = pred_arrivals_in_window[pred_arrivals_in_window.isin(list(_station_to_idx_global.keys()))]
            
            if len(valid_preds) > 0:
                pred_counts = valid_preds.value_counts()
                
                # Convert to station array
                station_array = np.zeros(_n_stations_global, dtype=np.float32)
                for station, count in pred_counts.items():
                    if station in _station_to_idx_global:
                        station_array[_station_to_idx_global[station]] = count
                
                chunk_results.append((current_time, station_array))
            else:
                chunk_results.append((current_time, np.zeros(_n_stations_global, dtype=np.float32)))
        else:
            chunk_results.append((current_time, np.zeros(_n_stations_global, dtype=np.float32)))
    
    return chunk_results


def evaluate_eta_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ETA (duration) prediction performance.
    
    Args:
        y_true: True duration values
        y_pred: Predicted duration values
        
    Returns:
        Dictionary with MSE and R2 metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {"mse": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    return {
        "mse": mse,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int, class_labels: Optional[np.ndarray] = None) -> float:
    """
    Calculate top-k accuracy for multi-class classification.
    
    Args:
        y_true: True class labels (original labels, not encoded)
        y_proba: Prediction probabilities matrix (n_samples, n_classes)
        k: Number of top predictions to consider
        class_labels: Array of class labels corresponding to probability columns (optional)
        
    Returns:
        Top-k accuracy score
    """
    if y_proba.ndim == 1:
        # Binary case - convert to 2D
        y_proba = np.column_stack([1 - y_proba, y_proba])
    
    # Get top k prediction indices
    top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
    
    # If class_labels provided, map indices to actual labels
    if class_labels is not None:
        top_k_preds = class_labels[top_k_indices]
    else:
        # Assume labels are indices 0, 1, 2, ...
        top_k_preds = top_k_indices
    
    # Check if true label is in top k predictions
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def evaluate_destination_prediction(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None,
    class_labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Evaluate destination prediction performance.
    
    Args:
        y_true: True destination station IDs
        y_pred: Predicted destination station IDs
        y_proba: Prediction probabilities (optional, for top-k metrics)
        class_labels: List of all possible class labels (optional, needed for top-k accuracy)
        
    Returns:
        Dictionary with precision, recall, accuracy, and top-k accuracy metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-class, use macro averaging
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    # Top-k accuracy if probabilities are provided
    if y_proba is not None:
        try:
            class_labels_array = np.array(class_labels) if class_labels is not None else None
            metrics["top_5_accuracy"] = top_k_accuracy(y_true, y_proba, k=5, class_labels=class_labels_array)
            metrics["top_10_accuracy"] = top_k_accuracy(y_true, y_proba, k=10, class_labels=class_labels_array)
        except Exception as e:
            print(f"Warning: Could not compute top-k accuracy: {e}")
            metrics["top_5_accuracy"] = np.nan
            metrics["top_10_accuracy"] = np.nan
    else:
        metrics["top_5_accuracy"] = np.nan
        metrics["top_10_accuracy"] = np.nan
    
    return metrics


def print_evaluation_results(eta_metrics: Dict[str, float], dest_metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print formatted evaluation results for both tasks.
    
    Args:
        eta_metrics: ETA prediction metrics
        dest_metrics: Destination prediction metrics
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📊 ETA PREDICTION METRICS:")
    print(f"  MSE:  {eta_metrics['mse']:.2f}")
    print(f"  RMSE: {eta_metrics['rmse']:.2f}")
    print(f"  MAE:  {eta_metrics['mae']:.2f}")
    print(f"  R²:   {eta_metrics['r2']:.4f}")
    
    print(f"\n🎯 DESTINATION PREDICTION METRICS:")
    print(f"  Accuracy:       {dest_metrics['accuracy']:.4f}")
    print(f"  Precision:      {dest_metrics['precision']:.4f}")
    print(f"  Recall:         {dest_metrics['recall']:.4f}")
    
    if not np.isnan(dest_metrics['top_5_accuracy']):
        print(f"  Top-5 Accuracy: {dest_metrics['top_5_accuracy']:.4f}")
    if not np.isnan(dest_metrics['top_10_accuracy']):
        print(f"  Top-10 Accuracy: {dest_metrics['top_10_accuracy']:.4f}")


def save_evaluation_results(
    eta_metrics: Dict[str, float], 
    dest_metrics: Dict[str, float], 
    model_name: str,
    output_path: str
):
    """
    Save evaluation results to CSV file.
    
    Args:
        eta_metrics: ETA prediction metrics
        dest_metrics: Destination prediction metrics
        model_name: Name of the model
        output_path: Path to save the results
    """
    # Combine all metrics
    all_metrics = {
        "model": model_name,
        **{f"eta_{k}": v for k, v in eta_metrics.items()},
        **{f"dest_{k}": v for k, v in dest_metrics.items()}
    }
    
    # Convert to DataFrame and save
    df = pd.DataFrame([all_metrics])
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")


def compare_models(results_df: pd.DataFrame):
    """
    Compare multiple models from a results DataFrame.
    
    Args:
        results_df: DataFrame with model comparison results
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    # ETA metrics comparison
    print(f"\n📊 ETA PREDICTION COMPARISON:")
    eta_cols = [col for col in results_df.columns if col.startswith('eta_')]
    eta_comparison = results_df[['model'] + eta_cols].round(4)
    print(eta_comparison.to_string(index=False))
    
    # Destination metrics comparison
    print(f"\n🎯 DESTINATION PREDICTION COMPARISON:")
    dest_cols = [col for col in results_df.columns if col.startswith('dest_')]
    dest_comparison = results_df[['model'] + dest_cols].round(4)
    print(dest_comparison.to_string(index=False))
    
    # Highlight best models
    print(f"\n🏆 BEST PERFORMERS:")
    best_r2_idx = results_df['eta_r2'].idxmax()
    best_accuracy_idx = results_df['dest_accuracy'].idxmax()
    
    print(f"  Best ETA (R²):        {results_df.loc[best_r2_idx, 'model']} (R² = {results_df.loc[best_r2_idx, 'eta_r2']:.4f})")
    print(f"  Best Destination:     {results_df.loc[best_accuracy_idx, 'model']} (Acc = {results_df.loc[best_accuracy_idx, 'dest_accuracy']:.4f})")


def load_and_prepare_data(
    train_path: str, 
    val_path: str, 
    test_path: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load and prepare train/val/test datasets.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data  
        test_path: Path to test data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("Loading datasets...")
    
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)
    
    print(f"  Train: {len(train_df):,} trips")
    print(f"  Val:   {len(val_df):,} trips")
    print(f"  Test:  {len(test_df):,} trips")
    
    return train_df, val_df, test_df 


def create_station_arrival_targets(
    df: pl.DataFrame, 
    delta_t: int = 1800,  # Default 30 minutes in seconds
    timestamp_col: str = "fecha_origen_recorrido",
    duration_col: str = "duracion_recorrido",
    destination_col: str = "id_estacion_destino"
) -> Tuple[np.ndarray, List[int]]:
    """
    Create station-level arrival count targets for the next delta_T time window.
    
    This function aggregates individual trip destinations into station-level arrival counts
    by counting how many arrivals each station will receive in the next delta_T time window.
    
    Args:
        df: DataFrame with trip data including timestamps, durations, and destinations
        delta_t: Time window in seconds (default: 1800 = 30 minutes)
        timestamp_col: Column name for trip start timestamp
        duration_col: Column name for trip duration in seconds
        destination_col: Column name for destination station ID
        
    Returns:
        Tuple of (arrival_targets, station_list) where:
        - arrival_targets: Array of shape (n_timestamps, n_stations) with arrival counts
        - station_list: List of station IDs corresponding to array columns
    """
    print(f"Creating station arrival targets for delta_T = {delta_t} seconds ({delta_t/60:.1f} minutes)")
    
    # Convert to pandas for easier time-based operations
    df_pd = df.to_pandas()
    
    # Sort by timestamp
    df_pd = df_pd.sort_values(timestamp_col)
    
    # Calculate arrival time for each trip (start time + duration)
    df_pd['arrival_time'] = df_pd[timestamp_col] + pd.to_timedelta(df_pd[duration_col], unit='s')
    
    # Get unique stations and timestamps
    unique_stations = sorted(df_pd[destination_col].unique())
    n_stations = len(unique_stations)
    station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
    
    # Get unique time windows (we'll create targets for each trip start time)
    trip_times = df_pd[timestamp_col].unique()
    n_times = len(trip_times)
    
    print(f"Processing {n_times:,} time points and {n_stations} stations")
    
    # Initialize target array
    arrival_targets = np.zeros((n_times, n_stations), dtype=np.int32)
    
    # For each trip start time, count arrivals in the next delta_T window
    for i, current_time in enumerate(trip_times):
        if i % 1000 == 0:
            print(f"  Processing time point {i+1:,}/{n_times:,}")
        
        # Define the time window for counting arrivals
        window_start = current_time
        window_end = current_time + pd.Timedelta(seconds=delta_t)
        
        # Find all trips that will arrive in this time window
        arrivals_in_window = df_pd[
            (df_pd['arrival_time'] >= window_start) & 
            (df_pd['arrival_time'] < window_end)
        ]
        
        # Count arrivals per station
        if len(arrivals_in_window) > 0:
            station_counts = arrivals_in_window[destination_col].value_counts()
            for station, count in station_counts.items():
                if station in station_to_idx:
                    arrival_targets[i, station_to_idx[station]] = count
    
    print(f"Created arrival targets with shape {arrival_targets.shape}")
    print(f"Average arrivals per time window: {arrival_targets.mean():.2f}")
    print(f"Max arrivals per station per window: {arrival_targets.max()}")
    
    return arrival_targets, unique_stations


def create_predicted_station_arrivals(
    df: pl.DataFrame,
    eta_predictions: np.ndarray,
    dest_predictions: np.ndarray,
    delta_t: int = 1800,
    timestamp_col: str = "fecha_origen_recorrido",
    destination_col: str = "id_estacion_destino"
) -> Tuple[np.ndarray, List[int]]:
    """
    Create station-level arrival count predictions using model outputs.
    
    This function uses ETA and destination predictions to estimate how many arrivals
    each station will receive in the next delta_T time window.
    
    Args:
        df: DataFrame with trip data including timestamps
        eta_predictions: Array of predicted trip durations in seconds
        dest_predictions: Array of predicted destination station IDs
        delta_t: Time window in seconds (default: 1800 = 30 minutes)
        timestamp_col: Column name for trip start timestamp
        destination_col: Column name for actual destination station ID (for reference)
        
    Returns:
        Tuple of (predicted_arrivals, station_list) where:
        - predicted_arrivals: Array of shape (n_timestamps, n_stations) with predicted counts
        - station_list: List of station IDs corresponding to array columns
    """
    print(f"Creating predicted station arrivals for delta_T = {delta_t} seconds ({delta_t/60:.1f} minutes)")
    
    # Convert to pandas for easier operations
    df_pd = df.to_pandas()
    
    # Add predictions to dataframe
    df_pd['eta_pred'] = eta_predictions
    df_pd['dest_pred'] = dest_predictions
    
    # Sort by timestamp
    df_pd = df_pd.sort_values(timestamp_col)
    
    # Calculate predicted arrival time for each trip
    df_pd['predicted_arrival_time'] = df_pd[timestamp_col] + pd.to_timedelta(df_pd['eta_pred'], unit='s')
    
    # Get unique stations (use actual destinations to ensure consistency)
    unique_stations = sorted(df_pd[destination_col].unique())
    n_stations = len(unique_stations)
    station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
    
    # Get unique time windows
    trip_times = df_pd[timestamp_col].unique()
    n_times = len(trip_times)
    
    print(f"Processing {n_times:,} time points and {n_stations} stations")
    
    # Initialize prediction array
    predicted_arrivals = np.zeros((n_times, n_stations), dtype=np.float32)
    
    # For each trip start time, count predicted arrivals in the next delta_T window
    for i, current_time in enumerate(trip_times):
        if i % 1000 == 0:
            print(f"  Processing time point {i+1:,}/{n_times:,}")
        
        # Define the time window for counting arrivals
        window_start = current_time
        window_end = current_time + pd.Timedelta(seconds=delta_t)
        
        # Find all trips predicted to arrive in this time window
        predicted_arrivals_in_window = df_pd[
            (df_pd['predicted_arrival_time'] >= window_start) & 
            (df_pd['predicted_arrival_time'] < window_end)
        ]
        
        # Count predicted arrivals per station
        if len(predicted_arrivals_in_window) > 0:
            # Handle cases where predicted station might not be in our station list
            valid_predictions = predicted_arrivals_in_window[
                predicted_arrivals_in_window['dest_pred'].isin(unique_stations)
            ]
            
            if len(valid_predictions) > 0:
                station_counts = valid_predictions['dest_pred'].value_counts()
                for station, count in station_counts.items():
                    if station in station_to_idx:
                        predicted_arrivals[i, station_to_idx[station]] = count
    
    print(f"Created predicted arrivals with shape {predicted_arrivals.shape}")
    print(f"Average predicted arrivals per time window: {predicted_arrivals.mean():.2f}")
    print(f"Max predicted arrivals per station per window: {predicted_arrivals.max()}")
    
    return predicted_arrivals, unique_stations


def evaluate_station_arrival_prediction(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    station_list: List[int]
) -> Dict[str, float]:
    """
    Evaluate station arrival count predictions.
    
    This function computes various metrics to assess how well the model predicts
    station-level arrival counts compared to actual arrival counts.
    
    Args:
        y_true: True arrival counts array of shape (n_timestamps, n_stations)
        y_pred: Predicted arrival counts array of shape (n_timestamps, n_stations)
        station_list: List of station IDs corresponding to array columns
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating station arrival predictions for {len(station_list)} stations")
    
    # Flatten arrays for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {
            "overall_mse": np.nan,
            "overall_rmse": np.nan,
            "overall_mae": np.nan,
            "overall_r2": np.nan,
            "total_arrivals_true": np.nan,
            "total_arrivals_pred": np.nan,
            "avg_station_mse": np.nan,
            "avg_station_r2": np.nan,
            "station_correlation": np.nan
        }
    
    # Overall metrics
    overall_mse = mean_squared_error(y_true_clean, y_pred_clean)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    overall_r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Total arrival counts
    total_arrivals_true = np.sum(y_true)
    total_arrivals_pred = np.sum(y_pred)
    
    # Per-station metrics
    station_mses = []
    station_r2s = []
    
    for i in range(y_true.shape[1]):  # For each station
        station_true = y_true[:, i]
        station_pred = y_pred[:, i]
        
        # Skip stations with no arrivals
        if np.sum(station_true) > 0:
            station_mse = mean_squared_error(station_true, station_pred)
            station_r2 = r2_score(station_true, station_pred) if np.var(station_true) > 0 else 0
            
            station_mses.append(station_mse)
            station_r2s.append(station_r2)
    
    avg_station_mse = np.mean(station_mses) if station_mses else np.nan
    avg_station_r2 = np.mean(station_r2s) if station_r2s else np.nan
    
    # Station-wise correlation (how well we predict relative popularity)
    station_totals_true = np.sum(y_true, axis=0)
    station_totals_pred = np.sum(y_pred, axis=0)
    
    if np.std(station_totals_true) > 0 and np.std(station_totals_pred) > 0:
        station_correlation = np.corrcoef(station_totals_true, station_totals_pred)[0, 1]
    else:
        station_correlation = np.nan
    
    return {
        "overall_mse": overall_mse,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_r2": overall_r2,
        "total_arrivals_true": total_arrivals_true,
        "total_arrivals_pred": total_arrivals_pred,
        "avg_station_mse": avg_station_mse,
        "avg_station_r2": avg_station_r2,
        "station_correlation": station_correlation,
        "n_stations": len(station_list),
        "n_timepoints": y_true.shape[0]
    }


def print_station_arrival_evaluation(
    metrics: Dict[str, float], 
    delta_t: int = 1800,
    model_name: str = "Model"
):
    """
    Print formatted evaluation results for station arrival predictions.
    
    Args:
        metrics: Station arrival prediction metrics
        delta_t: Time window in seconds used for aggregation
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - STATION ARRIVAL PREDICTION RESULTS")
    print(f"Time Window: {delta_t} seconds ({delta_t/60:.1f} minutes)")
    print(f"{'='*70}")
    
    print(f"\n🏢 OVERALL ARRIVAL COUNT METRICS:")
    print(f"  MSE:              {metrics['overall_mse']:.2f}")
    print(f"  RMSE:             {metrics['overall_rmse']:.2f}")
    print(f"  MAE:              {metrics['overall_mae']:.2f}")
    print(f"  R²:               {metrics['overall_r2']:.4f}")
    
    print(f"\n📊 TOTAL ARRIVAL COMPARISON:")
    print(f"  True Arrivals:    {metrics['total_arrivals_true']:.0f}")
    print(f"  Predicted Arrivals: {metrics['total_arrivals_pred']:.0f}")
    arrival_error = abs(metrics['total_arrivals_true'] - metrics['total_arrivals_pred'])
    arrival_error_pct = (arrival_error / metrics['total_arrivals_true']) * 100 if metrics['total_arrivals_true'] > 0 else 0
    print(f"  Absolute Error:   {arrival_error:.0f} ({arrival_error_pct:.1f}%)")
    
    print(f"\n🎯 PER-STATION METRICS:")
    print(f"  Avg Station MSE:  {metrics['avg_station_mse']:.2f}")
    print(f"  Avg Station R²:   {metrics['avg_station_r2']:.4f}")
    print(f"  Station Correlation: {metrics['station_correlation']:.4f}")
    
    print(f"\n📈 DATASET INFO:")
    print(f"  Number of Stations: {metrics['n_stations']}")
    print(f"  Number of Timepoints: {metrics['n_timepoints']}")


def create_station_arrival_targets_parallel(
    df: pl.DataFrame, 
    delta_t: int = 1800,
    timestamp_col: str = "fecha_origen_recorrido",
    duration_col: str = "duracion_recorrido",
    destination_col: str = "id_estacion_destino",
    n_processes: int = None
) -> Tuple[np.ndarray, List[int]]:
    """
    OPTIMIZED: Create station-level arrival count targets using parallel processing.
    
    This function leverages multiprocessing to efficiently process large datasets
    by splitting the work across time chunks and using all available CPU cores.
    
    Args:
        df: DataFrame with trip data including timestamps, durations, and destinations
        delta_t: Time window in seconds (default: 1800 = 30 minutes)
        timestamp_col: Column name for trip start timestamp
        duration_col: Column name for trip duration in seconds
        destination_col: Column name for destination station ID
        n_processes: Number of processes (default: all available cores)
        
    Returns:
        Tuple of (arrival_targets, station_list) where:
        - arrival_targets: Array of shape (n_timestamps, n_stations) with arrival counts
        - station_list: List of station IDs corresponding to array columns
    """
    import multiprocessing as mp
    import time
    
    # Set global variables for worker functions
    global _df_pd_global, _station_to_idx_global, _n_stations_global, _destination_col_global, _delta_t_global
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"🚀 OPTIMIZED: Using {n_processes} processes for station arrival targets")
    print(f"Creating station arrival targets for delta_T = {delta_t} seconds ({delta_t/60:.1f} minutes)")
    
    start_time = time.time()
    
    # Convert to pandas for time operations and add arrival time
    df_pd = df.to_pandas()
    df_pd = df_pd.sort_values(timestamp_col)
    df_pd['arrival_time'] = df_pd[timestamp_col] + pd.to_timedelta(df_pd[duration_col], unit='s')
    
    # Set global variables
    _df_pd_global = df_pd
    _destination_col_global = destination_col
    _delta_t_global = delta_t
    
    # Get stations and time points
    unique_stations = sorted(df_pd[destination_col].unique())
    n_stations = len(unique_stations)
    station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
    
    _station_to_idx_global = station_to_idx
    _n_stations_global = n_stations
    
    trip_times = df_pd[timestamp_col].unique()
    n_times = len(trip_times)
    
    print(f"Processing {n_times:,} time points and {n_stations} stations")
    
    # Split time points into chunks for parallel processing
    chunk_size = max(100, n_times // (n_processes * 4))
    time_chunks = [trip_times[i:i + chunk_size] for i in range(0, n_times, chunk_size)]
    
    print(f"📦 Split into {len(time_chunks)} chunks of ~{chunk_size:,} time points each")
    
    # Process chunks in parallel
    print("🔥 Processing in parallel...")
    
    if len(time_chunks) == 1:
        # Single chunk - no multiprocessing overhead
        results = [_process_time_chunk_true_worker(time_chunks[0])]
    else:
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(_process_time_chunk_true_worker, time_chunks)
    
    # Combine results
    print("🔗 Combining results...")
    arrival_targets = np.zeros((n_times, n_stations), dtype=np.int32)
    time_to_idx = {time: idx for idx, time in enumerate(trip_times)}
    
    for chunk_result in results:
        for time_point, station_array in chunk_result:
            if time_point in time_to_idx:
                arrival_targets[time_to_idx[time_point]] = station_array
    
    elapsed_time = time.time() - start_time
    print(f"⚡ Parallel processing completed in {elapsed_time:.1f} seconds")
    print(f"📊 Average arrivals per time window: {arrival_targets.mean():.2f}")
    print(f"📊 Max arrivals per station per window: {arrival_targets.max()}")
    
    # Clean up global variables
    _df_pd_global = None
    _station_to_idx_global = None
    _n_stations_global = None
    _destination_col_global = None
    _delta_t_global = None
    
    return arrival_targets, unique_stations


def create_predicted_station_arrivals_parallel(
    df: pl.DataFrame,
    eta_predictions: np.ndarray,
    dest_predictions: np.ndarray,
    delta_t: int = 1800,
    timestamp_col: str = "fecha_origen_recorrido",
    destination_col: str = "id_estacion_destino",
    n_processes: int = None
) -> Tuple[np.ndarray, List[int]]:
    """
    OPTIMIZED: Create station-level arrival count predictions using parallel processing.
    
    This function uses parallel processing to efficiently compute predicted arrivals
    across all CPU cores, dramatically reducing computation time.
    
    Args:
        df: DataFrame with trip data including timestamps
        eta_predictions: Array of predicted trip durations in seconds
        dest_predictions: Array of predicted destination station IDs
        delta_t: Time window in seconds (default: 1800 = 30 minutes)
        timestamp_col: Column name for trip start timestamp
        destination_col: Column name for actual destination station ID (for reference)
        n_processes: Number of processes (default: all available cores)
        
    Returns:
        Tuple of (predicted_arrivals, station_list) where:
        - predicted_arrivals: Array of shape (n_timestamps, n_stations) with predicted counts
        - station_list: List of station IDs corresponding to array columns
    """
    import multiprocessing as mp
    import time
    
    # Set global variables for worker functions
    global _df_pd_global, _station_to_idx_global, _n_stations_global, _delta_t_global
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"🚀 OPTIMIZED: Using {n_processes} processes for predicted station arrivals")
    print(f"Creating predicted station arrivals for delta_T = {delta_t} seconds ({delta_t/60:.1f} minutes)")
    
    start_time = time.time()
    
    # Convert to pandas and add predictions
    df_pd = df.to_pandas()
    df_pd['eta_pred'] = eta_predictions
    df_pd['dest_pred'] = dest_predictions
    df_pd = df_pd.sort_values(timestamp_col)
    
    # Calculate predicted arrival time
    df_pd['predicted_arrival_time'] = df_pd[timestamp_col] + pd.to_timedelta(df_pd['eta_pred'], unit='s')
    
    # Set global variables
    _df_pd_global = df_pd
    _delta_t_global = delta_t
    
    # Get stations and times (use actual destinations for consistency)
    unique_stations = sorted(df_pd[destination_col].unique())
    n_stations = len(unique_stations)
    station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
    
    _station_to_idx_global = station_to_idx
    _n_stations_global = n_stations
    
    trip_times = df_pd[timestamp_col].unique()
    n_times = len(trip_times)
    
    print(f"Processing {n_times:,} time points and {n_stations} stations")
    
    # Split into chunks
    chunk_size = max(100, n_times // (n_processes * 4))
    time_chunks = [trip_times[i:i + chunk_size] for i in range(0, n_times, chunk_size)]
    
    print(f"📦 Split into {len(time_chunks)} chunks of ~{chunk_size:,} time points each")
    
    # Process in parallel
    print("🔥 Processing predictions in parallel...")
    
    if len(time_chunks) == 1:
        results = [_process_time_chunk_pred_worker(time_chunks[0])]
    else:
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(_process_time_chunk_pred_worker, time_chunks)
    
    # Combine results
    print("🔗 Combining prediction results...")
    predicted_arrivals = np.zeros((n_times, n_stations), dtype=np.float32)
    time_to_idx = {time: idx for idx, time in enumerate(trip_times)}
    
    for chunk_result in results:
        for time_point, station_array in chunk_result:
            if time_point in time_to_idx:
                predicted_arrivals[time_to_idx[time_point]] = station_array
    
    elapsed_time = time.time() - start_time
    print(f"⚡ Parallel processing completed in {elapsed_time:.1f} seconds")
    print(f"📊 Average predicted arrivals per time window: {predicted_arrivals.mean():.2f}")
    print(f"📊 Max predicted arrivals per station per window: {predicted_arrivals.max()}")
    
    # Clean up global variables
    _df_pd_global = None
    _station_to_idx_global = None
    _n_stations_global = None
    _delta_t_global = None
    
    return predicted_arrivals, unique_stations 