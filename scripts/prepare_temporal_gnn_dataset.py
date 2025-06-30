#!/usr/bin/env python3
"""
Prepare temporal clustered EcoBici dataset for Gated Graph Neural Network (GNN) training.

This module provides dataset classes for handling temporal sequences of graph data
with dynamic adjacency matrices, specifically designed for the Gated GCN model.

Key features:
- Vectorized temporal sequence creation using Polars
- Lazy evaluation for memory efficiency
- Parallel processing for maximum performance
- Dynamic graph loading for each time step
- Proper train/val/test isolation with temporal integrity

Author: EcoBici-AI
"""

import polars as pl
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import torch
import pickle
import gc
import time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from multiprocessing import get_context
import os
from numpy.lib.stride_tricks import sliding_window_view


def load_cluster_metadata(metadata_path: str) -> Dict:
    """Load cluster metadata including centroids and station counts."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def identify_safe_features(columns: List[str]) -> List[str]:
    """
    Identify features that are safe for temporal prediction (no data leakage).
    
    Safe features:
    - Lag features (explicitly historical)
    - Rolling statistics (computed from past data)
    - Temporal features (hour, day, etc.)
    - Weather features (current conditions known at prediction time)
    - Static metadata (cluster info)
    """
    safe_features = []
    
    # lag features (historical data - safe)
    lag_features = [
        col for col in columns 
        if any(pattern in col for pattern in ['_lag_', '_rolling_', '_same_hour_', '_yesterday', '_last_week'])
    ]
    safe_features.extend(lag_features)
    
    # temporal features (timing information - safe)
    temporal_features = [
        col for col in columns 
        if any(pattern in col for pattern in [
            'hour_', 'dow_', 'month_', 'is_weekend', 'is_morning_rush', 
            'is_evening_rush', 'is_daytime', 'is_summer', 'is_winter'
        ])
    ]
    safe_features.extend(temporal_features)
    
    # weather features (only basic current conditions - no demand-aggregated weather)
    weather_features = []
    for col in columns:
        if any(pattern in col for pattern in ['temp', 'humidity', 'wind_speed', 'precipitation']):
            if not any(demand_term in col for demand_term in ['count', 'trip', 'usage', 'demand', 'activity']):
                weather_features.append(col)
    safe_features.extend(weather_features)
    
    # cluster metadata (static information - safe)
    metadata_features = [
        col for col in columns 
        if any(pattern in col for pattern in [
            'cluster_centroid_', 'cluster_station_count'
        ])
    ]
    safe_features.extend(metadata_features)
    
    # remove duplicates
    safe_features = list(set(safe_features))
    
    return safe_features


def identify_target_features(columns: List[str]) -> List[str]:
    """Identify target features for prediction (current arrivals/departures)."""
    # primary targets: external counts (what we want to predict)
    primary_targets = [
        'arr_external_count',
        'dep_external_count'
    ]
    
    # only include targets that exist in the dataset
    available_targets = [t for t in primary_targets if t in columns]
    
    # if no external targets, look for any arrival counts
    if not available_targets:
        available_targets = [
            col for col in columns 
            if (col.startswith('arr_') and 'count' in col and 
                not any(pattern in col for pattern in ['lag_', 'rolling_', 'same_']))
        ]
    
    return available_targets


def find_safe_correlation_feature(feature_cols: List[str], target_col: str) -> str:
    """
    Find a safe historical feature to use for computing correlations.
    
    This prevents data leakage by using lag features instead of target variables.
    """
    correlation_candidates = []
    
    # look for lag versions of the target variable
    if 'arr_external_count' in target_col:
        lag_patterns = ['arr_external_count_lag_', 'arr_total_count_lag_', 'arr_count_lag_']
    elif 'dep_external_count' in target_col:
        lag_patterns = ['dep_external_count_lag_', 'dep_total_count_lag_', 'dep_count_lag_']
    else:
        lag_patterns = ['_lag_1', '_lag_2', '_lag_3']
    
    for pattern in lag_patterns:
        candidates = [col for col in feature_cols if pattern in col]
        correlation_candidates.extend(candidates)
    
    # look for rolling statistics of similar variables
    rolling_patterns = [
        'arr_external_count_rolling_', 'dep_external_count_rolling_',
        'arr_total_count_rolling_', 'dep_total_count_rolling_'
    ]
    
    for pattern in rolling_patterns:
        candidates = [col for col in feature_cols if pattern in col]
        correlation_candidates.extend(candidates)
    
    # look for any lag feature as fallback
    if not correlation_candidates:
        correlation_candidates = [col for col in feature_cols if '_lag_' in col]
    
    # look for any rolling feature as last resort
    if not correlation_candidates:
        correlation_candidates = [col for col in feature_cols if '_rolling_' in col]
    
    if correlation_candidates:
        return correlation_candidates[0]
    elif feature_cols:
        print(f"Warning: No historical correlation feature found, using {feature_cols[0]}")
        return feature_cols[0]
    else:
        raise ValueError("No safe correlation feature available")


def precompute_correlation_matrices(
    df_lazy: pl.LazyFrame,
    timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    correlation_col: str,
    correlation_window: int,
    min_correlation: float,
    cache_dir: Optional[Path] = None
) -> Dict[int, np.ndarray]:
    """
    Pre-compute all correlation matrices for each timestamp using vectorized operations.
    
    This is the biggest optimization - compute correlations once per timestamp 
    instead of once per sequence (saves ~25x computation).
    """
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"correlation_cache_{correlation_window}_{min_correlation}.pkl"
        
        if cache_file.exists():
            print(f"    Loading pre-computed correlations from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    print(f"    Pre-computing correlation matrices...")
    
    # Create pivot table with all data at once (vectorized)
    print(f"      Creating pivot table...")
    pivot_data = (
        df_lazy
        .select(["ts_start", "cluster_id", correlation_col])
        .filter(pl.col(correlation_col).is_not_null())
        .pivot(
            values=correlation_col,
            index="ts_start", 
            columns="cluster_id",
            aggregate_function="first"
        )
        .sort("ts_start")
        .collect()
    )
    
    # Convert to numpy matrix
    ts_column = pivot_data.select("ts_start").to_numpy().flatten()
    data_matrix = pivot_data.drop("ts_start").to_numpy()
    
    # Handle missing values
    data_matrix = np.nan_to_num(data_matrix, nan=0.0)
    
    print(f"      Computing sliding window correlations...")
    correlation_matrices = {}
    
    # Create timestamp to index mapping
    ts_to_idx = {ts: idx for idx, ts in enumerate(ts_column)}
    
    # Use sliding windows for efficient correlation computation
    if len(ts_column) >= correlation_window:
        # Create sliding windows
        windows = sliding_window_view(data_matrix, (correlation_window, data_matrix.shape[1]))
        windows = windows[:, 0, :, :]  # shape: (n_windows, correlation_window, n_clusters)
        
        # Vectorized correlation computation for all windows
        correlation_matrices_batch = compute_correlations_vectorized(
            windows, min_correlation
        )
        
        # Map back to timestamps
        for i, corr_matrix in enumerate(correlation_matrices_batch):
            if i + correlation_window < len(ts_column):
                ts = ts_column[i + correlation_window]
                correlation_matrices[ts] = corr_matrix
    
    # Cache results
    if cache_file:
        print(f"      Caching correlation matrices...")
        with open(cache_file, 'wb') as f:
            pickle.dump(correlation_matrices, f)
    
    print(f"    ✓ Pre-computed {len(correlation_matrices)} correlation matrices")
    return correlation_matrices


def compute_correlations_vectorized(windows: np.ndarray, min_correlation: float) -> List[np.ndarray]:
    """
    Compute correlations for multiple windows in parallel using vectorized operations.
    
    Args:
        windows: shape (n_windows, correlation_window, n_clusters)
        min_correlation: minimum correlation threshold
        
    Returns:
        List of edge index arrays for each window
    """
    batch_correlations = []
    
    for window_data in windows:
        # Check for columns with variance
        std_devs = np.std(window_data, axis=0)
        valid_cols = std_devs > 1e-10
        
        if np.sum(valid_cols) < 2:
            # No valid correlations
            batch_correlations.append(np.empty((2, 0), dtype=np.int64))
            continue
        
        # Filter valid columns
        filtered_data = window_data[:, valid_cols]
        valid_indices = np.where(valid_cols)[0]
        
        # Compute correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix_filtered = np.corrcoef(filtered_data.T)
        
        corr_matrix_filtered = np.nan_to_num(corr_matrix_filtered, nan=0.0)
        
        # Reconstruct full correlation matrix
        n_clusters = window_data.shape[1]
        corr_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                corr_matrix[idx_i, idx_j] = corr_matrix_filtered[i, j]
        
        # Create edges
        edges = []
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j and abs(corr_matrix[i, j]) >= min_correlation:
                    edges.append([i, j])
        
        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
        
        batch_correlations.append(edge_index)
    
    return batch_correlations


def create_vectorized_temporal_sequences(
    df: pl.DataFrame,
    sequence_length: int = 24,
    prediction_horizon: int = 1,
    correlation_window: int = 168,
    min_correlation: float = 0.1,
    graph_update_frequency: int = 24,  # Update graph every N steps (optimization)
    use_multiprocessing: bool = True,
    n_workers: Optional[int] = None,
    use_cache: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Create temporal sequences using heavily optimized vectorized operations.
    
    Optimizations applied:
    - Pre-computed correlation matrices (cached)
    - Vectorized Polars operations with pivot tables
    - Reduced graph update frequency
    - Multiprocessing for batch processing
    - Memory-efficient sliding windows
    """
    print(f"Creating vectorized temporal sequences...")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Prediction horizon: {prediction_horizon}")
    print(f"  Correlation window: {correlation_window}")
    
    start_time = time.time()
    step_times = {}
    
    # step 1: identify safe features and targets
    step_start = time.time()
    print(f"  Step 1/6: Identifying safe features...")
    
    # get columns from LazyFrame schema (avoiding performance warning)
    all_columns = list(df.collect_schema().keys())
    feature_cols = identify_safe_features(all_columns)
    target_cols = identify_target_features(all_columns)
    
    if not feature_cols:
        raise ValueError("No safe feature columns found")
    if not target_cols:
        raise ValueError("No target columns found")
    
    step_times['feature_identification'] = time.time() - step_start
    print(f"    ✓ Safe features: {len(feature_cols)} ({step_times['feature_identification']:.2f}s)")
    print(f"    ✓ Targets: {target_cols}")
    
    # validate no data leakage
    unsafe_features = [f for f in feature_cols if f in target_cols]
    if unsafe_features:
        raise ValueError(f"Data leakage detected: target variables in features: {unsafe_features}")
    
    # find safe correlation feature
    safe_correlation_col = find_safe_correlation_feature(feature_cols, target_cols[0])
    print(f"    ✓ Using safe correlation feature: {safe_correlation_col}")
    
    # additional safety check
    if safe_correlation_col in target_cols:
        raise ValueError(f"Data leakage: correlation feature {safe_correlation_col} is a target variable")
    
    # step 2: extract dataset info
    step_start = time.time()
    print(f"  Step 2/6: Extracting dataset information...")
    
    clusters_series = df.select(pl.col('cluster_id').unique().sort()).collect()
    timestamps_series = df.select(pl.col('ts_start').unique().sort()).collect()
    
    clusters = clusters_series['cluster_id'].to_list()
    timestamps = timestamps_series['ts_start'].to_list()
    num_clusters = len(clusters)
    num_features = len(feature_cols)
    
    # create cluster to index mapping
    cluster_to_idx = {cluster: idx for idx, cluster in enumerate(clusters)}
    
    step_times['dataset_info'] = time.time() - step_start
    print(f"    ✓ Clusters: {num_clusters} ({step_times['dataset_info']:.2f}s)")
    print(f"    ✓ Timestamps: {len(timestamps)}")
    print(f"    ✓ Features per cluster: {num_features}")
    
    # step 3: validate temporal requirements
    step_start = time.time()
    print(f"  Step 3/6: Validating temporal requirements...")
    
    min_steps_needed = correlation_window + sequence_length + prediction_horizon
    
    if len(timestamps) < min_steps_needed:
        raise ValueError(
            f"Insufficient temporal data: need {min_steps_needed} steps, "
            f"have {len(timestamps)}"
        )
    
    # convert to lazy dataframe for optimization
    if not hasattr(df, 'collect'):
        df_lazy = df.lazy()
    else:
        df_lazy = df
    
    step_times['validation'] = time.time() - step_start
    print(f"    ✓ Temporal validation passed ({step_times['validation']:.2f}s)")
    print(f"    ✓ Required steps: {min_steps_needed}, Available: {len(timestamps)}")
    
    # step 4: create sequence windows
    step_start = time.time()
    print(f"  Step 4/6: Creating sequence windows...")
    
    valid_start_idx = correlation_window + sequence_length
    total_sequences = len(timestamps) - prediction_horizon + 1 - valid_start_idx
    
    # create all sequence indices at once
    sequence_starts = list(range(valid_start_idx, len(timestamps) - prediction_horizon + 1))
    
    # create sequence metadata using vectorized operations
    seq_metadata_df = pl.DataFrame({
        'sequence_id': range(len(sequence_starts)),
        'start_idx': sequence_starts,
        'target_idx': [s + prediction_horizon - 1 for s in sequence_starts]
    })
    
    step_times['sequence_planning'] = time.time() - step_start
    print(f"    ✓ Sequence windows created ({step_times['sequence_planning']:.2f}s)")
    print(f"    ✓ Total sequences to create: {total_sequences}")
    print(f"    ✓ Valid start index: {valid_start_idx}")
    
    # step 5a: pre-compute correlation matrices (MAJOR OPTIMIZATION)
    step_start = time.time()
    print(f"  Step 5a/7: Pre-computing correlation matrices...")
    
    cache_dir = Path("data/temporal_gnn/cache") if use_cache else None
    correlation_cache = precompute_correlation_matrices(
        df_lazy, timestamps, clusters, cluster_to_idx,
        safe_correlation_col, correlation_window, min_correlation, cache_dir
    )
    
    step_times['correlation_precompute'] = time.time() - step_start
    print(f"    ✓ Correlation pre-computation completed ({step_times['correlation_precompute']:.1f}s)")
    
    # step 5b: extract all features using vectorized operations
    step_start = time.time()
    print(f"  Step 5b/7: Pre-extracting features with vectorized operations...")
    
    # Create feature pivot table for ultra-fast access
    feature_pivot = create_feature_pivot_table(df_lazy, feature_cols, clusters)
    target_pivot = create_target_pivot_table(df_lazy, target_cols, clusters)
    
    step_times['feature_extraction'] = time.time() - step_start
    print(f"    ✓ Feature extraction completed ({step_times['feature_extraction']:.1f}s)")
    
    # step 6: process sequences in batches with optimizations
    step_start = time.time()
    print(f"  Step 6/7: Processing sequences with optimizations...")
    
    # determine optimal batch size and worker count
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)  # limit to 8 cores max
    
    batch_size = min(2000, max(100, total_sequences // n_workers))  # larger batches for efficiency
    sequences = []
    
    # create progress bar
    total_batches = (len(sequence_starts) + batch_size - 1) // batch_size
    pbar = tqdm(
        total=len(sequence_starts),
        desc="Processing sequences",
        unit="seq",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    if use_multiprocessing and n_workers > 1:
        print(f"    Using multiprocessing with {n_workers} workers...")
        
        # prepare batches for multiprocessing
        batch_args = []
        for batch_start in range(0, len(sequence_starts), batch_size):
            batch_end = min(batch_start + batch_size, len(sequence_starts))
            batch_sequences = sequence_starts[batch_start:batch_end]
            
            batch_args.append((
                batch_sequences, timestamps, clusters, cluster_to_idx,
                feature_cols, target_cols, sequence_length, prediction_horizon,
                correlation_cache, graph_update_frequency, feature_pivot, target_pivot
            ))
        
        # process batches in parallel
        with get_context("spawn").Pool(processes=n_workers) as pool:
            batch_results = []
            
            for i, result in enumerate(pool.imap(process_sequence_batch_optimized, batch_args)):
                batch_results.append(result)
                pbar.update(len(batch_args[i][0]))  # update by batch size
                pbar.set_description(f"Processed batch {i+1}/{len(batch_args)}")
        
        # flatten results
        for batch_data in batch_results:
            sequences.extend(batch_data)
    
    else:
        # single-threaded processing (fallback)
        for batch_start in range(0, len(sequence_starts), batch_size):
            batch_end = min(batch_start + batch_size, len(sequence_starts))
            batch_sequences = sequence_starts[batch_start:batch_end]
            current_batch = batch_start // batch_size + 1
            
            pbar.set_description(f"Processing batch {current_batch}/{total_batches}")
            
            # process this batch with optimizations
            batch_data = process_sequence_batch_optimized((
                batch_sequences, timestamps, clusters, cluster_to_idx,
                feature_cols, target_cols, sequence_length, prediction_horizon,
                correlation_cache, graph_update_frequency, feature_pivot, target_pivot
            ))
            
            sequences.extend(batch_data)
            pbar.update(len(batch_sequences))
            
            # memory cleanup
            if batch_start > 0 and batch_start % (batch_size * 5) == 0:
                gc.collect()
    
    pbar.close()
    
    step_times['sequence_processing'] = time.time() - step_start
    print(f"    ✓ Sequence processing completed ({step_times['sequence_processing']:.1f}s)")
    print(f"    ✓ Processing rate: {len(sequences)/step_times['sequence_processing']:.1f} sequences/second")
    
    # step 7: finalize and create optimized metadata
    step_start = time.time()
    print(f"  Step 7/7: Creating optimized metadata...")
    
    # create enhanced metadata with optimization details
    metadata = {
        'num_sequences': len(sequences),
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'correlation_window': correlation_window,
        'min_correlation': min_correlation,
        'graph_update_frequency': graph_update_frequency,
        'num_clusters': num_clusters,
        'num_features': num_features,
        'feature_columns': feature_cols,
        'target_columns': target_cols,
        'cluster_to_idx': cluster_to_idx,
        'optimizations_applied': {
            'pre_computed_correlations': True,
            'reduced_graph_frequency': graph_update_frequency,
            'vectorized_pivot_tables': True,
            'multiprocessing': use_multiprocessing,
            'correlation_caching': use_cache,
            'workers': n_workers if use_multiprocessing else 1,
            'estimated_speedup': f"~{52417*13.59/sum(step_times.values()):.1f}x"
        },
        'performance_breakdown': step_times
    }
    
    step_times['metadata_creation'] = time.time() - step_start
    total_time = time.time() - start_time
    
    print(f"    ✓ Metadata created ({step_times['metadata_creation']:.2f}s)")
    print(f"\n  🎉 Total processing completed in {total_time:.1f}s")
    print(f"     - Feature identification: {step_times['feature_identification']:.2f}s")
    print(f"     - Dataset info extraction: {step_times['dataset_info']:.2f}s") 
    print(f"     - Temporal validation: {step_times['validation']:.2f}s")
    print(f"     - Sequence planning: {step_times['sequence_planning']:.2f}s")
    if 'correlation_precompute' in step_times:
        print(f"     - Correlation pre-computation: {step_times['correlation_precompute']:.1f}s")
    if 'feature_extraction' in step_times:
        print(f"     - Feature extraction: {step_times['feature_extraction']:.1f}s")
    print(f"     - Sequence processing: {step_times['sequence_processing']:.1f}s")
    print(f"     - Metadata creation: {step_times['metadata_creation']:.2f}s")
    print(f"  📊 Final result: {len(sequences)} sequences ({len(sequences)/total_time:.1f} seq/s average)")
    print(f"  🚀 Estimated speedup: ~{52417*13.59/total_time:.1f}x faster than original")
    
    return sequences, metadata


def create_feature_pivot_table(df_lazy: pl.LazyFrame, feature_cols: List[str], clusters: List) -> Dict:
    """Create pivot tables for ultra-fast feature access."""
    print(f"      Creating feature pivot table...")
    
    # Create a more efficient feature extraction using pivot operations
    # This reduces the number of queries from thousands to just a few
    feature_pivot = {}
    
    # Process features in chunks to manage memory
    chunk_size = 50  # process 50 features at a time
    for i in range(0, len(feature_cols), chunk_size):
        chunk_cols = feature_cols[i:i+chunk_size]
        
        for col in chunk_cols:
            pivot_data = (
                df_lazy
                .select(["ts_start", "cluster_id", col])
                .filter(pl.col(col).is_not_null())
                .pivot(
                    values=col,
                    index="ts_start",
                    columns="cluster_id", 
                    aggregate_function="first"
                )
                .sort("ts_start")
                .collect()
            )
            
            if pivot_data.height > 0:
                ts_column = pivot_data.select("ts_start").to_numpy().flatten()
                data_matrix = pivot_data.drop("ts_start").to_numpy()
                data_matrix = np.nan_to_num(data_matrix, nan=0.0)
                
                feature_pivot[col] = {
                    'timestamps': ts_column,
                    'data': data_matrix
                }
    
    return feature_pivot


def create_target_pivot_table(df_lazy: pl.LazyFrame, target_cols: List[str], clusters: List) -> Dict:
    """Create pivot tables for ultra-fast target access."""
    print(f"      Creating target pivot table...")
    
    target_pivot = {}
    
    for col in target_cols:
        pivot_data = (
            df_lazy
            .select(["ts_start", "cluster_id", col])
            .filter(pl.col(col).is_not_null())
            .pivot(
                values=col,
                index="ts_start",
                columns="cluster_id",
                aggregate_function="first"
            )
            .sort("ts_start")
            .collect()
        )
        
        if pivot_data.height > 0:
            ts_column = pivot_data.select("ts_start").to_numpy().flatten()
            data_matrix = pivot_data.drop("ts_start").to_numpy()
            data_matrix = np.nan_to_num(data_matrix, nan=0.0)
            
            target_pivot[col] = {
                'timestamps': ts_column,
                'data': data_matrix
            }
    
    return target_pivot


def process_sequence_batch_optimized(args: Tuple) -> List[Dict]:
    """
    Optimized batch processing function designed for multiprocessing.
    
    Major optimizations:
    - Pre-computed correlations (no real-time correlation calculation)
    - Reduced graph update frequency
    - Vectorized feature/target extraction from pivot tables
    - Minimal data copying
    """
    (batch_sequences, timestamps, clusters, cluster_to_idx,
     feature_cols, target_cols, sequence_length, prediction_horizon,
     correlation_cache, graph_update_frequency, feature_pivot, target_pivot) = args
    
    batch_data = []
    
    for seq_start_idx in batch_sequences:
        # define time windows
        target_ts = timestamps[seq_start_idx + prediction_horizon - 1]
        seq_timestamps = timestamps[seq_start_idx - sequence_length:seq_start_idx]
        
        # extract features using pre-computed pivot tables (MUCH faster)
        x_seq = extract_features_from_pivot(
            seq_timestamps, clusters, feature_cols, feature_pivot
        )
        
        # extract targets using pre-computed pivot tables
        y_target = extract_targets_from_pivot(
            target_ts, clusters, target_cols, target_pivot
        )
        
        # create graphs with reduced frequency (major optimization)
        edge_seq = create_optimized_correlation_graphs(
            seq_timestamps, correlation_cache, graph_update_frequency
        )
        
        # create sequence dictionary
        sequence = {
            'x_seq': x_seq,
            'edge_seq': edge_seq,
            'y': y_target,
            'timestamps': seq_timestamps + [target_ts],
            'target_timestamp': target_ts
        }
        
        batch_data.append(sequence)
    
    return batch_data


def extract_features_from_pivot(
    seq_timestamps: List, clusters: List, feature_cols: List[str], feature_pivot: Dict
) -> np.ndarray:
    """Extract features using pre-computed pivot tables - ultra fast."""
    seq_len = len(seq_timestamps)
    num_clusters = len(clusters)
    num_features = len(feature_cols)
    
    x_seq = np.zeros((seq_len, num_clusters, num_features))
    
    for feat_idx, col in enumerate(feature_cols):
        if col in feature_pivot:
            pivot_data = feature_pivot[col]
            ts_array = pivot_data['timestamps']
            data_matrix = pivot_data['data']
            
            # Find indices for our timestamps
            for t_idx, ts in enumerate(seq_timestamps):
                ts_idx = np.searchsorted(ts_array, ts)
                if ts_idx < len(ts_array) and ts_array[ts_idx] == ts:
                    x_seq[t_idx, :, feat_idx] = data_matrix[ts_idx, :]
    
    return x_seq


def extract_targets_from_pivot(
    target_ts, clusters: List, target_cols: List[str], target_pivot: Dict
) -> np.ndarray:
    """Extract targets using pre-computed pivot tables - ultra fast."""
    num_clusters = len(clusters)
    num_targets = len(target_cols)
    
    y_target = np.zeros((num_clusters, num_targets))
    
    for targ_idx, col in enumerate(target_cols):
        if col in target_pivot:
            pivot_data = target_pivot[col]
            ts_array = pivot_data['timestamps']
            data_matrix = pivot_data['data']
            
            # Find index for target timestamp
            ts_idx = np.searchsorted(ts_array, target_ts)
            if ts_idx < len(ts_array) and ts_array[ts_idx] == target_ts:
                y_target[:, targ_idx] = data_matrix[ts_idx, :]
    
    # squeeze if single target
    if num_targets == 1:
        y_target = y_target.squeeze(-1)
    
    return y_target


def create_optimized_correlation_graphs(
    seq_timestamps: List, correlation_cache: Dict, graph_update_frequency: int
) -> List[np.ndarray]:
    """
    Create correlation graphs with optimized frequency.
    
    Major optimization: only update graph every N steps instead of every step.
    """
    edge_seq = []
    
    # Use reduced frequency for graph updates
    for i, ts in enumerate(seq_timestamps):
        # Only update graph at specified intervals or at the end
        if i % graph_update_frequency == 0 or i == len(seq_timestamps) - 1:
            # Use pre-computed correlation
            if ts in correlation_cache:
                current_edges = correlation_cache[ts]
            else:
                current_edges = np.empty((2, 0), dtype=np.int64)
        else:
            # Reuse previous graph (saves computation)
            if edge_seq:
                current_edges = edge_seq[-1]
            else:
                current_edges = np.empty((2, 0), dtype=np.int64)
        
        edge_seq.append(current_edges)
    
    return edge_seq


def process_sequence_batch(
    df_lazy: pl.LazyFrame,
    batch_sequences: List[int],
    timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    feature_cols: List[str],
    target_cols: List[str],
    safe_correlation_col: str,
    sequence_length: int,
    prediction_horizon: int,
    correlation_window: int,
    min_correlation: float
) -> List[Dict]:
    """Process a batch of sequences using vectorized operations."""
    
    batch_data = []
    
    # create progress bar for individual sequences within batch (only for large batches)
    if len(batch_sequences) > 100:
        seq_iterator = tqdm(
            batch_sequences, 
            desc="Processing batch sequences", 
            leave=False,
            unit="seq"
        )
    else:
        seq_iterator = batch_sequences
    
    for seq_start_idx in seq_iterator:
        # define time windows
        target_ts = timestamps[seq_start_idx + prediction_horizon - 1]
        seq_timestamps = timestamps[seq_start_idx - sequence_length:seq_start_idx]
        
        # extract features and targets using vectorized polars operations
        x_seq = extract_vectorized_features(
            df_lazy, seq_timestamps, clusters, cluster_to_idx, feature_cols
        )
        
        y_target = extract_vectorized_targets(
            df_lazy, target_ts, clusters, cluster_to_idx, target_cols
        )
        
        # create dynamic graphs using vectorized correlation computation
        edge_seq = create_vectorized_correlation_graphs(
            df_lazy, seq_timestamps, timestamps, clusters, cluster_to_idx,
            safe_correlation_col, correlation_window, min_correlation
        )
        
        # create sequence dictionary
        sequence = {
            'x_seq': x_seq,
            'edge_seq': edge_seq,
            'y': y_target,
            'timestamps': seq_timestamps + [target_ts],
            'target_timestamp': target_ts
        }
        
        batch_data.append(sequence)
    
    return batch_data


def extract_vectorized_features(
    df_lazy: pl.LazyFrame,
    seq_timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    feature_cols: List[str]
) -> np.ndarray:
    """Extract features using vectorized Polars operations."""
    
    # create timestamp filter for this sequence
    ts_filter = pl.col('ts_start').is_in(seq_timestamps)
    
    # extract all data for these timestamps at once
    seq_data = (
        df_lazy
        .filter(ts_filter)
        .select(['cluster_id', 'ts_start'] + feature_cols)
        .collect()
    )
    
    # initialize feature matrix
    seq_len = len(seq_timestamps)
    num_clusters = len(clusters)
    num_features = len(feature_cols)
    x_seq = np.zeros((seq_len, num_clusters, num_features))
    
    # fill feature matrix using vectorized operations
    if seq_data.height > 0:
        for t_idx, ts in enumerate(seq_timestamps):
            ts_rows = seq_data.filter(pl.col('ts_start') == ts)
            
            if ts_rows.height > 0:
                # get cluster data as numpy arrays
                cluster_ids = ts_rows['cluster_id'].to_numpy()
                feature_arrays = ts_rows.select(feature_cols).to_numpy()
                
                # map cluster ids to indices and fill matrix
                for i, cluster_id in enumerate(cluster_ids):
                    if cluster_id in cluster_to_idx:
                        cluster_idx = cluster_to_idx[cluster_id]
                        x_seq[t_idx, cluster_idx, :] = feature_arrays[i]
    
    # handle NaN values
    x_seq = np.nan_to_num(x_seq, nan=0.0)
    
    return x_seq


def extract_vectorized_targets(
    df_lazy: pl.LazyFrame,
    target_ts,
    clusters: List,
    cluster_to_idx: Dict,
    target_cols: List[str]
) -> np.ndarray:
    """Extract targets using vectorized Polars operations."""
    
    # extract target data for this timestamp
    target_data = (
        df_lazy
        .filter(pl.col('ts_start') == target_ts)
        .select(['cluster_id'] + target_cols)
        .collect()
    )
    
    # initialize target array
    num_clusters = len(clusters)
    num_targets = len(target_cols)
    y_target = np.zeros((num_clusters, num_targets))
    
    if target_data.height > 0:
        # get data as numpy arrays
        cluster_ids = target_data['cluster_id'].to_numpy()
        target_arrays = target_data.select(target_cols).to_numpy()
        
        # fill target matrix
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in cluster_to_idx:
                cluster_idx = cluster_to_idx[cluster_id]
                y_target[cluster_idx, :] = target_arrays[i]
    
    # handle NaN values
    y_target = np.nan_to_num(y_target, nan=0.0)
    
    # squeeze if single target
    if num_targets == 1:
        y_target = y_target.squeeze(-1)
    
    return y_target


def create_vectorized_correlation_graphs(
    df_lazy: pl.LazyFrame,
    seq_timestamps: List,
    all_timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    correlation_col: str,
    correlation_window: int,
    min_correlation: float
) -> List[np.ndarray]:
    """Create correlation graphs using vectorized operations."""
    
    edge_seq = []
    
    for ts in seq_timestamps:
        try:
            current_idx = all_timestamps.index(ts)
        except ValueError:
            edge_seq.append(np.empty((2, 0), dtype=np.int64))
            continue
        
        if current_idx < correlation_window:
            edge_seq.append(np.empty((2, 0), dtype=np.int64))
            continue
        
        # get historical window
        hist_timestamps = all_timestamps[current_idx - correlation_window:current_idx]
        
        # extract correlation data using vectorized operations
        hist_filter = pl.col('ts_start').is_in(hist_timestamps)
        
        corr_data = (
            df_lazy
            .filter(hist_filter)
            .select(['cluster_id', 'ts_start', correlation_col])
            .collect()
        )
        
        if corr_data.height > 0:
            # create pivot table for correlation computation
            correlation_matrix = create_correlation_matrix(
                corr_data, hist_timestamps, clusters, cluster_to_idx, correlation_col
            )
            
            # compute edges from correlation matrix
            edge_index = compute_correlation_edges(
                correlation_matrix, clusters, min_correlation
            )
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
        
        edge_seq.append(edge_index)
    
    return edge_seq


def create_correlation_matrix(
    corr_data: pl.DataFrame,
    hist_timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    correlation_col: str
) -> np.ndarray:
    """Create correlation matrix using vectorized operations."""
    
    num_timestamps = len(hist_timestamps)
    num_clusters = len(clusters)
    correlation_data = np.zeros((num_timestamps, num_clusters))
    
    if corr_data.height > 0:
        # group by timestamp and cluster for efficient processing
        for t_idx, ts in enumerate(hist_timestamps):
            ts_data = corr_data.filter(pl.col('ts_start') == ts)
            
            if ts_data.height > 0:
                cluster_ids = ts_data['cluster_id'].to_numpy()
                values = ts_data[correlation_col].to_numpy()
                
                # handle NaN and inf values in the input data
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                
                for i, cluster_id in enumerate(cluster_ids):
                    if cluster_id in cluster_to_idx:
                        cluster_idx = cluster_to_idx[cluster_id]
                        correlation_data[t_idx, cluster_idx] = values[i]
    
    return correlation_data


def compute_correlation_edges(
    correlation_data: np.ndarray,
    clusters: List,
    min_correlation: float
) -> np.ndarray:
    """Compute edges from correlation matrix with proper handling of constant values."""
    
    num_clusters = len(clusters)
    
    # check if we have enough data points for correlation
    if correlation_data.shape[0] < 2:
        return np.empty((2, 0), dtype=np.int64)
    
    try:
        # check for columns with zero variance (constant values)
        std_devs = np.std(correlation_data, axis=0)
        valid_cols = std_devs > 1e-10  # threshold for numerical stability
        
        if np.sum(valid_cols) < 2:
            # not enough columns with variance, return empty graph
            return np.empty((2, 0), dtype=np.int64)
        
        # filter out constant columns before correlation computation
        filtered_data = correlation_data[:, valid_cols]
        valid_indices = np.where(valid_cols)[0]
        
        # suppress warnings for cleaner output
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix_filtered = np.corrcoef(filtered_data.T)
        
        # handle NaN and inf values
        corr_matrix_filtered = np.nan_to_num(corr_matrix_filtered, nan=0.0, posinf=0.0, neginf=0.0)
        
        # reconstruct full correlation matrix
        corr_matrix = np.zeros((num_clusters, num_clusters))
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                corr_matrix[idx_i, idx_j] = corr_matrix_filtered[i, j]
        
        # set diagonal to 1 for valid columns
        for idx in valid_indices:
            corr_matrix[idx, idx] = 1.0
            
    except Exception as e:
        # fallback to empty graph if correlation computation fails
        return np.empty((2, 0), dtype=np.int64)
    
    # create edges based on correlation threshold
    edges = []
    
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j and abs(corr_matrix[i, j]) >= min_correlation:
                edges.append([i, j])
    
    # convert to edge_index format
    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    
    return edge_index


def save_temporal_sequences(
    sequences: List[Dict],
    metadata: Dict,
    output_path: Path,
    compress: bool = True
):
    """Save temporal sequences to disk."""
    print(f"Saving {len(sequences)} sequences to {output_path}")
    
    # prepare data for saving
    data_to_save = {
        'sequences': sequences,
        'metadata': metadata
    }
    
    # save with compression
    if compress:
        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        torch.save(data_to_save, output_path)
    
    # print size info
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Prepare temporal GNN dataset from clustered EcoBici data')
    parser.add_argument('--input_dir', type=str, default='data/clustered',
                        help='Directory containing clustered parquet files')
    parser.add_argument('--output_dir', type=str, default='data/temporal_gnn',
                        help='Output directory for temporal GNN data')
    parser.add_argument('--sequence_length', type=int, default=24,
                        help='Length of input sequences in time steps (default: 24)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                        help='Steps ahead to predict (default: 1)')
    parser.add_argument('--correlation_window', type=int, default=168,
                        help='Window size for computing correlations (default: 168 = 1 week)')
    parser.add_argument('--min_correlation', type=float, default=0.1,
                        help='Minimum correlation threshold for graph edges (default: 0.1)')
    parser.add_argument('--graph_update_frequency', type=int, default=1,
                        help='Update graph every N steps instead of every step (optimization, default: 1)')
    parser.add_argument('--disable_multiprocessing', action='store_true',
                        help='Disable multiprocessing (use single-threaded processing)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--disable_cache', action='store_true',
                        help='Disable correlation caching')
    
    args = parser.parse_args()
    
    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load metadata
    metadata_path = Path(args.input_dir) / 'dataset_metadata.json'
    metadata = load_cluster_metadata(str(metadata_path))
    
    print(f"Loaded metadata for {metadata.get('n_clusters', 'unknown')} clusters")
    print(f"Data splits: Train={metadata['train_shape']}, Val={metadata['val_shape']}, Test={metadata['test_shape']}")
    
    # process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split for temporal sequences...")
        print(f"{'='*60}")
        
        # determine input file path
        clustering_enabled = metadata.get('clustering_enabled', metadata.get('n_clusters') is not None)
        if clustering_enabled:
            input_file = Path(args.input_dir) / f'{split}_cluster_features.parquet'
        else:
            input_file = Path(args.input_dir) / f'{split}_station_features.parquet'
        
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping {split} split")
            continue
        
        print(f"Loading data from {input_file}...")
        
        # load split data using lazy evaluation for efficiency
        df = pl.scan_parquet(str(input_file))
        
        # get basic info about the data
        info_df = df.select([pl.col('cluster_id').n_unique().alias('clusters'),
                           pl.col('ts_start').n_unique().alias('timestamps'),
                           pl.count().alias('total_rows')]).collect()
        
        print(f"  Dataset info: {info_df['total_rows'][0]} records, {info_df['clusters'][0]} clusters, {info_df['timestamps'][0]} timestamps")
        
        # create temporal sequences using optimized vectorized approach
        try:
            sequences, seq_metadata = create_vectorized_temporal_sequences(
                df=df,
                sequence_length=args.sequence_length,
                prediction_horizon=args.prediction_horizon,
                correlation_window=args.correlation_window,
                min_correlation=args.min_correlation,
                graph_update_frequency=args.graph_update_frequency,
                use_multiprocessing=not args.disable_multiprocessing,
                n_workers=args.workers,
                use_cache=not args.disable_cache
            )
            
            # save sequences
            print(f"  Saving sequences...")
            save_start = time.time()
            output_file = output_dir / f'{split}_temporal_sequences.pkl'
            save_temporal_sequences(sequences, seq_metadata, output_file)
            save_time = time.time() - save_start
            
            print(f"  Successfully created {len(sequences)} sequences for {split} (saved in {save_time:.1f}s)")
            
        except Exception as e:
            print(f"  Error processing {split}: {str(e)}")
            continue
        
        # cleanup memory
        del sequences
        gc.collect()
    
    # save processing configuration with optimization details
    config = {
        'sequence_length': args.sequence_length,
        'prediction_horizon': args.prediction_horizon,
        'correlation_window': args.correlation_window,
        'min_correlation': args.min_correlation,
        'graph_update_frequency': args.graph_update_frequency,
        'optimization_settings': {
            'multiprocessing_enabled': not args.disable_multiprocessing,
            'workers': args.workers,
            'caching_enabled': not args.disable_cache,
            'graph_update_frequency': args.graph_update_frequency
        },
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'original_metadata': metadata
    }
    
    with open(output_dir / 'temporal_processing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TEMPORAL DATASET PREPARATION COMPLETED!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence parameters:")
    print(f"  - Length: {args.sequence_length} steps")
    print(f"  - Prediction horizon: {args.prediction_horizon} steps")
    print(f"  - Correlation window: {args.correlation_window} steps")
    print(f"  - Min correlation threshold: {args.min_correlation}")
    print(f"Optimization settings:")
    print(f"  - Graph update frequency: every {args.graph_update_frequency} step(s)")
    print(f"  - Multiprocessing: {'disabled' if args.disable_multiprocessing else 'enabled'}")
    print(f"  - Workers: {args.workers if args.workers else 'auto-detect'}")
    print(f"  - Correlation caching: {'disabled' if args.disable_cache else 'enabled'}")
    print(f"\nTo train the Gated GCN model, run:")
    print(f"python scripts/train_temporal_gnn.py --data_dir {output_dir}")


if __name__ == '__main__':
    main() 