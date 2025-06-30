#!/usr/bin/env python3
"""
Prepare temporal clustered EcoBici dataset for Gated Graph Neural Network (GNN) training.

This script processes the clustered dataset to:
1. Create temporal sequences using only historical data (no data leakage)
2. Generate dynamic graphs based on historical correlations  
3. Structure data for temporal GNN with spatial-temporal relationships
4. Save sequences in format compatible with Gated GCN model

Focus: Predict external movements using temporal sequences of historical information.
Uses lag features, rolling statistics, and historical patterns exclusively.
Prevents data leakage by strict temporal ordering and historical-only correlations.

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
from sklearn.preprocessing import StandardScaler


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
    # be conservative to avoid potential leakage from weather features aggregated by demand
    weather_features = []
    for col in columns:
        # only include basic weather that's not derived from demand patterns
        if any(pattern in col for pattern in ['temp', 'humidity', 'wind_speed', 'precipitation']):
            # exclude if it might be aggregated by demand (contains demand-related terms)
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
    # priority order for correlation features (all historical/safe)
    correlation_candidates = []
    
    # 1. look for lag versions of the target variable
    if 'arr_external_count' in target_col:
        lag_patterns = ['arr_external_count_lag_', 'arr_total_count_lag_', 'arr_count_lag_']
    elif 'dep_external_count' in target_col:
        lag_patterns = ['dep_external_count_lag_', 'dep_total_count_lag_', 'dep_count_lag_']
    else:
        lag_patterns = ['_lag_1', '_lag_2', '_lag_3']
    
    for pattern in lag_patterns:
        candidates = [col for col in feature_cols if pattern in col]
        correlation_candidates.extend(candidates)
    
    # 2. look for rolling statistics of similar variables
    rolling_patterns = [
        'arr_external_count_rolling_', 'dep_external_count_rolling_',
        'arr_total_count_rolling_', 'dep_total_count_rolling_'
    ]
    
    for pattern in rolling_patterns:
        candidates = [col for col in feature_cols if pattern in col]
        correlation_candidates.extend(candidates)
    
    # 3. look for any lag feature as fallback
    if not correlation_candidates:
        correlation_candidates = [col for col in feature_cols if '_lag_' in col]
    
    # 4. look for any rolling feature as last resort
    if not correlation_candidates:
        correlation_candidates = [col for col in feature_cols if '_rolling_' in col]
    
    # return the first available candidate or fallback to first feature
    if correlation_candidates:
        return correlation_candidates[0]
    elif feature_cols:
        print(f"Warning: No historical correlation feature found, using {feature_cols[0]}")
        return feature_cols[0]
    else:
        raise ValueError("No safe correlation feature available")


def create_temporal_sequences(
    df: pl.DataFrame,
    sequence_length: int = 24,
    prediction_horizon: int = 1,
    correlation_window: int = 168  # 1 week for stable correlations
) -> Tuple[List[Dict], Dict]:
    """
    Create temporal sequences without data leakage.
    
    Each sequence contains:
    - Historical features for sequence_length time steps
    - Dynamic graphs computed from historical correlations  
    - Future target values for prediction
    
    Args:
        df: DataFrame with temporal cluster data
        sequence_length: Number of historical time steps to use
        prediction_horizon: Steps ahead to predict (typically 1)
        correlation_window: Window size for computing correlations
        
    Returns:
        Tuple of (sequences_list, metadata_dict)
    """
    print(f"Creating temporal sequences...")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Prediction horizon: {prediction_horizon}")
    print(f"  Correlation window: {correlation_window}")
    
    # identify safe features and targets
    feature_cols = identify_safe_features(df.columns)
    target_cols = identify_target_features(df.columns)
    
    if not feature_cols:
        raise ValueError("No safe feature columns found")
    if not target_cols:
        raise ValueError("No target columns found")
    
    print(f"  Safe features: {len(feature_cols)}")
    print(f"  Targets: {target_cols}")
    
    # validate that no target variables are in safe features (data leakage check)
    unsafe_features = [f for f in feature_cols if f in target_cols]
    if unsafe_features:
        raise ValueError(f"Data leakage detected: target variables in features: {unsafe_features}")
    
    # find safe correlation feature
    safe_correlation_col = find_safe_correlation_feature(feature_cols, target_cols[0])
    print(f"  Using safe correlation feature: {safe_correlation_col}")
    
    # additional safety check
    if safe_correlation_col in target_cols:
        raise ValueError(f"Data leakage: correlation feature {safe_correlation_col} is a target variable")
    
    # get unique clusters and timestamps
    clusters = sorted(df['cluster_id'].unique())
    timestamps = sorted(df['ts_start'].unique())
    
    num_clusters = len(clusters)
    num_features = len(feature_cols)
    
    print(f"  Clusters: {num_clusters}")
    print(f"  Timestamps: {len(timestamps)}")
    
    # create cluster to index mapping
    cluster_to_idx = {cluster: idx for idx, cluster in enumerate(clusters)}
    
    # minimum time steps needed for valid sequences
    min_steps_needed = correlation_window + sequence_length + prediction_horizon
    
    if len(timestamps) < min_steps_needed:
        raise ValueError(
            f"Insufficient temporal data: need {min_steps_needed} steps, "
            f"have {len(timestamps)}"
        )
    
    sequences = []
    
    # create sequences starting after sufficient historical data
    valid_start_idx = correlation_window + sequence_length
    
    print(f"  Creating sequences from index {valid_start_idx} to {len(timestamps) - prediction_horizon}")
    
    for t_idx in range(valid_start_idx, len(timestamps) - prediction_horizon + 1):
        # define time windows (strict historical ordering)
        target_ts = timestamps[t_idx + prediction_horizon - 1]
        seq_timestamps = timestamps[t_idx - sequence_length:t_idx]
        
        # extract sequence features (only historical data)
        x_seq = extract_sequence_features(
            df, seq_timestamps, clusters, cluster_to_idx, feature_cols
        )
        
        # extract target values (future data)
        y_target = extract_target_values(
            df, target_ts, clusters, cluster_to_idx, target_cols
        )
        
        # create dynamic graphs using historical lag features (safe for correlation)
        # use historical lag features instead of target variables to avoid data leakage
        edge_seq = create_dynamic_correlation_graphs(
            df, seq_timestamps, clusters, cluster_to_idx, safe_correlation_col, correlation_window
        )
        
        # create sequence dictionary
        sequence = {
            'x_seq': x_seq,
            'edge_seq': edge_seq,
            'y': y_target,
            'timestamps': seq_timestamps + [target_ts],
            'target_timestamp': target_ts
        }
        
        sequences.append(sequence)
        
        if len(sequences) % 100 == 0:
            print(f"    Created {len(sequences)} sequences...")
    
    print(f"  Created {len(sequences)} total sequences")
    
    # create metadata
    metadata = {
        'num_sequences': len(sequences),
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'correlation_window': correlation_window,
        'num_clusters': num_clusters,
        'num_features': num_features,
        'feature_columns': feature_cols,
        'target_columns': target_cols,
        'cluster_to_idx': cluster_to_idx
    }
    
    return sequences, metadata


def extract_sequence_features(
    df: pl.DataFrame, 
    timestamps: List, 
    clusters: List, 
    cluster_to_idx: Dict, 
    feature_cols: List
) -> np.ndarray:
    """Extract feature matrix for a temporal sequence."""
    seq_len = len(timestamps)
    num_clusters = len(clusters)
    num_features = len(feature_cols)
    
    # initialize feature matrix [seq_len, num_clusters, num_features]
    x_seq = np.zeros((seq_len, num_clusters, num_features))
    
    for t_idx, ts in enumerate(timestamps):
        # get data for this timestamp
        ts_data = df.filter(pl.col('ts_start') == ts)
        
        if ts_data.height == 0:
            continue  # keep zeros for missing data
        
        # extract features for each cluster
        for cluster_id in clusters:
            cluster_idx = cluster_to_idx[cluster_id]
            cluster_data = ts_data.filter(pl.col('cluster_id') == cluster_id)
            
            if cluster_data.height > 0:
                # extract feature values
                try:
                    feature_values = cluster_data.select(feature_cols).to_numpy()[0]
                    x_seq[t_idx, cluster_idx, :] = feature_values
                except:
                    # handle missing features gracefully
                    continue
    
    # handle NaN values
    x_seq = np.nan_to_num(x_seq, nan=0.0)
    
    return x_seq


def extract_target_values(
    df: pl.DataFrame, 
    target_ts, 
    clusters: List, 
    cluster_to_idx: Dict, 
    target_cols: List
) -> np.ndarray:
    """Extract target values for prediction."""
    num_clusters = len(clusters)
    num_targets = len(target_cols)
    
    # initialize target array [num_clusters, num_targets]
    y_target = np.zeros((num_clusters, num_targets))
    
    # get data for target timestamp
    target_data = df.filter(pl.col('ts_start') == target_ts)
    
    if target_data.height > 0:
        for cluster_id in clusters:
            cluster_idx = cluster_to_idx[cluster_id]
            cluster_data = target_data.filter(pl.col('cluster_id') == cluster_id)
            
            if cluster_data.height > 0:
                try:
                    # extract target values
                    target_values = cluster_data.select(target_cols).to_numpy()[0]
                    y_target[cluster_idx, :] = target_values
                except:
                    # handle missing targets gracefully
                    continue
    
    # handle NaN values
    y_target = np.nan_to_num(y_target, nan=0.0)
    
    # squeeze if single target
    if num_targets == 1:
        y_target = y_target.squeeze(-1)
    
    return y_target


def create_dynamic_correlation_graphs(
    df: pl.DataFrame,
    seq_timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    correlation_col: str,
    correlation_window: int,
    min_correlation: float = 0.1
) -> List[np.ndarray]:
    """
    Create dynamic graphs based on historical correlations using safe features.
    
    For each timestamp in the sequence, compute correlations using only
    historical data (correlation_window steps before that timestamp).
    Uses historical lag features to prevent data leakage.
    
    Args:
        correlation_col: Safe historical feature (lag/rolling) for correlation computation
    """
    edge_seq = []
    
    # get all available timestamps for correlation computation
    all_timestamps = sorted(df['ts_start'].unique())
    
    for ts in seq_timestamps:
        # find index of current timestamp
        try:
            current_idx = all_timestamps.index(ts)
        except ValueError:
            # timestamp not in data, use empty graph
            edge_seq.append(np.empty((2, 0), dtype=np.int64))
            continue
        
        # check if we have enough historical data
        if current_idx < correlation_window:
            # insufficient historical data, use empty graph
            edge_seq.append(np.empty((2, 0), dtype=np.int64))
            continue
        
        # get historical window (strictly before current timestamp)
        hist_timestamps = all_timestamps[current_idx - correlation_window:current_idx]
        
        # extract historical feature data for correlation computation
        correlation_data = extract_correlation_data(
            df, hist_timestamps, clusters, cluster_to_idx, correlation_col
        )
        
        # compute correlation graph
        edge_index = compute_correlation_graph(
            correlation_data, clusters, min_correlation
        )
        
        edge_seq.append(edge_index)
    
    return edge_seq


def extract_correlation_data(
    df: pl.DataFrame,
    timestamps: List,
    clusters: List,
    cluster_to_idx: Dict,
    correlation_col: str
) -> np.ndarray:
    """Extract historical feature data for correlation computation (data leakage safe)."""
    num_timestamps = len(timestamps)
    num_clusters = len(clusters)
    
    # initialize correlation data [time_steps, clusters]
    correlation_data = np.zeros((num_timestamps, num_clusters))
    
    for t_idx, ts in enumerate(timestamps):
        ts_data = df.filter(pl.col('ts_start') == ts)
        
        if ts_data.height > 0:
            for cluster_id in clusters:
                cluster_idx = cluster_to_idx[cluster_id]
                cluster_data = ts_data.filter(pl.col('cluster_id') == cluster_id)
                
                if cluster_data.height > 0 and correlation_col in cluster_data.columns:
                    try:
                        correlation_value = cluster_data.select(correlation_col).to_numpy()[0, 0]
                        correlation_data[t_idx, cluster_idx] = correlation_value
                    except:
                        continue
    
    return correlation_data


def compute_correlation_graph(
    correlation_data: np.ndarray,
    clusters: List,
    min_correlation: float
) -> np.ndarray:
    """
    Compute graph edges based on historical feature correlations.
    
    Creates graph connections between clusters that show similar historical patterns.
    Uses only historical data to prevent data leakage.
    """
    num_clusters = len(clusters)
    
    # compute pairwise correlations
    try:
        corr_matrix = np.corrcoef(correlation_data.T)
        # handle NaN correlations
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    except:
        # fallback to empty graph if correlation computation fails
        return np.empty((2, 0), dtype=np.int64)
    
    # create edges based on correlation threshold
    edges = []
    
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j and abs(corr_matrix[i, j]) >= min_correlation:
                edges.append([i, j])
    
    # convert to edge_index format [2, num_edges]
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
        
        # load split data
        df = pl.read_parquet(str(input_file))
        
        print(f"  Loaded {df.shape[0]} records with {df.shape[1]} columns")
        
        # create temporal sequences
        try:
            sequences, seq_metadata = create_temporal_sequences(
                df=df,
                sequence_length=args.sequence_length,
                prediction_horizon=args.prediction_horizon,
                correlation_window=args.correlation_window
            )
            
            # save sequences
            output_file = output_dir / f'{split}_temporal_sequences.pkl'
            save_temporal_sequences(sequences, seq_metadata, output_file)
            
            print(f"  Successfully created {len(sequences)} sequences for {split}")
            
        except Exception as e:
            print(f"  Error processing {split}: {str(e)}")
            continue
        
        # cleanup memory
        del df, sequences
        gc.collect()
    
    # save processing configuration
    config = {
        'sequence_length': args.sequence_length,
        'prediction_horizon': args.prediction_horizon,
        'correlation_window': args.correlation_window,
        'min_correlation': args.min_correlation,
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
    print(f"\nTo train the Gated GCN model, run:")
    print(f"python scripts/train_temporal_gnn.py --data_dir {output_dir}")


if __name__ == '__main__':
    main() 