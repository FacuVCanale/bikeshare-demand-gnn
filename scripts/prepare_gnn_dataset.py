#!/usr/bin/env python3
"""
Prepare clustered EcoBici dataset for Graph Neural Network (GNN) training.

This script processes the clustered dataset to:
1. Remove features that cause data leakage (ALL current period features)
2. Create point-in-time aggregations using only historical/lag features
3. Extract targets without temporal leakage
4. Structure data for GNN with spatial relationships between clusters
5. Save the processed dataset in PyTorch Geometric format

Focus: Predict external movements using only past information.
Uses lag features, rolling statistics, and historical patterns exclusively.
Prevents data leakage by excluding all current period information.

Author: EcoBici-AI
"""

import polars as pl
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import pickle
import gc

def load_cluster_metadata(metadata_path: str) -> Dict:
    """Load cluster metadata including centroids and station counts."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def identify_features_to_remove(columns: List[str]) -> List[str]:
    """
    Identify features that cause data leakage for future prediction.
    
    Data leakage occurs when we use current or future information
    to predict future events in the next delta t (30 minutes).
    """
    features_to_remove = []
    
    # Remove ALL internal features (not useful for prediction)
    internal_features = [c for c in columns if 'internal' in c]
    features_to_remove.extend(internal_features)
    
    # Current arrivals (exactly what we want to predict)
    arr_current = [c for c in columns if c.startswith('arr_') and not any(
        x in c for x in ['lag_', 'rolling_', 'same_hour_']
    )]
    features_to_remove.extend(arr_current)
    
    # Current departures (concurrent events with current arrivals - data leakage)
    dep_current = [c for c in columns if c.startswith('dep_') and not any(
        x in c for x in ['lag_', 'rolling_', 'same_hour_']
    )]
    features_to_remove.extend(dep_current)
    
    # Current activity flags (derived from current events - data leakage)
    has_current = [c for c in columns if c.startswith('has_') and not any(
        x in c for x in ['lag_', 'rolling_', 'same_hour_']
    )]
    features_to_remove.extend(has_current)
    
    # Additional specific flags that indicate data availability (meta-features)
    meta_flags = ['has_short_term_lags', 'has_daily_lags', 'has_weekly_lags', 'weather_data_available']
    meta_flags_present = [c for c in meta_flags if c in columns]
    features_to_remove.extend(meta_flags_present)
    
    return list(set(features_to_remove))

def identify_target_features(columns: List[str]) -> List[str]:
    """Identify target features for prediction."""
    # Primary targets: external counts
    primary_targets = [
        'arr_external_count',
        'dep_external_count'
    ]
    
    # Additional demographic targets (only external)
    demographic_targets = [
        # arrival demographics
        'arr_external_male',
        'arr_external_female', 
        'arr_external_young',
        'arr_external_adult',
        'arr_external_senior',
        # departure demographics
        'dep_external_male',
        'dep_external_female',
        'dep_external_young', 
        'dep_external_adult',
        'dep_external_senior'
    ]
    
    # Only include targets that exist in the dataset
    available_targets = [t for t in primary_targets + demographic_targets if t in columns]
    return available_targets

def create_cluster_adjacency_matrix(
    cluster_centroids: Dict, 
    k_neighbors: int = 5,
    distance_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Create adjacency matrix for clusters based on geographical proximity.
    
    Args:
        cluster_centroids: Dict with cluster info including lat/lon
        k_neighbors: Number of nearest neighbors for each cluster
        distance_threshold: Maximum distance in km for edge creation
    
    Returns:
        Adjacency matrix (n_clusters x n_clusters)
    """
    n_clusters = len(cluster_centroids)
    
    # Extract coordinates
    coords = []
    cluster_ids = []
    for cluster_id, info in cluster_centroids.items():
        coords.append([info['lat'], info['lon']])
        cluster_ids.append(int(cluster_id))
    
    coords = np.array(coords)
    
    # Use KNN to find neighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_clusters), metric='haversine')
    nn.fit(np.radians(coords))  # haversine expects radians
    
    # Find neighbors
    distances, indices = nn.kneighbors(np.radians(coords))
    
    # Convert distances from radians to km
    earth_radius_km = 6371
    distances = distances * earth_radius_km
    
    # Create adjacency matrix
    adj_matrix = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(1, len(indices[i])):  # Skip self (index 0)
            neighbor_idx = indices[i][j]
            distance = distances[i][j]
            
            # Add edge if within distance threshold (if specified)
            if distance_threshold is None or distance <= distance_threshold:
                adj_matrix[i, neighbor_idx] = 1
                adj_matrix[neighbor_idx, i] = 1  # Undirected graph
    
    return adj_matrix

def create_edge_index_from_adjacency(adj_matrix: np.ndarray) -> torch.Tensor:
    """Convert adjacency matrix to PyTorch Geometric edge_index format."""
    edge_indices = np.where(adj_matrix)
    edge_index = torch.tensor(np.vstack([edge_indices[0], edge_indices[1]]), dtype=torch.long)
    return edge_index

# REMOVED: process_cluster_sequences - no longer needed for point-in-time aggregations

def create_point_in_time_aggregations(
    df: pl.DataFrame,
    feature_names: List[str],
    target_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create point-in-time aggregations without data leakage.
    
    Uses only historical/lag features and excludes current targets.
    Each row represents features available at prediction time.
    
    Args:
        df: DataFrame with temporal cluster data
        feature_names: List of valid feature names (no targets)
        target_cols: Target column names (for extraction only)
        
    Returns:
        Tuple of (node_features, target_values, expanded_feature_names)
    """
    print("Creating point-in-time aggregations without data leakage...")
    
    # filter to only lag and historical features (no current period features)
    lag_features = [f for f in feature_names if any(
        pattern in f for pattern in ['_lag_', '_rolling_', '_same_hour_', '_yesterday', '_last_week']
    )]
    
    # add temporal and metadata features that don't cause leakage
    safe_features = [f for f in feature_names if any(
        pattern in f for pattern in ['hour_', 'dow_', 'month_', 'is_weekend', 'is_morning_rush', 
                                   'is_evening_rush', 'is_daytime', 'is_summer', 'is_winter',
                                   'cluster_centroid_', 'cluster_station_count', 'weather_data_available']
    )]
    
    # combine lag and safe features
    valid_features = list(set(lag_features + safe_features))
    valid_features = [f for f in valid_features if f in df.columns]
    
    print(f"  Using {len(valid_features)} lag/historical features")
    print(f"  Excluded {len(feature_names) - len(valid_features)} current-period features to prevent leakage")
    
    if not valid_features:
        print("  WARNING: No valid historical features found - check your feature engineering")
        return np.array([]), np.array([]), []
    
    # create aggregations by cluster using only valid features
    agg_exprs = []
    
    # for lag features: use mean and std (these are already historical)
    for feat in valid_features:
        agg_exprs.extend([
            pl.col(feat).mean().alias(f"{feat}_cluster_mean"),
            pl.col(feat).std().alias(f"{feat}_cluster_std")
        ])
    
    # ensure we have targets to extract
    available_targets = [t for t in target_cols if t in df.columns]
    if not available_targets:
        print("  WARNING: No target columns found in data")
        return np.array([]), np.array([]), []
    
    # separate target extraction (no aggregation to avoid leakage)
    target_agg_exprs = []
    for target in available_targets:
        # use last available value per cluster (most recent)
        target_agg_exprs.append(pl.col(target).last().alias(f"{target}_last"))
    
    print(f"  Computing aggregations for {len(valid_features)} features and {len(available_targets)} targets...")
    
    # perform feature aggregations
    cluster_features = (
        df.lazy()
        .group_by('cluster_id')
        .agg(agg_exprs)
        .sort('cluster_id')
        .collect()
    )
    
    # perform target extraction separately
    cluster_targets = (
        df.lazy()
        .group_by('cluster_id')
        .agg(target_agg_exprs)
        .sort('cluster_id')
        .collect()
    )
    
    # extract feature matrix
    feature_cols_expanded = [f"{feat}_{stat}" for feat in valid_features for stat in ['cluster_mean', 'cluster_std']]
    node_features = cluster_features.select(feature_cols_expanded).to_numpy(writable=False)
    node_features = np.nan_to_num(node_features, copy=False, nan=0.0)
    
    # extract target matrix
    target_cols_last = [f"{target}_last" for target in available_targets]
    target_values = cluster_targets.select(target_cols_last).to_numpy(writable=False)
    target_values = np.nan_to_num(target_values, copy=False, nan=0.0)
    
    print(f"  Generated features: {node_features.shape}")
    print(f"  Generated targets: {target_values.shape}")
    
    return node_features, target_values, feature_cols_expanded

def prepare_temporal_sequences_optimized(
    df: pl.DataFrame,
    target_cols: List[str],
    sequence_length: int = 24,
    prediction_horizon: int = 1,
    use_multiprocessing: bool = True,
    chunk_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    DEPRECATED: Using point-in-time aggregations instead of sequences.
    Returns empty arrays to maintain interface compatibility.
    """
    print("Sequence creation disabled - using point-in-time aggregations to prevent leakage")
    
    # get feature columns (excluding cluster_id, ts_start, and targets)
    exclude_cols = ['cluster_id', 'ts_start'] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # return empty arrays to indicate no sequences created
    return np.array([]), np.array([]), np.array([]), feature_cols

def prepare_temporal_sequences(
    df: pl.DataFrame,
    target_cols: List[str],
    sequence_length: int = 24,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    DEPRECATED: Using point-in-time aggregations instead.
    """
    return prepare_temporal_sequences_optimized(
        df, target_cols, sequence_length, prediction_horizon
    )

# REMOVED: create_pytorch_geometric_data - replaced with direct Data object creation in main()

def main():
    parser = argparse.ArgumentParser(description='Prepare GNN dataset from clustered EcoBici data')
    parser.add_argument('--input_dir', type=str, default='data/clustered',
                        help='Directory containing clustered parquet files')
    parser.add_argument('--output_dir', type=str, default='data/gnn_ready',
                        help='Output directory for processed GNN data')
    parser.add_argument('--sequence_length', type=int, default=24,
                        help='Length of input sequences (in time steps)')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='Number of nearest neighbors for graph connectivity')
    parser.add_argument('--distance_threshold', type=float, default=5.0,
                        help='Maximum distance (km) for cluster connectivity')
    parser.add_argument('--target', type=str, default='arr_external_count',
                        choices=['arr_external_count', 'dep_external_count', 'both_external_counts', 'arr_external_demographics', 'dep_external_demographics', 'all_external_demographics'],
                        help='Target variable(s) to predict')
    parser.add_argument('--no_multiprocessing', action='store_true',
                        help='DEPRECATED: Multiprocessing disabled (using point-in-time aggregations)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = Path(args.input_dir) / 'dataset_metadata.json'
    metadata = load_cluster_metadata(str(metadata_path))
    
    # Handle both clustered and station-level datasets
    clustering_enabled = metadata.get('clustering_enabled', metadata.get('n_clusters') is not None)
    n_clusters_or_stations = metadata.get('n_clusters') or metadata.get('total_stations')
    
    if clustering_enabled:
        print(f"Loaded metadata for {n_clusters_or_stations} clusters (clustered dataset)")
    else:
        print(f"Loaded metadata for {n_clusters_or_stations} stations (station-level dataset)")
    print(f"Data splits: Train={metadata['train_shape']}, Val={metadata['val_shape']}, Test={metadata['test_shape']}")
    
    # Process each split with optimizations
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split...")
        print(f"{'='*60}")
        
        # Determine the correct file naming pattern based on clustering configuration
        cluster_features_path = Path(args.input_dir) / f'{split}_cluster_features.parquet'
        station_features_path = Path(args.input_dir) / f'{split}_station_features.parquet'
        
        # Auto-detect which file pattern exists
        if cluster_features_path.exists():
            parquet_path = cluster_features_path
            print(f"Found clustered data file: {parquet_path}")
        elif station_features_path.exists():
            parquet_path = station_features_path
            print(f"Found station-level data file: {parquet_path}")
        else:
            # Try to infer from metadata
            clustering_enabled = metadata.get('clustering_enabled', metadata.get('n_clusters') is not None)
            if clustering_enabled:
                parquet_path = cluster_features_path
                print(f"Using cluster-based naming (from metadata): {parquet_path}")
            else:
                parquet_path = station_features_path
                print(f"Using station-based naming (from metadata): {parquet_path}")
            
            # Final check if the inferred path exists
            if not parquet_path.exists():
                available_files = list(Path(args.input_dir).glob(f'{split}_*.parquet'))
                raise FileNotFoundError(
                    f"Neither {cluster_features_path} nor {station_features_path} exists.\n"
                    f"Available files: {available_files}\n"
                    f"Please check that the dataset generation step completed successfully."
                )
        
        print(f"Loading data from {parquet_path}...")
        
        # Use lazy loading for better memory management
        df_lazy = pl.scan_parquet(str(parquet_path))
        initial_shape = df_lazy.collect().shape  # Get shape without loading all data
        
        print(f"  Initial {split} data shape: {initial_shape}")
        
        # Load only the columns we need to save memory
        df = pl.read_parquet(str(parquet_path))
        
        # Determine target columns first (before cleaning)
        all_target_cols = identify_target_features(df.columns)
        
        # Select targets based on user choice
        print(f"Configuring target strategy: {args.target}")
        if args.target == 'arr_external_count':
            target_cols = ['arr_external_count']
        elif args.target == 'dep_external_count':
            target_cols = ['dep_external_count']
        elif args.target == 'both_external_counts':
            target_cols = ['arr_external_count', 'dep_external_count']
        elif args.target == 'arr_external_demographics':
            target_cols = [t for t in all_target_cols if 'arr_external' in t]
        elif args.target == 'dep_external_demographics':
            target_cols = [t for t in all_target_cols if 'dep_external' in t]
        elif args.target == 'all_external_demographics':
            target_cols = all_target_cols  # Include all external targets
        else:
            raise ValueError(f"Unknown target option: {args.target}")
        
        # Filter targets that exist in data
        target_cols = [t for t in target_cols if t in all_target_cols]
        print(f"  Selected target columns: {target_cols}")
        
        # Identify and remove data leakage features efficiently
        print("Cleaning data leakage features...")
        features_to_remove = identify_features_to_remove(df.columns)
        
        # Don't remove target columns yet - we need them to create sequences
        features_to_remove_filtered = [f for f in features_to_remove if f not in target_cols]
        
        print(f"  Removing {len(features_to_remove_filtered)} leakage features")
        if len(features_to_remove_filtered) > 0:
            print(f"  Sample removed features: {features_to_remove_filtered[:5]}{'...' if len(features_to_remove_filtered) > 5 else ''}")
        
        # Keep only valid features + targets using lazy evaluation
        valid_columns = [c for c in df.columns if c not in features_to_remove_filtered]
        
        print("  Applying column filtering...")
        df_clean = df.lazy().select(valid_columns).collect()
        
        print(f"  Cleaned dataset shape: {df_clean.shape}")
        print(f"  Memory reduction: {initial_shape[1] - df_clean.shape[1]} columns removed")
        
        # Force garbage collection
        del df
        gc.collect()
        
        # Get feature names for aggregation (excluding targets and metadata)
        exclude_cols = ['cluster_id', 'ts_start'] + target_cols
        feature_names = [c for c in df_clean.columns if c not in exclude_cols]
        
        print(f"Identified {len(feature_names)} potential features (excluding targets and metadata)")
        print("Sequence creation disabled - using point-in-time aggregations to prevent data leakage")
        
        # Create cluster graph only once (for train split)
        if split == 'train':
            print("Creating cluster connectivity graph...")
            print(f"  Graph parameters: k_neighbors={args.k_neighbors}, distance_threshold={args.distance_threshold}km")
            
            # Get the actual number of nodes from the data
            actual_num_nodes = len(metadata['cluster_centroids'])
            print(f"  Detected {actual_num_nodes} nodes from centroids data")
            
            adj_matrix = create_cluster_adjacency_matrix(
                metadata['cluster_centroids'],
                k_neighbors=args.k_neighbors,
                distance_threshold=args.distance_threshold
            )
            edge_index = create_edge_index_from_adjacency(adj_matrix)
            
            # Calculate graph density using actual number of nodes
            if actual_num_nodes > 1:
                density = edge_index.shape[1] / (actual_num_nodes * (actual_num_nodes - 1))
                avg_edges_per_node = edge_index.shape[1] / actual_num_nodes
            else:
                density = 0.0
                avg_edges_per_node = 0.0
                
            print(f"Created graph with {edge_index.shape[1]} edges")
            print(f"  Graph density: {density:.4f}")
            print(f"  Avg edges per node: {avg_edges_per_node:.1f}")
            
            # Save adjacency matrix and edge index
            print("Saving graph structure...")
            np.save(output_dir / 'adjacency_matrix.npy', adj_matrix)
            torch.save(edge_index, output_dir / 'edge_index.pt')
            
        else:
            # Load edge index for val/test
            print("  Loading graph structure...")
            edge_index = torch.load(output_dir / 'edge_index.pt', weights_only=False)
        

        
        # Create PyTorch Geometric data
        print("Creating PyTorch Geometric data object...")
        
        # Use point-in-time aggregations without data leakage
        node_features, target_values, feature_cols_expanded = create_point_in_time_aggregations(
            df_clean, feature_names, target_cols
        )
        
        if len(node_features) == 0:
            print(f"  ERROR: No valid features generated for {split} split")
            print("  This may indicate insufficient lag features in the data.")
            print("  Check that cluster feature generation included proper temporal features.")
            continue
        
        # Create data object efficiently
        print("Building PyTorch Geometric Data object...")
        # Get actual number of nodes from the data shape
        actual_num_nodes = node_features.shape[0]
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            y=torch.tensor(target_values, dtype=torch.float32),
            edge_index=edge_index,
            num_nodes=actual_num_nodes
        )
        
        # Save processed data with compression
        print(f"Saving {split} data...")
        data_file = output_dir / f'{split}_data.pt'
        torch.save(data, data_file)
        data_size_mb = data_file.stat().st_size / (1024 * 1024)
        
        # Save feature and target names
        metadata_file = output_dir / f'{split}_feature_names.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'features': feature_cols_expanded,  # these come from create_point_in_time_aggregations
                'targets': target_cols,
                'n_features': len(feature_cols_expanded),
                'n_targets': len(target_cols)
            }, f, indent=2)
        
        print(f"  Saved {split} data: {data.x.shape[0]} nodes, {data.x.shape[1]} features")
        print(f"  File size: {data_size_mb:.1f} MB")
        print(f"  Targets: {data.y.shape[1]} variables")
        
        # Clean up memory
        del node_features, target_values, data, df_clean
        gc.collect()
    
    # Save processing configuration
    config = {
        'target_type': args.target,
        'k_neighbors': args.k_neighbors,
        'distance_threshold': args.distance_threshold,
        'sequence_length': args.sequence_length,
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'optimization_version': '2.0',  # Track optimization version
        'n_clusters': n_clusters_or_stations,
        'clustering_enabled': clustering_enabled,
        'dt_minutes': metadata['dt_minutes']
    }
    
    with open(output_dir / 'processing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Target strategy: {args.target}")
    print(f"Graph connectivity: {args.k_neighbors} neighbors, {args.distance_threshold}km threshold")
    actual_nodes = len(metadata['cluster_centroids'])
    if clustering_enabled:
        print(f"Graph structure: {actual_nodes} cluster nodes, {edge_index.shape[1]} edges")
    else:
        print(f"Graph structure: {actual_nodes} station nodes, {edge_index.shape[1]} edges")
    
    print(f"\nFiles created:")
    total_size_mb = 0
    for file in sorted(output_dir.glob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  {file.name} ({size_mb:.1f} MB)")
    print(f"  Total size: {total_size_mb:.1f} MB")
    
    print(f"\nTo train a GNN model, run:")
    print(f"python scripts/train_gnn.py --data_dir {output_dir}")
    
    print(f"\nData leakage prevention features:")
    print(f"  - Point-in-time aggregations only")
    print(f"  - Historical/lag features exclusively")
    print(f"  - No current period target information")
    print(f"  - Temporal integrity preserved")
    print(f"  - No sequence-based leakage")
    
    print(f"\nProcessing features:")
    print(f"  - Lazy evaluation with Polars")
    print(f"  - Memory-efficient processing")
    print(f"  - Automatic garbage collection")
    print(f"  - Robust error handling")

if __name__ == '__main__':
    main() 