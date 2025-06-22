#!/usr/bin/env python3
"""
Prepare clustered EcoBici dataset for Graph Neural Network (GNN) training.

This script processes the clustered dataset to:
1. Remove features that cause data leakage (including ALL internal features)
2. Create proper targets for predicting external arrivals in next time step
3. Structure data for GNN with spatial relationships between clusters
4. Save the processed dataset in PyTorch Geometric format

Focus: Predict external movements (inter-cluster trips) - arrivals, departures, or both.
Internal movements within clusters are excluded as they're not useful for prediction.

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

def prepare_temporal_sequences(
    df: pl.DataFrame,
    target_cols: List[str],
    sequence_length: int = 24,  # 24 * 30min = 12 hours
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare temporal sequences for each cluster.
    
    Args:
        df: Input dataframe sorted by cluster_id and ts_start
        sequence_length: Number of historical time steps
        prediction_horizon: Number of future steps to predict
    
    Returns:
        Tuple of (X, y, timestamps, feature_names)
    """
    # Sort by cluster and time
    df_sorted = df.sort(['cluster_id', 'ts_start'])
    
    # Get feature columns (excluding cluster_id, ts_start, and targets)
    exclude_cols = ['cluster_id', 'ts_start'] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    sequences = []
    targets = []
    timestamps = []
    
    # Group by cluster
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_data = df_sorted.filter(pl.col('cluster_id') == cluster_id)
        
        n_samples = len(cluster_data)
        if n_samples < sequence_length + prediction_horizon:
            continue
            
        cluster_features = cluster_data.select(feature_cols).to_numpy()
        
        # Check if target columns exist in data
        available_targets = [t for t in target_cols if t in cluster_data.columns]
        if not available_targets:
            print(f"Warning: No target columns found for cluster {cluster_id}")
            print(f"  Looking for: {target_cols}")
            print(f"  Available columns: {[c for c in cluster_data.columns if 'arr_' in c]}")
            continue
            
        cluster_targets = cluster_data.select(available_targets).to_numpy()
        cluster_timestamps = cluster_data.select('ts_start').to_numpy().flatten()
        
        # Create sequences
        for i in range(n_samples - sequence_length - prediction_horizon + 1):
            # Input sequence
            X_seq = cluster_features[i:i + sequence_length]
            # Target (next time step)
            y_seq = cluster_targets[i + sequence_length:i + sequence_length + prediction_horizon]
            # Timestamp of prediction
            ts_seq = cluster_timestamps[i + sequence_length]
            
            sequences.append(X_seq)
            targets.append(y_seq.flatten())  # Flatten for single step prediction
            timestamps.append(ts_seq)
    
    return np.array(sequences), np.array(targets), np.array(timestamps), feature_cols

def create_pytorch_geometric_data(
    X: np.ndarray,
    y: np.ndarray, 
    timestamps: np.ndarray,
    edge_index: torch.Tensor,
    feature_names: List[str],
    target_names: List[str],
    cluster_features: Optional[Dict] = None
) -> Data:
    """
    Create PyTorch Geometric Data object.
    
    Args:
        X: Node features [n_nodes, sequence_length, n_features]
        y: Target values [n_nodes, n_targets]
        timestamps: Timestamps for each sample
        edge_index: Graph edges
        feature_names: Names of input features
        target_names: Names of target features
        cluster_features: Additional static cluster features
    
    Returns:
        PyTorch Geometric Data object
    """
    # Flatten temporal features for each node
    n_nodes, seq_len, n_features = X.shape
    X_flat = X.reshape(n_nodes, seq_len * n_features)
    
    # Convert to tensors
    x = torch.tensor(X_flat, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Add cluster static features if available
    if cluster_features is not None:
        cluster_static = []
        for i in range(n_nodes):
            cluster_info = cluster_features.get(str(i), {})
            static_features = [
                cluster_info.get('lat', 0.0),
                cluster_info.get('lon', 0.0), 
                cluster_info.get('station_count', 0)
            ]
            cluster_static.append(static_features)
        
        cluster_static_tensor = torch.tensor(cluster_static, dtype=torch.float32)
        x = torch.cat([x, cluster_static_tensor], dim=1)
        
        # Update feature names
        static_feature_names = ['cluster_lat', 'cluster_lon', 'cluster_station_count']
        expanded_feature_names = []
        for t in range(seq_len):
            for fname in feature_names:
                expanded_feature_names.append(f"{fname}_t_{t}")
        expanded_feature_names.extend(static_feature_names)
        feature_names = expanded_feature_names
    
    # Create data object
    data = Data(
        x=x,
        y=y_tensor,
        edge_index=edge_index,
        timestamps=timestamps,
        feature_names=feature_names,
        target_names=target_names
    )
    
    return data

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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = Path(args.input_dir) / 'dataset_metadata.json'
    metadata = load_cluster_metadata(str(metadata_path))
    
    print(f"Loaded metadata for {metadata['n_clusters']} clusters")
    print(f"Data splits: Train={metadata['train_shape']}, Val={metadata['val_shape']}, Test={metadata['test_shape']}")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Load data
        parquet_path = Path(args.input_dir) / f'{split}_cluster_features.parquet'
        df = pl.read_parquet(str(parquet_path))
        
        print(f"Loaded {split} data: {df.shape}")
        
        # Determine target columns first (before cleaning)
        all_target_cols = identify_target_features(df.columns)
        
        # Select targets based on user choice
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
        print(f"Target columns ({args.target}): {target_cols}")
        
        # Identify and remove data leakage features (but keep targets for now)
        features_to_remove = identify_features_to_remove(df.columns)
        
        # Don't remove target columns yet - we need them to create sequences
        features_to_remove_filtered = [f for f in features_to_remove if f not in target_cols]
        
        print(f"Removing {len(features_to_remove_filtered)} features that cause data leakage:")
        print(f"  {features_to_remove_filtered[:10]}{'...' if len(features_to_remove_filtered) > 10 else ''}")
        
        # Keep only valid features + targets
        valid_columns = [c for c in df.columns if c not in features_to_remove_filtered]
        df_clean = df.select(valid_columns)
        
        print(f"Cleaned dataset shape: {df_clean.shape}")
        
        # Prepare sequences for GNN
        print("Creating temporal sequences...")
        X, y, timestamps, feature_names = prepare_temporal_sequences(
            df_clean, 
            target_cols,  # Pass target columns explicitly
            sequence_length=args.sequence_length,
            prediction_horizon=1
        )
        
        print(f"Created {len(X)} sequences with shape: X={X.shape}, y={y.shape}")
        
        # Create cluster graph only once (for train split)
        if split == 'train':
            print("Creating cluster connectivity graph...")
            adj_matrix = create_cluster_adjacency_matrix(
                metadata['cluster_centroids'],
                k_neighbors=args.k_neighbors,
                distance_threshold=args.distance_threshold
            )
            edge_index = create_edge_index_from_adjacency(adj_matrix)
            
            print(f"Created graph with {edge_index.shape[1]} edges")
            
            # Save adjacency matrix and edge index
            np.save(output_dir / 'adjacency_matrix.npy', adj_matrix)
            torch.save(edge_index, output_dir / 'edge_index.pt')
            
        else:
            # Load edge index for val/test
            edge_index = torch.load(output_dir / 'edge_index.pt')
        

        
        # Create PyTorch Geometric data
        print("Creating PyTorch Geometric data object...")
        
        # Note: For now, we'll create a simplified version
        # In a full GNN implementation, you'd need to properly handle
        # the temporal sequences across all clusters simultaneously
        
        # For demonstration, let's create node features as cluster-level aggregations
        # Create mean and std aggregations with explicit column names (for older Polars compatibility)
        mean_exprs = [pl.col(fn).mean().alias(f"{fn}_mean") for fn in feature_names]
        std_exprs = [pl.col(fn).std().alias(f"{fn}_std") for fn in feature_names]
        
        cluster_stats = df_clean.group_by('cluster_id').agg(mean_exprs + std_exprs).sort('cluster_id')
        
        # Get cluster node features
        node_features = cluster_stats.select([c for c in cluster_stats.columns if c != 'cluster_id']).to_numpy()
        node_features = np.nan_to_num(node_features, 0)  # Replace NaN with 0
        
        # Create targets (average values per cluster for selected targets)
        cluster_targets = df_clean.group_by('cluster_id').agg([
            pl.col(target_cols).mean()
        ]).sort('cluster_id')
        target_values = cluster_targets.select(target_cols).to_numpy()
        
        # Create data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            y=torch.tensor(target_values, dtype=torch.float32),
            edge_index=edge_index,
            num_nodes=metadata['n_clusters']
        )
        
        # Save processed data
        torch.save(data, output_dir / f'{split}_data.pt')
        
        # Save feature and target names
        feature_cols_expanded = [f"{fn}_{stat}" for fn in feature_names for stat in ['mean', 'std']]
        
        with open(output_dir / f'{split}_feature_names.json', 'w') as f:
            json.dump({
                'features': feature_cols_expanded,
                'targets': target_cols,
                'n_features': len(feature_cols_expanded),
                'n_targets': len(target_cols)
            }, f, indent=2)
        
        print(f"Saved {split} data: {data.x.shape[0]} nodes, {data.x.shape[1]} features")
    
    # Save processing configuration
    config = {
        'sequence_length': args.sequence_length,
        'k_neighbors': args.k_neighbors, 
        'distance_threshold': args.distance_threshold,
        'target': args.target,
        'features_removed': features_to_remove,
        'n_clusters': metadata['n_clusters'],
        'dt_minutes': metadata['dt_minutes']
    }
    
    with open(output_dir / 'processing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nProcessing complete! Data saved to {output_dir}")
    print(f"Graph has {metadata['n_clusters']} nodes and {edge_index.shape[1]} edges")
    print(f"Features per node: {node_features.shape[1]}")
    print(f"Target variables: {target_cols}")
    print(f"All internal features removed as requested")

if __name__ == '__main__':
    main() 