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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
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

def create_temporal_snapshot_data(
    df: pl.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    timestamp: pd.Timestamp,
    edge_index: torch.Tensor,
    n_clusters: int,
    sequence_length: int = 24
) -> Optional[Data]:
    """
    Create a single temporal snapshot for GNN training.
    
    This function creates one Data object representing the graph state at a specific
    timestamp, using historical data as features and next timestep as target.
    
    Args:
        df: DataFrame with all temporal data
        feature_cols: List of feature column names
        target_cols: List of target column names
        timestamp: The current timestamp for this snapshot
        edge_index: Graph connectivity
        n_clusters: Total number of clusters
        sequence_length: Number of historical timesteps to use as features
        
    Returns:
        PyTorch Geometric Data object or None if insufficient data
    """
    # Get data up to and including current timestamp
    historical_mask = df['ts_start'] <= timestamp
    historical_data = df.filter(historical_mask)
    
    # Get next timestep for targets
    # Assuming 30-minute intervals
    next_timestamp = timestamp + pd.Timedelta(minutes=30)
    target_mask = df['ts_start'] == next_timestamp
    target_data = df.filter(target_mask)
    
    if target_data.height == 0:
        return None  # No target data available
    
    # Initialize feature matrix for all clusters
    node_features = []
    node_targets = []
    valid_nodes = []
    
    for cluster_id in range(n_clusters):
        # Get historical data for this cluster
        cluster_hist = historical_data.filter(pl.col('cluster_id') == cluster_id).sort('ts_start')
        cluster_target = target_data.filter(pl.col('cluster_id') == cluster_id)
        
        if cluster_hist.height < sequence_length or cluster_target.height == 0:
            # Insufficient data for this cluster - use zeros
            node_features.append(np.zeros(len(feature_cols) * sequence_length))
            node_targets.append(np.zeros(len(target_cols)))
            valid_nodes.append(0)  # Mark as invalid
        else:
            # Get last sequence_length timesteps
            recent_data = cluster_hist.tail(sequence_length)
            
            # Extract features and flatten temporal dimension
            features = recent_data.select(feature_cols).to_numpy().flatten()
            targets = cluster_target.select(target_cols).to_numpy().flatten()
            
            node_features.append(features)
            node_targets.append(targets)
            valid_nodes.append(1)  # Mark as valid
    
    # Convert to tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    y = torch.tensor(np.array(node_targets), dtype=torch.float32)
    valid_mask = torch.tensor(valid_nodes, dtype=torch.bool)
    
    # Create data object
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        valid_mask=valid_mask,  # Mask to identify nodes with valid data
        timestamp=timestamp.isoformat(),
        num_nodes=n_clusters
    )
    
    return data

def prepare_temporal_gnn_dataset(
    df: pl.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    edge_index: torch.Tensor,
    n_clusters: int,
    sequence_length: int = 24,
    stride: int = 1
) -> List[Data]:
    """
    Prepare temporal GNN dataset with proper time-based splits.
    
    Creates multiple temporal snapshots, each representing the graph state
    at a specific time with historical features and future targets.
    
    Args:
        df: DataFrame with temporal data
        feature_cols: List of feature columns
        target_cols: List of target columns
        edge_index: Graph connectivity
        n_clusters: Number of clusters
        sequence_length: Historical window size
        stride: Number of timesteps to skip between snapshots
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    print("Creating temporal snapshots for GNN...")
    
    # Get unique timestamps
    timestamps = sorted(df['ts_start'].unique().to_list())
    
    # Need at least sequence_length + 1 timestamps
    if len(timestamps) < sequence_length + 1:
        print(f"Warning: Not enough timestamps ({len(timestamps)}) for sequence length {sequence_length}")
        return []
    
    # Create snapshots
    data_list = []
    
    # Start from sequence_length to ensure we have enough history
    # End before last timestamp to ensure we have targets
    start_idx = sequence_length
    end_idx = len(timestamps) - 1
    
    for i in tqdm(range(start_idx, end_idx, stride), desc="Creating temporal snapshots"):
        current_timestamp = pd.Timestamp(timestamps[i])
        
        snapshot = create_temporal_snapshot_data(
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            timestamp=current_timestamp,
            edge_index=edge_index,
            n_clusters=n_clusters,
            sequence_length=sequence_length
        )
        
        if snapshot is not None:
            data_list.append(snapshot)
    
    print(f"Created {len(data_list)} temporal snapshots")
    return data_list

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
                        choices=['arr_external_count', 'dep_external_count', 'both_external_counts', 
                                'arr_external_demographics', 'dep_external_demographics', 'all_external_demographics'],
                        help='Target variable(s) to predict')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for temporal snapshots (1 = every timestep, 2 = every other, etc.)')
    
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
    
    # Create cluster graph only once
    print("Creating cluster connectivity graph...")
    print(f"  Graph parameters: k_neighbors={args.k_neighbors}, distance_threshold={args.distance_threshold}km")
    
    adj_matrix = create_cluster_adjacency_matrix(
        metadata['cluster_centroids'],
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold
    )
    edge_index = create_edge_index_from_adjacency(adj_matrix)
    
    # Calculate graph density
    if n_clusters_or_stations > 1:
        density = edge_index.shape[1] / (n_clusters_or_stations * (n_clusters_or_stations - 1))
        avg_edges_per_node = edge_index.shape[1] / n_clusters_or_stations
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
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split...")
        print(f"{'='*60}")
        
        # Determine the correct file naming pattern
        cluster_features_path = Path(args.input_dir) / f'{split}_cluster_features.parquet'
        station_features_path = Path(args.input_dir) / f'{split}_station_features.parquet'
        
        # Auto-detect which file pattern exists
        if cluster_features_path.exists():
            parquet_path = cluster_features_path
        elif station_features_path.exists():
            parquet_path = station_features_path
        else:
            # Try to infer from metadata
            if clustering_enabled:
                parquet_path = cluster_features_path
            else:
                parquet_path = station_features_path
        
        print(f"Loading data from {parquet_path}...")
        df = pl.read_parquet(str(parquet_path))
        print(f"  Initial {split} data shape: {df.shape}")
        
        # Determine target columns
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
            target_cols = all_target_cols
        else:
            raise ValueError(f"Unknown target option: {args.target}")
        
        # Filter targets that exist
        target_cols = [t for t in target_cols if t in all_target_cols]
        print(f"  Selected target columns: {target_cols}")
        
        # Identify and remove data leakage features
        print("Cleaning data leakage features...")
        features_to_remove = identify_features_to_remove(df.columns)
        
        # Don't remove target columns yet - we need them for creating targets
        features_to_remove_filtered = [f for f in features_to_remove if f not in target_cols]
        
        print(f"  Removing {len(features_to_remove_filtered)} leakage features")
        
        # Get feature columns (everything except removed features, targets, and metadata)
        metadata_cols = ['cluster_id', 'ts_start']
        feature_cols = [c for c in df.columns 
                       if c not in features_to_remove_filtered 
                       and c not in target_cols 
                       and c not in metadata_cols]
        
        print(f"  Using {len(feature_cols)} features for model input")
        print(f"  Sample features: {feature_cols[:5]}...")
        
        # Prepare temporal dataset
        data_list = prepare_temporal_gnn_dataset(
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            edge_index=edge_index,
            n_clusters=n_clusters_or_stations,
            sequence_length=args.sequence_length,
            stride=args.stride
        )
        
        if not data_list:
            print(f"  WARNING: No valid temporal snapshots created for {split}")
            continue
        
        # Save processed data
        print(f"Saving {split} data...")
        
        # Save as a list of Data objects
        data_file = output_dir / f'{split}_data_list.pt'
        torch.save(data_list, data_file)
        
        # For compatibility with existing training scripts, also save first snapshot
        if data_list:
            compat_file = output_dir / f'{split}_data.pt'
            torch.save(data_list[0], compat_file)
        
        # Save metadata
        metadata_file = output_dir / f'{split}_feature_names.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'features': feature_cols,
                'targets': target_cols,
                'n_features': len(feature_cols) * args.sequence_length,  # Features are flattened sequences
                'n_targets': len(target_cols),
                'sequence_length': args.sequence_length,
                'n_snapshots': len(data_list),
                'temporal_stride': args.stride
            }, f, indent=2)
        
        print(f"  Saved {split} data: {len(data_list)} temporal snapshots")
        print(f"  Each snapshot: {n_clusters_or_stations} nodes, {len(feature_cols) * args.sequence_length} features")
        print(f"  Targets: {len(target_cols)} variables per node")
        
        # Clean up memory
        del df, data_list
        gc.collect()
    
    # Save processing configuration
    config = {
        'target_type': args.target,
        'k_neighbors': args.k_neighbors,
        'distance_threshold': args.distance_threshold,
        'sequence_length': args.sequence_length,
        'temporal_stride': args.stride,
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'optimization_version': '3.0',  # New version without data leakage
        'n_clusters': n_clusters_or_stations,
        'clustering_enabled': clustering_enabled,
        'dt_minutes': metadata['dt_minutes'],
        'temporal_approach': 'snapshot_based',  # New approach
        'leakage_fixed': True  # Flag to indicate this version fixes leakage
    }
    
    with open(output_dir / 'processing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Target strategy: {args.target} (predicting next timestep values)")
    print(f"Graph connectivity: {args.k_neighbors} neighbors, {args.distance_threshold}km threshold")
    print(f"Temporal approach: Snapshot-based with {args.sequence_length} historical steps")
    print(f"Data leakage: FIXED - using only historical data to predict future")
    
    print(f"\nFiles created:")
    total_size_mb = 0
    for file in sorted(output_dir.glob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  {file.name} ({size_mb:.1f} MB)")
    print(f"  Total size: {total_size_mb:.1f} MB")
    
    print(f"\nIMPORTANT: This version fixes the data leakage issue!")
    print(f"- Features: Only historical data (t-23 to t)")
    print(f"- Targets: Next timestep values (t+1)")
    print(f"- No aggregation across all time periods")
    print(f"\nTo train a GNN model, run:")
    print(f"python scripts/train_gnn.py --data_dir {output_dir}")

if __name__ == '__main__':
    main() 