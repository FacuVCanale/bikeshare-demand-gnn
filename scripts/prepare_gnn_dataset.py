#!/usr/bin/env python3
"""
Prepare EcoBici dataset for Graph Neural Network (GNN) training.

This script takes the dataset exactly as output by feature_engineering.py and:
1. Uses the data without any modifications or aggregations
2. Creates spatial relationships between stations as graph edges
3. Structures data for GNN with PyTorch Geometric format
4. Prepares temporal splits for training

No feature engineering - uses data exactly as provided.

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
import torch
from torch_geometric.data import Data
import gc

def load_station_coordinates_from_data(df: pl.DataFrame) -> Dict:
    """Extract station coordinates from the dataset to create graph structure."""
    print("Extracting station coordinates for graph creation...")
    
    # TODO: Implement real coordinate extraction from actual data
    print("  Coordinate extraction not implemented yet")
    pass

def create_station_adjacency_matrix(
    station_coordinates: Dict, 
    k_neighbors: int = 5,
    distance_threshold: Optional[float] = None
) -> np.ndarray:
    """Create adjacency matrix for stations based on geographical proximity."""
    n_stations = len(station_coordinates)
    
    # Extract coordinates
    coords = []
    station_ids = []
    for station_id, info in station_coordinates.items():
        coords.append([info['lat'], info['lon']])
        station_ids.append(int(station_id))
    
    coords = np.array(coords)
    
    # Use KNN to find neighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_stations), metric='haversine')
    nn.fit(np.radians(coords))  # haversine expects radians
    
    # Find neighbors
    distances, indices = nn.kneighbors(np.radians(coords))
    
    # Convert distances from radians to km
    earth_radius_km = 6371
    distances = distances * earth_radius_km
    
    # Create adjacency matrix
    adj_matrix = np.zeros((n_stations, n_stations))
    
    for i in range(n_stations):
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

def create_station_id_mapping(station_ids: List[int]) -> Dict[int, int]:
    """Create mapping from station_id to node index for graph."""
    return {station_id: idx for idx, station_id in enumerate(sorted(station_ids))}

def prepare_data_for_gnn(
    df: pl.DataFrame,
    station_id_mapping: Dict[int, int],
    target_cols: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Prepare data exactly as it comes from feature_engineering.py for GNN.
    No aggregations or modifications - just format conversion.
    """
    print("Preparing data for GNN (no modifications)...")
    
    # Identify feature columns (exclude metadata and targets)
    exclude_cols = ['datetime', 'station_id'] + target_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  Using {len(feature_cols)} feature columns as-is")
    print(f"  Using {len(target_cols)} target columns")
    
    # Add node index for graph structure
    df = df.with_columns([
        pl.col('station_id').map_elements(
            lambda x: station_id_mapping[x], 
            return_dtype=pl.Int64
        ).alias('node_idx')
    ])
    
    # Sort by node index and datetime for consistency
    df = df.sort(['node_idx', 'datetime'])
    
    # Extract features and targets
    features = df.select(feature_cols).to_numpy(writable=False)
    targets = df.select(target_cols).to_numpy(writable=False)
    
    # Handle NaN values
    features = np.nan_to_num(features, copy=False, nan=0.0)
    targets = np.nan_to_num(targets, copy=False, nan=0.0)
    
    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    
    print(f"  Data shape: {x.shape[0]} samples, {x.shape[1]} features")
    print(f"  Target shape: {y.shape}")
    
    return x, y, feature_cols

def main():
    parser = argparse.ArgumentParser(description='Prepare GNN dataset from feature_engineering.py output')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to single input file from feature_engineering.py (CSV or Parquet)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing split files from feature_engineering.py (train_features.csv, etc.)')
    parser.add_argument('--output_dir', type=str, default='data/gnn_ready',
                        help='Output directory for processed GNN data')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='Number of nearest neighbors for graph connectivity')
    parser.add_argument('--distance_threshold', type=float, default=5.0,
                        help='Maximum distance (km) for station connectivity')
    parser.add_argument('--target', type=str, default='arrivals',
                        choices=['arrivals', 'departures', 'both'],
                        help='Target variable(s) to predict')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction of data for training (only used with single input file)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data for validation (only used with single input file)')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if args.input_file is None and args.input_dir is None:
        raise ValueError("Must specify either --input_file or --input_dir")
    if args.input_file is not None and args.input_dir is not None:
        raise ValueError("Cannot specify both --input_file and --input_dir")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input mode and load data
    if args.input_file is not None:
        # Single file mode - load and create splits internally
        print(f"Single file mode: Loading data from {args.input_file}")
        df = load_single_file(args.input_file)
        
        # Create temporal splits
        print("Creating temporal data splits...")
        df_sorted = df.sort('datetime')
        n_total = len(df_sorted)
        
        train_end = int(n_total * args.train_split)
        val_end = int(n_total * (args.train_split + args.val_split))
        
        splits = {
            'train': df_sorted[:train_end],
            'val': df_sorted[train_end:val_end],
            'test': df_sorted[val_end:]
        }
        
        print(f"Data splits:")
        for split_name, split_df in splits.items():
            print(f"  {split_name}: {len(split_df)} records")
            
    else:
        # Pre-split files mode - load existing splits
        print(f"Pre-split mode: Loading split files from {args.input_dir}")
        splits = load_split_files(args.input_dir)
        
        print(f"Loaded pre-existing splits:")
        for split_name, split_df in splits.items():
            print(f"  {split_name}: {len(split_df)} records")
    
    # Configure target columns (same for both modes)
    # Use first split to determine available columns
    sample_df = next(iter(splits.values()))
    
    print(f"Configuring target strategy: {args.target}")
    if args.target == 'arrivals':
        target_cols = ['arrivals']
    elif args.target == 'departures':
        target_cols = ['departures'] 
    elif args.target == 'both':
        target_cols = ['arrivals', 'departures']
    else:
        raise ValueError(f"Unknown target option: {args.target}")
    
    # Verify targets exist
    missing_targets = [t for t in target_cols if t not in sample_df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")
    
    print(f"Selected target columns: {target_cols}")
    
    # FIXED: Extract station coordinates ONLY from training data to prevent data leakage
    print("Creating graph structure using ONLY training data (preventing data leakage)...")
    train_data = splits['train']
    station_coordinates = load_station_coordinates_from_data(train_data)
    unique_station_ids = sorted(list(station_coordinates.keys()))
    station_id_mapping = create_station_id_mapping(unique_station_ids)
    
    # Check if val/test have stations not in train (which would be problematic)
    all_data = pl.concat(list(splits.values()))
    all_stations = set(all_data.select('station_id').unique().to_series().to_list())
    train_stations = set(train_data.select('station_id').unique().to_series().to_list())
    unseen_stations = all_stations - train_stations
    
    if unseen_stations:
        print(f"  WARNING: Found {len(unseen_stations)} stations in val/test not in train: {sorted(list(unseen_stations))[:5]}...")
        print(f"  These stations will be excluded from val/test splits to prevent leakage")
        
        # Filter val/test to only include stations seen in training
        print("  Filtering val/test splits to only include training stations...")
        filtered_splits = {'train': splits['train']}
        for split_name in ['val', 'test']:
            if split_name in splits:
                original_len = len(splits[split_name])
                filtered_splits[split_name] = splits[split_name].filter(
                    pl.col('station_id').is_in(list(train_stations))
                )
                filtered_len = len(filtered_splits[split_name])
                print(f"    {split_name}: {original_len} → {filtered_len} samples ({original_len-filtered_len} removed)")
        
        splits = filtered_splits
    else:
        print(f"  ✓ All stations in val/test are present in train - no leakage detected")
    
    print(f"  Graph will use {len(unique_station_ids)} stations from training data only")
    
    print("Creating station connectivity graph...")
    print(f"  Graph parameters: k_neighbors={args.k_neighbors}, distance_threshold={args.distance_threshold}km")
    
    adj_matrix = create_station_adjacency_matrix(
        station_coordinates,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold
    )
    edge_index = create_edge_index_from_adjacency(adj_matrix)
    
    # Calculate graph statistics
    n_stations = len(station_coordinates)
    if n_stations > 1:
        density = edge_index.shape[1] / (n_stations * (n_stations - 1))
        avg_edges_per_node = edge_index.shape[1] / n_stations
    else:
        density = 0.0
        avg_edges_per_node = 0.0
        
    print(f"Created graph with {edge_index.shape[1]} edges")
    print(f"  Graph density: {density:.4f}")
    print(f"  Avg edges per node: {avg_edges_per_node:.1f}")
    
    # Save graph structure
    print("Saving graph structure...")
    np.save(output_dir / 'adjacency_matrix.npy', adj_matrix)
    torch.save(edge_index, output_dir / 'edge_index.pt')
    
    # Process each split
    for split_name, split_df in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*60}")
        
        # Prepare data for GNN (no modifications)
        x, y, feature_cols = prepare_data_for_gnn(
            split_df, station_id_mapping, target_cols
        )
        
        if len(x) == 0:
            print(f"  ERROR: No data for {split_name} split")
            continue
        
        # Create PyTorch Geometric data object
        print("Creating PyTorch Geometric data object...")
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            num_nodes=n_stations
        )
        
        # Save processed data
        print(f"Saving {split_name} data...")
        data_file = output_dir / f'{split_name}_data.pt'
        torch.save(data, data_file)
        data_size_mb = data_file.stat().st_size / (1024 * 1024)
        
        # Save metadata (compatible with train_gnn.py naming)
        if split_name == 'train':
            metadata_file = output_dir / 'train_feature_names.json'
        else:
            metadata_file = output_dir / f'{split_name}_metadata.json'
            
        with open(metadata_file, 'w') as f:
            json.dump({
                'features': feature_cols,
                'targets': target_cols,
                'station_ids': unique_station_ids,
                'station_id_mapping': station_id_mapping,
                'n_features': len(feature_cols),
                'n_targets': len(target_cols),
                'n_stations': n_stations,
                'n_samples': len(x)
            }, f, indent=2)
        
        print(f"  Saved {split_name} data: {data.x.shape[0]} samples, {data.x.shape[1]} features")
        print(f"  File size: {data_size_mb:.1f} MB")
        print(f"  Targets: {data.y.shape[1]} variables")
        
        # Clean up memory
        del x, y, data
        gc.collect()
    
    # Save processing configuration
    config = {
        'input_mode': 'single_file' if args.input_file else 'pre_split',
        'input_source': str(args.input_file) if args.input_file else str(args.input_dir),
        'target_type': args.target,
        'k_neighbors': args.k_neighbors,
        'distance_threshold': args.distance_threshold,
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'n_stations': n_stations,
        'feature_engineering_compatible': True,
        'no_modifications': True
    }
    
    # Add split info for single file mode
    if args.input_file:
        config.update({
            'train_split': args.train_split,
            'val_split': args.val_split
        })
    
    with open(output_dir / 'processing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save station coordinates and mapping
    with open(output_dir / 'station_coordinates.json', 'w') as f:
        json.dump(station_coordinates, f, indent=2)
    
    with open(output_dir / 'station_mapping.json', 'w') as f:
        json.dump(station_id_mapping, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    
    if args.input_file:
        print(f"Input: {args.input_file} (single file)")
    else:
        print(f"Input: {args.input_dir} (pre-split files)")
        
    print(f"Output directory: {output_dir}")
    print(f"Target strategy: {args.target}")
    print(f"Graph connectivity: {args.k_neighbors} neighbors, {args.distance_threshold}km threshold")
    print(f"Graph structure: {n_stations} station nodes, {edge_index.shape[1]} edges")
    
    print(f"\nFiles created:")
    total_size_mb = 0
    for file in sorted(output_dir.glob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  {file.name} ({size_mb:.1f} MB)")
    print(f"  Total size: {total_size_mb:.1f} MB")
    
    print(f"\nKey features:")
    print(f"  - Uses data exactly as provided by feature_engineering.py")
    print(f"  - No aggregations or modifications")
    print(f"  - Station-level graph structure with {edge_index.shape[1]} edges")
    print(f"  - Temporal data splitting")
    print(f"  - PyTorch Geometric format ready for GNN training")
    
    print(f"\nTo train a GNN model, run:")
    print(f"python scripts/train_gnn.py --data_dir {output_dir}")


def load_single_file(file_path):
    """Load data from a single file (original behavior)."""
    input_path = Path(file_path)
    if input_path.suffix.lower() == '.parquet':
        print("  Detected parquet format")
        df = pl.read_parquet(str(input_path))
    elif input_path.suffix.lower() == '.csv':
        print("  Detected CSV format")
        df = pl.read_csv(str(input_path), try_parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Date range: {df.select(pl.col('datetime').min()).item()} to {df.select(pl.col('datetime').max()).item()}")
    
    return df


def load_split_files(input_dir):
    """Load pre-split files from feature_engineering.py output."""
    input_path = Path(input_dir)
    
    # Expected file names from feature_engineering.py
    split_files = {
        'train': input_path / 'train_features.csv',
        'val': input_path / 'val_features.csv', 
        'test': input_path / 'test_features.csv'
    }
    
    # Check if all split files exist
    missing_files = [str(path) for path in split_files.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing split files: {missing_files}")
    
    # Load all splits
    splits = {}
    for split_name, file_path in split_files.items():
        print(f"  Loading {split_name}: {file_path}")
        df = pl.read_csv(str(file_path), try_parse_dates=True)
        splits[split_name] = df
        
        start_date = df.select(pl.col('datetime').min()).item()
        end_date = df.select(pl.col('datetime').max()).item()
        print(f"    {len(df)} records ({start_date} to {end_date})")
    
    # Load metadata if available
    metadata_path = input_path / 'feature_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Loaded feature metadata: {len(metadata.get('feature_names', []))} features")
    
    return splits

if __name__ == '__main__':
    main() 