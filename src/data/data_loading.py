"""
Data loading utilities for EcoBici-AI project.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, List
import os
import glob


def load_all_raw_data(data_dir: Union[str, Path] = "data/raw") -> pd.DataFrame:
    """
    Load all raw EcoBici CSV files from the specified directory.
    
    Args:
        data_dir: Directory containing raw CSV files
        
    Returns:
        Combined DataFrame with all trip data
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # load and combine all files
    dataframes = []
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file.name}: {e}")
    
    if not dataframes:
        raise ValueError("No data could be loaded from CSV files")
    
    # combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    return combined_df


def load_ecobici_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load EcoBici data from a single file (CSV or Parquet).
    
    Args:
        data_path: Path to the data file
        
    Returns:
        DataFrame with trip data
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.suffix.lower() == '.csv':
        return pd.read_csv(data_path)
    elif data_path.suffix.lower() == '.parquet':
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def load_processed_data(data_dir: Union[str, Path] = "data/processed") -> dict:
    """
    Load processed data files (parquet format).
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Dictionary with loaded DataFrames
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    
    data = {}
    
    # load common processed files
    common_files = [
        'dispatch_counts.parquet',
        'arrival_counts.parquet',
        'station_features.parquet',
        'trip_features.parquet'
    ]
    
    for filename in common_files:
        filepath = data_dir / filename
        if filepath.exists():
            data[filename.replace('.parquet', '')] = pd.read_parquet(filepath)
    
    # load any other parquet files
    for parquet_file in data_dir.glob("*.parquet"):
        if parquet_file.name not in common_files:
            key = parquet_file.stem
            data[key] = pd.read_parquet(parquet_file)
    
    return data


def load_clustered_data(data_dir: Union[str, Path] = "data/clustered") -> dict:
    """
    Load clustered data files for training.
    
    Args:
        data_dir: Directory containing clustered data files
        
    Returns:
        Dictionary with train/val/test DataFrames
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Clustered data directory not found: {data_dir}")
    
    data = {}
    
    # load train/val/test splits
    for split in ['train', 'val', 'test']:
        filepath = data_dir / f"{split}_cluster_features.parquet"
        if filepath.exists():
            data[split] = pd.read_parquet(filepath)
    
    return data


def save_data(df: pd.DataFrame, output_path: Union[str, Path], 
              format: str = 'parquet') -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        format: File format ('parquet' or 'csv')
    """
    output_path = Path(output_path)
    
    # create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path}")


__all__ = [
    'load_all_raw_data',
    'load_ecobici_data',
    'load_processed_data',
    'load_clustered_data',
    'save_data'
] 