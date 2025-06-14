"""
Embedding utilities for EcoBici data.

This module provides functionality for:
- Station embeddings and neural network features  
- Coordinate embedding transformations
- User embedding mappings
- Feature encoding for deep learning models
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn


def create_station_embeddings_mapping(df: pl.DataFrame, 
                                     station_col: str = "station_id") -> Tuple[Dict[int, int], int]:
    """
    Create mapping from station_id to embedding index for neural networks.
    
    Args:
        df: DataFrame containing station_id column
        station_col: Name of the station ID column
        
    Returns:
        Tuple of (station_id_to_index_dict, num_stations)
    """
    print("Creating station embeddings mapping...")
    
    unique_stations = sorted(df[station_col].unique().to_list())
    station_id_to_idx = {station_id: idx for idx, station_id in enumerate(unique_stations)}
    
    print(f"  -> Created mapping for {len(unique_stations)} unique stations")
    
    return station_id_to_idx, len(unique_stations)


def add_station_embedding_index(df: pl.DataFrame, 
                               station_id_to_idx: Dict[int, int],
                               station_col: str = "station_id") -> pl.DataFrame:
    """
    Add station embedding index column based on station_id mapping.
    
    Args:
        df: DataFrame with station_id column
        station_id_to_idx: Mapping from station_id to embedding index
        station_col: Name of the station ID column
        
    Returns:
        DataFrame with added station_idx column
    """
    print("Adding station embedding indices...")
    
    # convert mapping to polars expression
    mapping_expr = pl.col(station_col).replace(station_id_to_idx, default=None)
    
    df = df.with_columns([
        mapping_expr.alias("station_idx")
    ])
    
    print(f"  -> Added embedding indices for {df.height:,} rows")
    
    return df


def create_user_embeddings_mapping(df: pl.DataFrame,
                                  user_col: str = "id_usuario") -> Tuple[Dict[int, int], int]:
    """
    Create mapping from user_id to embedding index for neural networks.
    
    Args:
        df: DataFrame containing user_id column
        user_col: Name of the user ID column
        
    Returns:
        Tuple of (user_id_to_index_dict, num_users)
    """
    print("Creating user embeddings mapping...")
    
    unique_users = sorted(df[user_col].unique().to_list())
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    print(f"  -> Created mapping for {len(unique_users)} unique users")
    
    return user_id_to_idx, len(unique_users)


def encode_categorical_features(df: pl.DataFrame, 
                               categorical_cols: List[str],
                               create_embeddings: bool = True) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features for deep learning models.
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of categorical column names
        create_embeddings: Whether to create embedding mappings
        
    Returns:
        Tuple of (encoded_dataframe, encoders_dict)
    """
    print(f"Encoding {len(categorical_cols)} categorical features...")
    
    df_encoded = df.clone()
    encoders = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            print(f"  -> Warning: Column '{col}' not found, skipping")
            continue
            
        print(f"  -> Encoding {col}...")
        
        # get unique values
        unique_values = sorted(df[col].unique().to_list())
        
        if create_embeddings:
            # create embedding mapping
            value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            encoders[col] = {
                'type': 'embedding',
                'mapping': value_to_idx,
                'num_classes': len(unique_values),
                'embedding_dim': min(50, (len(unique_values) + 1) // 2)  # rule of thumb
            }
            
            # add encoded column
            df_encoded = df_encoded.with_columns([
                pl.col(col).replace(value_to_idx, default=None).alias(f"{col}_idx")
            ])
            
        else:
            # simple label encoding
            value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            encoders[col] = {
                'type': 'label',
                'mapping': value_to_idx,
                'num_classes': len(unique_values)
            }
            
            df_encoded = df_encoded.with_columns([
                pl.col(col).replace(value_to_idx, default=None).alias(f"{col}_encoded")
            ])
    
    print(f"  -> Encoded {len([c for c in categorical_cols if c in df.columns])} categorical features")
    
    return df_encoded, encoders


def create_coordinate_embeddings(df: pl.DataFrame,
                                lat_col: str = "lat",
                                lon_col: str = "lon",
                                embedding_dim: int = 8,
                                grid_size: int = 20) -> pl.DataFrame:
    """
    Create coordinate-based embeddings by discretizing lat/lon into a grid.
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        embedding_dim: Dimension of coordinate embeddings
        grid_size: Size of the coordinate grid (grid_size x grid_size)
        
    Returns:
        DataFrame with coordinate embedding indices
    """
    print(f"Creating coordinate embeddings with {grid_size}x{grid_size} grid...")
    
    # calculate coordinate bounds
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    
    print(f"  -> Coordinate bounds: lat=[{lat_min:.6f}, {lat_max:.6f}], lon=[{lon_min:.6f}, {lon_max:.6f}]")
    
    # discretize coordinates to grid
    df_with_grid = df.with_columns([
        # create grid indices
        ((pl.col(lat_col) - lat_min) / (lat_max - lat_min) * (grid_size - 1))
        .round(0).clip(0, grid_size - 1).cast(pl.Int32).alias("lat_grid"),
        
        ((pl.col(lon_col) - lon_min) / (lon_max - lon_min) * (grid_size - 1))
        .round(0).clip(0, grid_size - 1).cast(pl.Int32).alias("lon_grid")
    ])
    
    # create combined coordinate embedding index
    df_with_grid = df_with_grid.with_columns([
        (pl.col("lat_grid") * grid_size + pl.col("lon_grid")).alias("coord_embedding_idx")
    ])
    
    num_coord_embeddings = grid_size * grid_size
    print(f"  -> Created {num_coord_embeddings} coordinate embedding indices")
    
    return df_with_grid


def prepare_features_for_neural_network(df: pl.DataFrame,
                                       numerical_cols: List[str],
                                       categorical_cols: List[str],
                                       target_col: str,
                                       test_size: float = 0.2,
                                       val_size: float = 0.1) -> Dict[str, Any]:
    """
    Prepare features for neural network training with proper scaling and encoding.
    
    Args:
        df: DataFrame with features
        numerical_cols: List of numerical feature columns
        categorical_cols: List of categorical feature columns  
        target_col: Name of target column
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        
    Returns:
        Dictionary with prepared datasets and metadata
    """
    print("Preparing features for neural network...")
    
    # convert to pandas for sklearn compatibility
    df_pd = df.to_pandas()
    
    # separate features and target
    X_numerical = df_pd[numerical_cols].copy()
    X_categorical = df_pd[categorical_cols].copy()
    y = df_pd[target_col].copy()
    
    # scale numerical features
    print("  -> Scaling numerical features...")
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=df_pd.index)
    
    # encode categorical features
    print("  -> Encoding categorical features...")
    categorical_encoders = {}
    X_categorical_encoded = X_categorical.copy()
    
    for col in categorical_cols:
        if col in X_categorical.columns:
            le = LabelEncoder()
            X_categorical_encoded[col] = le.fit_transform(X_categorical[col].fillna('missing'))
            categorical_encoders[col] = le
    
    # combine features
    X_combined = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)
    
    # split data
    print("  -> Splitting data...")
    n_total = len(X_combined)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    
    # temporal split (assuming data is time-ordered)
    X_train = X_combined.iloc[:n_train]
    X_val = X_combined.iloc[n_train:n_train+n_val]
    X_test = X_combined.iloc[n_train+n_val:]
    
    y_train = y.iloc[:n_train]
    y_val = y.iloc[n_train:n_train+n_val]
    y_test = y.iloc[n_train+n_val:]
    
    print(f"  -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'numerical_scaler': scaler,
        'categorical_encoders': categorical_encoders,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_dim': X_combined.shape[1]
    }


class StationEmbedding(nn.Module):
    """Neural network module for station embeddings."""
    
    def __init__(self, num_stations: int, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_stations, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, station_ids):
        return self.dropout(self.embedding(station_ids))


class CoordinateEmbedding(nn.Module):
    """Neural network module for coordinate embeddings."""
    
    def __init__(self, grid_size: int = 20, embedding_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(grid_size * grid_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, coord_indices):
        return self.dropout(self.embedding(coord_indices))


class TemporalEmbedding(nn.Module):
    """Neural network module for temporal embeddings."""
    
    def __init__(self, 
                 hour_dim: int = 8,
                 dow_dim: int = 4, 
                 month_dim: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.hour_embedding = nn.Embedding(24, hour_dim)
        self.dow_embedding = nn.Embedding(7, dow_dim)
        self.month_embedding = nn.Embedding(12, month_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hour, dow, month):
        hour_emb = self.hour_embedding(hour)
        dow_emb = self.dow_embedding(dow)  
        month_emb = self.month_embedding(month)
        
        combined = torch.cat([hour_emb, dow_emb, month_emb], dim=-1)
        return self.dropout(combined)


def convert_to_torch_tensors(data: Dict[str, Any], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Convert prepared data to PyTorch tensors.
    
    Args:
        data: Dictionary from prepare_features_for_neural_network
        device: Device to place tensors on
        
    Returns:
        Dictionary with PyTorch tensors
    """
    print(f"Converting to PyTorch tensors on device: {device}")
    
    tensors = {}
    
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        if key in data:
            tensor_data = torch.FloatTensor(data[key].values).to(device)
            tensors[key] = tensor_data
            print(f"  -> {key}: {tensor_data.shape}")
    
    return tensors


def create_embedding_layer_configs(encoders: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Create embedding layer configurations from encoders.
    
    Args:
        encoders: Dictionary of encoders from encode_categorical_features
        
    Returns:
        Dictionary with embedding configurations
    """
    configs = {}
    
    for col, encoder_info in encoders.items():
        if encoder_info['type'] == 'embedding':
            configs[col] = {
                'num_embeddings': encoder_info['num_classes'],
                'embedding_dim': encoder_info['embedding_dim']
            }
    
    print(f"Created embedding configs for {len(configs)} categorical features")
    
    return configs 