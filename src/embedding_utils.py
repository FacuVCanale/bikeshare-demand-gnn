"""
Utility functions for working with station embeddings in PyTorch models.

This module provides helper functions for creating and using entity embeddings
for station IDs in neural network models.
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from typing import Dict, Tuple, Optional
import polars as pl
import os


class StationEmbedding(nn.Module):
    """
    Station embedding layer for neural networks.
    
    Uses the recommended embedding dimension based on sqrt(n_stations).
    """
    
    def __init__(self, n_stations: int, embedding_dim: Optional[int] = None, 
                 dropout: float = 0.1):
        super().__init__()
        
        if embedding_dim is None:
            embedding_dim = int(np.sqrt(n_stations))
            
        self.embedding = nn.Embedding(n_stations, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        
    def forward(self, station_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer.
        
        Args:
            station_indices: Tensor of shape (batch_size,) with station indices (0-based)
            
        Returns:
            Embedded stations of shape (batch_size, embedding_dim)
        """
        embedded = self.embedding(station_indices)
        return self.dropout(embedded)


def load_station_mapping(mapping_path: str = "../../data/processed/station_id_to_idx.pkl") -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    Load station ID to index mapping and embedding info.
    
    Args:
        mapping_path: Path to the pickle file with station mapping
        
    Returns:
        Tuple of (station_id_to_idx, embedding_info)
    """
    with open(mapping_path, 'rb') as f:
        station_id_to_idx = pickle.load(f)
        
    # Load embedding info
    info_path = mapping_path.replace('station_id_to_idx.pkl', 'embedding_info.pkl')
    with open(info_path, 'rb') as f:
        embedding_info = pickle.load(f)
        
    return station_id_to_idx, embedding_info


def prepare_station_tensors(df: pl.DataFrame, 
                          station_id_to_idx: Dict[int, int],
                          device: torch.device = None) -> torch.Tensor:
    """
    Convert station IDs in dataframe to tensor of indices for embedding lookup.
    
    Args:
        df: DataFrame with station_id column
        station_id_to_idx: Mapping from station_id to 0-based index
        device: PyTorch device to place tensor on
        
    Returns:
        Tensor of station indices
    """
    if device is None:
        device = torch.device('cpu')
        
    # Convert station IDs to indices
    station_ids = df['station_id'].to_list()
    station_indices = [station_id_to_idx[sid] for sid in station_ids]
    
    return torch.tensor(station_indices, dtype=torch.long, device=device)


def create_embedding_matrix(station_id_to_idx: Dict[int, int], 
                          embedding_dim: Optional[int] = None,
                          pretrained_path: Optional[str] = None) -> torch.Tensor:
    """
    Create or load pre-trained embedding matrix.
    
    Args:
        station_id_to_idx: Station ID to index mapping
        embedding_dim: Embedding dimension (auto-calculated if None)
        pretrained_path: Path to pre-trained embeddings (optional)
        
    Returns:
        Embedding matrix tensor of shape (n_stations, embedding_dim)
    """
    n_stations = len(station_id_to_idx)
    
    if embedding_dim is None:
        embedding_dim = int(np.sqrt(n_stations))
        
    if pretrained_path and os.path.exists(pretrained_path):
        # Load pre-trained embeddings
        embeddings = torch.load(pretrained_path)
        print(f"Loaded pre-trained embeddings from: {pretrained_path}")
    else:
        # Initialize new embeddings
        embeddings = torch.empty(n_stations, embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        print(f"Initialized new embeddings: {n_stations} stations, {embedding_dim} dimensions")
        
    return embeddings


class EcoBiciModel(nn.Module):
    """
    Example neural network model using station embeddings.
    
    This demonstrates how to incorporate station embeddings into a model
    for predicting bike arrivals.
    """
    
    def __init__(self, n_stations: int, n_features: int, 
                 embedding_dim: Optional[int] = None,
                 hidden_sizes: list = [256, 128, 64],
                 dropout: float = 0.2):
        super().__init__()
        
        if embedding_dim is None:
            embedding_dim = int(np.sqrt(n_stations))
            
        # Station embedding layer
        self.station_embedding = StationEmbedding(n_stations, embedding_dim, dropout)
        
        # Feature processing layers
        input_size = n_features + embedding_dim
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # Output layer (regression for bike arrivals)
        layers.append(nn.Linear(prev_size, 1))
        
        self.feature_net = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor, station_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            features: Tensor of shape (batch_size, n_features) with numerical features
            station_indices: Tensor of shape (batch_size,) with station indices
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Get station embeddings
        station_emb = self.station_embedding(station_indices)
        
        # Concatenate features and embeddings
        combined = torch.cat([features, station_emb], dim=1)
        
        # Pass through feature network
        output = self.feature_net(combined)
        
        return output


def demo_embedding_usage():
    """
    Demonstrate how to use station embeddings in a model.
    """
    print("🚀 STATION EMBEDDING USAGE DEMO")
    print("=" * 40)
    
    try:
        # Load station mapping
        station_id_to_idx, embedding_info = load_station_mapping()
        n_stations = embedding_info['n_stations']
        embedding_dim = embedding_info['embedding_dim']
        
        print(f"✅ Loaded station mapping:")
        print(f"   Stations: {n_stations}")
        print(f"   Embedding dim: {embedding_dim}")
        
        # Create example model
        n_features = 50  # example number of features
        model = EcoBiciModel(n_stations, n_features, embedding_dim)
        
        print(f"\n✅ Created model:")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Station embeddings: {model.station_embedding.embedding.weight.shape}")
        
        # Create dummy data for testing
        batch_size = 32
        dummy_features = torch.randn(batch_size, n_features)
        dummy_station_indices = torch.randint(0, n_stations, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            predictions = model(dummy_features, dummy_station_indices)
            
        print(f"\n✅ Forward pass successful:")
        print(f"   Input features: {dummy_features.shape}")
        print(f"   Station indices: {dummy_station_indices.shape}")
        print(f"   Predictions: {predictions.shape}")
        
        return model, station_id_to_idx
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find embedding files. Run feat_eng.ipynb first.")
        print(f"   Missing: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return None, None


if __name__ == "__main__":
    # Run demo
    demo_embedding_usage() 