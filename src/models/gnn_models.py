"""
Graph Neural Network models for EcoBici demand prediction.

This module implements various GNN architectures for predicting bike station demand
using spatial and temporal patterns in the clustered bike-sharing data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GraphConv, SAGEConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm
)
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from abc import ABC, abstractmethod


class BaseGNNModel(nn.Module, ABC):
    """
    Abstract base class for Graph Neural Network models.
    
    Provides a common interface for GNN models with PyTorch Geometric compatibility.
    """
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Model predictions
        """
        pass
    
    def fit(self, train_data: Data, val_data: Optional[Data] = None, **kwargs) -> Dict[str, Any]:
        """
        Training implementation - handled by external trainer.
        
        This method is implemented by external training modules like GNNTrainer.
        """
        raise NotImplementedError("Training logic should be implemented in training module")
    
    def predict(self, data: Data) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data
            
        Returns:
            Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(data)
            return predictions.cpu().numpy()


class TemporalGCN(BaseGNNModel):
    """
    Graph Convolutional Network with temporal features for bike demand prediction.
    
    Uses GCN layers to capture spatial dependencies between clusters and
    processes temporal sequences through the graph structure.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_targets: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
        
        # build network layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # input layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GCN.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, num_features]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment (optional)
        
        Returns:
            Predictions [num_nodes, num_targets] or [batch_size, num_targets]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # no activation after last conv
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # final prediction
        x = self.predictor(x)
        
        return x


class SpatialGAT(BaseGNNModel):
    """
    Graph Attention Network for capturing spatial relationships with attention mechanism.
    
    Uses attention to focus on the most relevant neighboring clusters for prediction.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_targets: int = 1,
        num_heads: int = 8,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.num_heads = num_heads
        self.dropout = dropout
        
        # build network layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # input layer
        self.convs.append(
            GATConv(
                num_features, 
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=attention_dropout,
                concat=True
            )
        )
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=True
                )
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # output layer (single head)
        self.convs.append(
            GATConv(
                hidden_dim,
                hidden_dim,
                heads=1,
                dropout=attention_dropout,
                concat=False
            )
        )
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the GAT"""
        x, edge_index = data.x, data.edge_index
        
        # apply graph attention layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # no activation after last conv
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # final prediction
        x = self.predictor(x)
        
        return x


class GraphSAGE(BaseGNNModel):
    """
    GraphSAGE model for inductive learning on bike demand prediction.
    
    Samples and aggregates features from node neighborhoods, suitable for
    scenarios where new clusters might be added.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_targets: int = 1,
        dropout: float = 0.2,
        aggregation: str = 'mean',
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.dropout = dropout
        
        # build network layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # input layer
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggregation))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # output layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through GraphSAGE"""
        x, edge_index = data.x, data.edge_index
        
        # apply SAGE convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # no activation after last conv
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # final prediction
        x = self.predictor(x)
        
        return x


class GraphTransformer(BaseGNNModel):
    """
    Graph Transformer using TransformerConv layers for bike demand prediction.
    
    Applies transformer-style attention mechanisms to graph-structured data,
    potentially capturing complex spatial-temporal dependencies.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_targets: int = 1,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.num_heads = num_heads
        self.dropout = dropout
        
        # ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # build network layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # input layer
        self.convs.append(
            TransformerConv(
                num_features,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False
            )
        )
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False
                )
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # output layer
        self.convs.append(
            TransformerConv(
                hidden_dim,
                hidden_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        )
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through Graph Transformer"""
        x, edge_index = data.x, data.edge_index
        
        # apply transformer convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # no activation after last conv
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # final prediction
        x = self.predictor(x)
        
        return x


class MemoryEfficientGraphTransformer(BaseGNNModel):
    """
    Memory-efficient Graph Transformer using optimization techniques.
    
    Features:
    - Gradient checkpointing for reduced memory usage
    - Optimized attention computation
    - Reduced intermediate tensor storage
    - Memory monitoring and cleanup
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,  # reduced default from 128
        num_layers: int = 2,   # reduced default from 3
        num_targets: int = 1,
        num_heads: int = 4,    # reduced default from 8
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_gradient_checkpointing: bool = True,
        max_attention_size: int = 1000  # limit attention computation
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_attention_size = max_attention_size
        
        # ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # build network layers with reduced size
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # input layer
        self.convs.append(
            TransformerConv(
                num_features,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False,
                edge_dim=None,  # no edge features to save memory
                bias=False      # no bias to save parameters
            )
        )
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                    edge_dim=None,
                    bias=False
                )
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # output layer (single head for efficiency)
        if num_layers > 1:
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False,
                    edge_dim=None,
                    bias=False
                )
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # simplified prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),  # reduce intermediate size
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_targets)
        )
        
    def _checkpoint_conv_layer(self, conv_layer, batch_norm, x, edge_index, layer_idx):
        """Apply convolution with optional checkpointing"""
        if batch_norm is not None:
            def conv_bn(x_in, edge_idx):
                out = conv_layer(x_in, edge_idx)
                out = batch_norm(out)
                return out
            
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    conv_bn, x, edge_index, use_reentrant=False
                )
            else:
                x = conv_bn(x, edge_index)
        else:
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    conv_layer, x, edge_index, use_reentrant=False
                )
            else:
                x = conv_layer(x, edge_index)
        
        return x
        
    def forward(self, data: Data) -> torch.Tensor:
        """Memory-efficient forward pass"""
        x, edge_index = data.x, data.edge_index
        
        # limit graph size if too large for memory
        num_nodes = x.size(0)
        if num_nodes > self.max_attention_size:
            # sample nodes to fit memory constraints
            import warnings
            warnings.warn(f"Graph has {num_nodes} nodes, sampling {self.max_attention_size} for memory efficiency")
            
            # simple random sampling - can be improved with more sophisticated methods
            indices = torch.randperm(num_nodes)[:self.max_attention_size]
            x = x[indices]
            
            # update edge_index to match sampled nodes
            mask = torch.isin(edge_index, indices)
            mask = mask[0] & mask[1]  # both source and target must be in sampled nodes
            edge_index = edge_index[:, mask]
            
            # remap edge indices to new node numbering
            for i, idx in enumerate(indices):
                edge_index[edge_index == idx] = i
        
        # apply transformer convolutions with checkpointing
        for i, conv in enumerate(self.convs):
            batch_norm = self.batch_norms[i] if self.batch_norms else None
            
            x = self._checkpoint_conv_layer(conv, batch_norm, x, edge_index, i)
            
            # apply activation and dropout (except for last layer)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            # periodic memory cleanup
            if i % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # final prediction
        output = self.predictor(x)
        
        # if we sampled nodes, need to expand output back to original size
        if num_nodes > self.max_attention_size:
            # pad output to match original graph size
            full_output = torch.zeros(num_nodes, output.size(1), 
                                    device=output.device, dtype=output.dtype)
            full_output[indices] = output
            output = full_output
        
        return output
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.use_gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for speed"""
        self.use_gradient_checkpointing = False


class HybridSpatioTemporalGNN(BaseGNNModel):
    """
    Hybrid model combining multiple GNN architectures for comprehensive
    spatio-temporal modeling of bike demand.
    
    Combines GCN for base spatial modeling, GAT for attention-based relationships,
    and temporal processing for time-series patterns.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_targets: int = 1,
        dropout: float = 0.2,
        use_temporal: bool = True,
        temporal_dim: int = 64
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets
        self.dropout = dropout
        self.use_temporal = use_temporal
        
        # spatial processing branch (GCN + GAT)
        self.spatial_gcn = nn.ModuleList([
            GCNConv(num_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        
        self.spatial_gat = nn.ModuleList([
            GATConv(num_features, hidden_dim // 4, heads=4, concat=True),
            GATConv(hidden_dim, hidden_dim // 2, heads=2, concat=True)
        ])
        
        # temporal processing branch (if enabled)
        if use_temporal:
            self.temporal_lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=temporal_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )
            final_dim = hidden_dim + temporal_dim
        else:
            final_dim = hidden_dim
        
        # fusion and prediction layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # combine GCN + GAT
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
        # batch normalization
        self.bn_gcn = nn.ModuleList([BatchNorm(hidden_dim), BatchNorm(hidden_dim)])
        self.bn_gat = nn.ModuleList([BatchNorm(hidden_dim), BatchNorm(hidden_dim)])
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through hybrid model"""
        x, edge_index = data.x, data.edge_index
        
        # gcn branch
        x_gcn = x
        for i, (conv, bn) in enumerate(zip(self.spatial_gcn, self.bn_gcn)):
            x_gcn = conv(x_gcn, edge_index)
            x_gcn = bn(x_gcn)
            if i < len(self.spatial_gcn) - 1:
                x_gcn = F.relu(x_gcn)
                x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)
        
        # gat branch
        x_gat = x
        for i, (conv, bn) in enumerate(zip(self.spatial_gat, self.bn_gat)):
            x_gat = conv(x_gat, edge_index)
            x_gat = bn(x_gat)
            if i < len(self.spatial_gat) - 1:
                x_gat = F.elu(x_gat)
                x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        
        # fuse spatial representations
        x_spatial = self.fusion(torch.cat([x_gcn, x_gat], dim=1))
        
        # temporal processing (if enabled)
        if self.use_temporal:
            # reshape for LSTM: [batch_size, seq_len, features]
            # note: this assumes temporal sequences are provided in the input
            batch_size = x_spatial.size(0)
            x_temporal = x_spatial.unsqueeze(1)  # add sequence dimension
            
            # apply LSTM
            lstm_out, _ = self.temporal_lstm(x_temporal)
            x_temporal = lstm_out[:, -1, :]  # take last output
            
            # combine spatial and temporal
            x_final = torch.cat([x_spatial, x_temporal], dim=1)
        else:
            x_final = x_spatial
        
        # final prediction
        predictions = self.predictor(x_final)
        
        return predictions


# model factory function
def create_gnn_model(
    model_type: str,
    num_features: int,
    num_targets: int = 1,
    memory_efficient: bool = True,
    **kwargs
) -> BaseGNNModel:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: Type of GNN model ('gcn', 'gat', 'sage', 'transformer', 'hybrid')
        num_features: Number of input features
        num_targets: Number of target variables
        memory_efficient: Use memory-efficient versions when available
        **kwargs: Additional model-specific parameters
    
    Returns:
        Initialized GNN model
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return TemporalGCN(num_features=num_features, num_targets=num_targets, **kwargs)
    elif model_type == 'gat':
        return SpatialGAT(num_features=num_features, num_targets=num_targets, **kwargs)
    elif model_type == 'sage':
        return GraphSAGE(num_features=num_features, num_targets=num_targets, **kwargs)
    elif model_type == 'transformer':
        if memory_efficient:
            return MemoryEfficientGraphTransformer(num_features=num_features, num_targets=num_targets, **kwargs)
        else:
        return GraphTransformer(num_features=num_features, num_targets=num_targets, **kwargs)
    elif model_type == 'memory-efficient-transformer':
        return MemoryEfficientGraphTransformer(num_features=num_features, num_targets=num_targets, **kwargs)
    elif model_type == 'hybrid':
        return HybridSpatioTemporalGNN(num_features=num_features, num_targets=num_targets, **kwargs)
    else:
        available_types = "'gcn', 'gat', 'sage', 'transformer', 'memory-efficient-transformer', 'hybrid'"
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {available_types}")


# utility functions for model evaluation
def calculate_gnn_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate evaluation metrics for GNN predictions.
    
    Args:
        predictions: Model predictions [num_samples, num_targets]
        targets: Ground truth targets [num_samples, num_targets]
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # ensure same device
        predictions = predictions.to(targets.device)
        
        # mean squared error
        mse = F.mse_loss(predictions, targets).item()
        
        # mean absolute error
        mae = F.l1_loss(predictions, targets).item()
        
        # root mean squared error
        rmse = torch.sqrt(F.mse_loss(predictions, targets)).item()
        
        # r-squared (coefficient of determination)
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = (1 - ss_res / ss_tot).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> None:
    """
    Print a summary of the GNN model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input features (num_nodes, num_features)
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # print model architecture
    print("\nModel Architecture:")
    print(model) 