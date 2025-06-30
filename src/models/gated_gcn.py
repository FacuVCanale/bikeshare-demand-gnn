"""
Gated Graph Convolutional Neural Network implementation.

This module implements the Gate Graph Convolutional Neural Network (GGCN) 
as described in "Predicting travel demand of a bike sharing system using 
graph convolutional neural networks" (Behroozi & Edrisi, 2024).

The model combines:
1. Graph Convolutional layers for spatial aggregation
2. GRU cells for temporal gating and gradient stabilization
3. Dynamic graph support for time-varying spatial relationships

Key features:
- Prevents data leakage by processing temporal sequences sequentially
- Uses only historical information for each prediction step
- Maintains temporal state through GRU gates between time steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Optional, Union
import warnings


class GatedGCNLayer(nn.Module):
    """
    Single Gated Graph Convolutional Layer.
    
    Combines graph convolution for spatial aggregation with GRU gating
    for temporal modeling and gradient stabilization.
    
    Architecture follows Equations 13-20 from the paper:
    1. Spatial aggregation: m_i = Σ_j A_ji * W * h_j
    2. Temporal gating: h_i = GRU(m_i, h_i_prev)
    """
    
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        bias: bool = True,
        add_self_loops: bool = True,
        normalize: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dropout = dropout
        
        # graph convolution for spatial aggregation
        self.conv = GCNConv(
            in_channels=in_feats,
            out_channels=out_feats,
            improved=False,
            cached=False,  # dynamic graphs require non-cached convolution
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias
        )
        
        # gru cell for temporal gating
        self.gru = nn.GRUCell(
            input_size=out_feats,
            hidden_size=out_feats,
            bias=bias
        )
        
        # layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_feats)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through gated GCN layer.
        
        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Graph connectivity [2, num_edges]
            h_prev: Previous hidden state [num_nodes, out_feats]
                   If None, initializes with zeros
        
        Returns:
            Updated hidden state [num_nodes, out_feats]
        """
        # spatial aggregation via graph convolution
        m = self.conv(x, edge_index)
        
        # apply dropout for regularization
        if self.dropout > 0 and self.training:
            m = F.dropout(m, p=self.dropout, training=self.training)
        
        # initialize hidden state if first time step
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0), self.out_feats,
                dtype=x.dtype, device=x.device
            )
        
        # temporal gating through GRU
        h = self.gru(m, h_prev)
        
        # layer normalization for training stability
        h = self.layer_norm(h)
        
        return h


class GateGCN(nn.Module):
    """
    Gated Graph Convolutional Neural Network for temporal demand prediction.
    
    This model processes temporal sequences of graph-structured data using
    multiple gated GCN layers followed by a prediction head.
    
    The model maintains temporal state across time steps through GRU gates,
    allowing it to capture both spatial dependencies (via graph convolution)
    and temporal dynamics (via gating mechanism).
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_targets: int = 1,
        dropout: float = 0.2,
        use_residual: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.dropout = dropout
        self.use_residual = use_residual
        
        # activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu
            warnings.warn(f"Unknown activation '{activation}', using ReLU")
        
        # build gated GCN layers
        self.layers = nn.ModuleList()
        
        # input layer
        self.layers.append(
            GatedGCNLayer(
                in_feats=num_features,
                out_feats=hidden_dim,
                dropout=dropout
            )
        )
        
        # hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GatedGCNLayer(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    dropout=dropout
                )
            )
        
        # prediction head (following paper architecture)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
        # residual projection if needed
        if use_residual and num_features != hidden_dim:
            self.residual_proj = nn.Linear(num_features, hidden_dim)
        else:
            self.residual_proj = None
    
    def forward(
        self,
        x_seq: torch.Tensor,
        edge_seq: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the temporal gated GCN.
        
        Args:
            x_seq: Temporal sequence of node features 
                   [seq_len, num_nodes, num_features]
            edge_seq: List of edge indices for each time step
                     Length should equal seq_len
        
        Returns:
            Predictions for next time step [num_nodes, num_targets]
        """
        seq_len, num_nodes, _ = x_seq.shape
        
        if len(edge_seq) != seq_len:
            raise ValueError(
                f"Length mismatch: x_seq has {seq_len} time steps, "
                f"but edge_seq has {len(edge_seq)} graphs"
            )
        
        # initialize hidden states for all layers
        hidden_states = [None] * self.num_layers
        
        # process temporal sequence
        for t in range(seq_len):
            x_t = x_seq[t]  # [num_nodes, num_features]
            edge_index_t = edge_seq[t]  # [2, num_edges_t]
            
            # pass through all gated GCN layers
            layer_input = x_t
            for layer_idx, layer in enumerate(self.layers):
                h_new = layer(layer_input, edge_index_t, hidden_states[layer_idx])
                
                # residual connection for first layer if dimensions match
                if (layer_idx == 0 and self.use_residual and 
                    self.residual_proj is not None):
                    residual = self.residual_proj(x_t)
                    h_new = h_new + residual
                elif (layer_idx == 0 and self.use_residual and 
                      x_t.shape[-1] == h_new.shape[-1]):
                    h_new = h_new + x_t
                
                # update hidden state and prepare for next layer
                hidden_states[layer_idx] = h_new
                layer_input = h_new
                
                # apply activation except for last layer
                if layer_idx < len(self.layers) - 1:
                    layer_input = self.activation(layer_input)
        
        # final prediction using last hidden state
        final_hidden = hidden_states[-1]  # [num_nodes, hidden_dim]
        predictions = self.predictor(final_hidden)  # [num_nodes, num_targets]
        
        # squeeze if single target
        if self.num_targets == 1:
            predictions = predictions.squeeze(-1)
        
        return predictions
    
    def reset_hidden_states(self):
        """
        Reset internal hidden states.
        
        This method doesn't do anything for this implementation since
        hidden states are reinitialized for each forward pass, but
        it's provided for interface compatibility.
        """
        pass


def create_gated_gcn(
    num_features: int,
    num_targets: int = 1,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    use_residual: bool = True,
    activation: str = 'relu'
) -> GateGCN:
    """
    Factory function to create a Gated GCN model.
    
    Args:
        num_features: Number of input features per node
        num_targets: Number of target variables to predict
        hidden_dim: Hidden dimension size
        num_layers: Number of gated GCN layers
        dropout: Dropout rate for regularization
        use_residual: Whether to use residual connections
        activation: Activation function name
    
    Returns:
        Configured GateGCN model
    """
    return GateGCN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_targets=num_targets,
        dropout=dropout,
        use_residual=use_residual,
        activation=activation
    )


def count_parameters(model: GateGCN) -> dict:
    """
    Count the number of parameters in a GateGCN model.
    
    Args:
        model: GateGCN model instance
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_params = {}
    for i, layer in enumerate(model.layers):
        layer_params[f'layer_{i}'] = sum(p.numel() for p in layer.parameters())
    
    predictor_params = sum(p.numel() for p in model.predictor.parameters())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'gated_layers': layer_params,
        'predictor_head': predictor_params
    } 