import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN
from torch_geometric_temporal.nn.attention import ASTGCN
from typing import Optional

class EcoBiciGNNPredictor(nn.Module):
    """
    Graph Neural Network for predicting bike arrivals at clusters.
    Uses A3T-GCN (Attention Temporal Graph Convolutional Network) architecture.
    """
    
    def __init__(
        self, 
        node_features: int,
        hidden_dim: int = 64,
        out_channels: int = 32,
        periods: int = 1,
        dropout: float = 0.2,
        model_type: str = "a3tgcn"
    ):
        """
        Args:
            node_features: Number of input features per node
            hidden_dim: Hidden dimension size
            out_channels: Output channels for temporal convolution
            periods: Number of periods for attention (used in A3T-GCN)
            dropout: Dropout rate
            model_type: Type of model ("a3tgcn", "tgcn", "astgcn")
        """
        super(EcoBiciGNNPredictor, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.dropout = dropout
        self.model_type = model_type
        
        # select temporal GNN layer
        if model_type == "a3tgcn":
            self.temporal_gnn = A3TGCN(
                in_channels=node_features,
                out_channels=out_channels,
                periods=periods
            )
        elif model_type == "tgcn":
            self.temporal_gnn = TGCN(
                in_channels=node_features,
                out_channels=out_channels
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.linear1 = nn.Linear(out_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        
        # activation
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            
        Returns:
            predictions: [num_nodes, 1] - predicted arrivals for each cluster
        """
        # apply temporal GNN
        h = self.temporal_gnn(x, edge_index, edge_weight)
        
        # apply dropout
        h = self.dropout_layer(h)
        
        # apply output layers
        h = self.linear1(h)
        h = self.relu(h)
        h = self.dropout_layer(h)
        h = self.linear2(h)
        
        return h.squeeze(-1)  # [num_nodes]


class MultiStepGNNPredictor(nn.Module):
    """
    Multi-step ahead prediction GNN model.
    Predicts multiple time steps into the future.
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        out_channels: int = 32,
        prediction_horizon: int = 6,  # predict 6 steps ahead (3 hours)
        dropout: float = 0.2
    ):
        super(MultiStepGNNPredictor, self).__init__()
        
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        
        # temporal GNN layer
        self.temporal_gnn = A3TGCN(
            in_channels=node_features,
            out_channels=out_channels,
            periods=12  # use 12 periods for attention (6 hours)
        )
        
        # separate output heads for each prediction step
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(prediction_horizon)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """
        Forward pass for multi-step prediction
        
        Returns:
            predictions: [num_nodes, prediction_horizon]
        """
        # apply temporal GNN
        h = self.temporal_gnn(x, edge_index, edge_weight)
        h = self.dropout(h)
        
        # apply each output head
        predictions = []
        for head in self.output_heads:
            pred = head(h).squeeze(-1)  # [num_nodes]
            predictions.append(pred)
        
        # stack predictions: [num_nodes, prediction_horizon]
        return torch.stack(predictions, dim=1)


class SequenceToSequenceGNN(nn.Module):
    """
    Sequence-to-sequence GNN for learning from temporal sequences.
    Uses multiple timestamps as input to predict future arrivals.
    """
    
    def __init__(
        self,
        node_features: int,
        sequence_length: int = 12,
        hidden_dim: int = 64,
        out_channels: int = 32,
        dropout: float = 0.2
    ):
        super(SequenceToSequenceGNN, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # encoder: process sequence of inputs
        self.encoder_gnn = A3TGCN(
            in_channels=node_features,
            out_channels=out_channels,
            periods=sequence_length
        )
        
        # lstm to process temporal sequence
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # decoder: generate predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_sequence: list, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """
        Forward pass with sequence input
        
        Args:
            x_sequence: List of tensors [sequence_length, num_nodes, num_features]
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            predictions: [num_nodes] - predicted arrivals
        """
        # process each time step through GNN
        gnn_outputs = []
        for x_t in x_sequence:
            h_t = self.encoder_gnn(x_t, edge_index, edge_weight)
            gnn_outputs.append(h_t)
        
        # stack to create sequence: [sequence_length, num_nodes, out_channels]
        gnn_sequence = torch.stack(gnn_outputs, dim=0)
        
        # transpose for LSTM: [num_nodes, sequence_length, out_channels]
        gnn_sequence = gnn_sequence.transpose(0, 1)
        
        # process through LSTM
        lstm_out, (hidden, cell) = self.lstm(gnn_sequence)
        
        # use last hidden state for prediction
        final_hidden = lstm_out[:, -1, :]  # [num_nodes, hidden_dim]
        
        # decode to predictions
        predictions = self.decoder(final_hidden).squeeze(-1)  # [num_nodes]
        
        return predictions 