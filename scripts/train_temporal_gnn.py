#!/usr/bin/env python3
"""
Training script for Temporal Graph Neural Networks on EcoBici clustered data.

This script loads temporal sequences with dynamic graphs and trains the Gated GCN model
for bike demand prediction using the methodology from Behroozi & Edrisi (2024).

Features:
- Loads temporal sequences with dynamic adjacency matrices
- Trains Gated GCN model with proper temporal data handling
- Prevents data leakage through strict temporal ordering
- Evaluates on temporal test sequences
- Supports various hyperparameter configurations

Author: EcoBici-AI
"""

import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import json
import argparse
import sys
import pickle
from typing import Dict, Any, List

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.temporal_graph_dataset import create_temporal_splits
from src.training.temporal_gnn_trainer import train_temporal_gnn_experiment
from src.models.gated_gcn import GateGCN


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def load_temporal_sequences(data_dir: str) -> Dict[str, Any]:
    """
    Load temporal sequences created by prepare_temporal_gnn_dataset.py
    
    Args:
        data_dir: Directory containing temporal sequence files
        
    Returns:
        Dictionary containing loaded sequences and metadata
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # load processing configuration
    config_path = data_path / 'temporal_processing_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            processing_config = json.load(f)
    else:
        processing_config = {}
    
    # load temporal sequences for each split
    splits = {}
    for split in ['train', 'val', 'test']:
        seq_file = data_path / f'{split}_temporal_sequences.pkl'
        
        if seq_file.exists():
            print(f"Loading {split} temporal sequences from {seq_file}")
            with open(seq_file, 'rb') as f:
                data = pickle.load(f)
                splits[split] = {
                    'sequences': data['sequences'],
                    'metadata': data['metadata']
                }
            print(f"  Loaded {len(data['sequences'])} sequences")
        else:
            print(f"Warning: {seq_file} not found, skipping {split} split")
    
    return {
        'splits': splits,
        'processing_config': processing_config
    }


def create_datasets_from_sequences(splits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create TemporalGraphDataset objects from loaded sequences.
    
    Args:
        splits: Dictionary containing sequences for each split
        
    Returns:
        Dictionary with dataset objects
    """
    from src.dataset.temporal_graph_dataset import TemporalGraphDataset, TemporalGraphData
    
    datasets = {}
    
    for split_name, split_data in splits.items():
        print(f"Creating {split_name} dataset...")
        
        sequences = split_data['sequences']
        metadata = split_data['metadata']
        
        # convert sequences to TemporalGraphData objects
        temporal_data_list = []
        
        for seq in sequences:
            # convert numpy arrays to tensors
            x_seq = torch.tensor(seq['x_seq'], dtype=torch.float32)
            edge_seq = [torch.tensor(edge_idx, dtype=torch.long) for edge_idx in seq['edge_seq']]
            y = torch.tensor(seq['y'], dtype=torch.float32)
            
            temporal_data = TemporalGraphData(
                x_seq=x_seq,
                edge_seq=edge_seq,
                y=y,
                num_nodes=metadata['num_clusters'],
                timestamps=seq.get('timestamps')
            )
            
            temporal_data_list.append(temporal_data)
        
        # create custom dataset class for pre-loaded sequences
        class PreloadedTemporalDataset:
            def __init__(self, sequences, metadata):
                self.sequences = sequences
                self.metadata = metadata
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return self.sequences[idx]
            
            def get_data_info(self):
                if self.sequences:
                    sample = self.sequences[0]
                    return {
                        'num_sequences': len(self.sequences),
                        'sequence_length': sample.seq_len,
                        'num_nodes': sample.num_nodes,
                        'num_features': sample.num_features,
                        'num_targets': 1 if len(sample.y.shape) == 1 else sample.y.shape[-1],
                        'feature_columns': self.metadata.get('feature_columns', []),
                        'target_columns': self.metadata.get('target_columns', [])
                    }
                return {}
        
        datasets[split_name] = PreloadedTemporalDataset(temporal_data_list, metadata)
        
        info = datasets[split_name].get_data_info()
        print(f"  {split_name} dataset: {info.get('num_sequences', 0)} sequences, "
              f"{info.get('num_features', 0)} features, {info.get('num_nodes', 0)} nodes")
    
    return datasets


def get_model_config() -> Dict[str, Any]:
    """Get default model configuration for Gated GCN following the paper."""
    return {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'use_residual': True,
        'activation': 'relu'
    }


def get_training_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        'optimizer_name': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'scheduler_name': 'plateau',
        'loss_function': 'mse',
        'loss_params': {}
    }


def run_temporal_gnn_experiment(
    datasets: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    fit_params: Dict[str, Any],
    experiment_name: str = "gated_gcn_experiment"
) -> Dict[str, Any]:
    """
    Run a complete Gated GCN training experiment.
    
    Args:
        datasets: Dictionary with train/val/test datasets
        model_config: Model configuration
        training_config: Training configuration
        fit_params: Fitting parameters (epochs, patience, etc.)
        experiment_name: Name for the experiment
        
    Returns:
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running Temporal GNN Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # check required datasets
    required_splits = ['train', 'val', 'test']
    for split in required_splits:
        if split not in datasets:
            raise ValueError(f"Missing {split} dataset")
    
    # print dataset info
    for split in required_splits:
        info = datasets[split].get_data_info()
        print(f"{split.title()} dataset: {info.get('num_sequences', 0)} sequences")
    
    train_info = datasets['train'].get_data_info()
    print(f"Features: {train_info.get('num_features', 0)}")
    print(f"Targets: {train_info.get('num_targets', 0)}")
    print(f"Nodes (clusters): {train_info.get('num_nodes', 0)}")
    
    # create model
    model = GateGCN(
        num_features=train_info['num_features'],
        num_targets=train_info['num_targets'],
        **model_config
    )
    
    print(f"\nModel Architecture:")
    print(f"  Hidden dim: {model_config['hidden_dim']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Dropout: {model_config['dropout']}")
    print(f"  Activation: {model_config['activation']}")
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # run experiment using temporal trainer
    results = train_temporal_gnn_experiment(
        train_dataset=datasets['train'],
        val_dataset=datasets['val'],
        test_dataset=datasets['test'],
        model_config=model_config,
        training_config=training_config,
        experiment_name=experiment_name
    )
    
    # add fit parameters to results
    results['fit_params'] = fit_params
    
    # print final results
    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    training_history = results.get('training_history', {})
    test_metrics = results.get('test_metrics', {})
    
    print(f"Training completed:")
    print(f"  Best validation loss: {training_history.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Epochs trained: {training_history.get('epochs_trained', 'N/A')}")
    
    if test_metrics:
        print(f"\nTest Results:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Temporal GNN (Gated GCN) for EcoBici demand prediction')
    
    # data arguments
    parser.add_argument('--data_dir', type=str, default='data/temporal_gnn',
                        help='Directory containing temporal sequence data')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    
    # model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of Gated GCN layers (default: 2, following paper)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'elu', 'leaky_relu', 'gelu'],
                        help='Activation function (default: relu)')
    parser.add_argument('--use_residual', action='store_true', default=True,
                        help='Use residual connections (default: True)')
    
    # training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for temporal sequences (default: 1)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer (default: adam)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler (default: plateau)')
    parser.add_argument('--loss_function', type=str, default='mse',
                        choices=['mse', 'mae', 'huber'],
                        help='Loss function (default: mse)')
    
    # other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto, default: auto)')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='How often to log progress (default: 10)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save detailed results')
    
    args = parser.parse_args()
    
    # set random seed
    set_seed(args.seed)
    
    # set experiment name
    if args.experiment_name is None:
        args.experiment_name = f"gated_gcn_{args.hidden_dim}h_{args.num_layers}l"
    
    print("Loading temporal sequence data...")
    try:
        loaded_data = load_temporal_sequences(args.data_dir)
        splits = loaded_data['splits']
        processing_config = loaded_data['processing_config']
        
        print(f"Data loaded successfully from {args.data_dir}")
        print(f"Processing config: {processing_config.get('sequence_length', 'unknown')} steps, "
              f"{processing_config.get('prediction_horizon', 'unknown')} horizon")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Make sure you have run the temporal data preparation script first:")
        print("  python scripts/prepare_temporal_gnn_dataset.py")
        return
    
    # create datasets
    print("\nCreating temporal datasets...")
    datasets = create_datasets_from_sequences(splits)
    
    # prepare configurations
    model_config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'use_residual': args.use_residual,
        'activation': args.activation
    }
    
    training_config = {
        'optimizer_name': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler_name': args.scheduler,
        'loss_function': args.loss_function,
        'loss_params': {}
    }
    
    fit_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stopping_patience': args.patience,
        'save_best_model': True,
        'log_frequency': args.log_frequency,
        'min_improvement': 0.001
    }
    
    # run experiment
    try:
        results = run_temporal_gnn_experiment(
            datasets=datasets,
            model_config=model_config,
            training_config=training_config,
            fit_params=fit_params,
            experiment_name=args.experiment_name
        )
        
        # save results if requested
        if args.save_results:
            results_dir = Path('experiments/temporal_gnn/results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # save configuration and metrics
            results_summary = {
                'experiment_name': args.experiment_name,
                'model_config': model_config,
                'training_config': training_config,
                'fit_params': fit_params,
                'processing_config': processing_config,
                'test_metrics': results.get('test_metrics', {}),
                'training_history': {
                    'best_val_loss': results['training_history'].get('best_val_loss'),
                    'epochs_trained': results['training_history'].get('epochs_trained'),
                    'final_val_metrics': results['training_history'].get('final_val_metrics', {})
                }
            }
            
            results_file = results_dir / f'{args.experiment_name}_results.json'
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\nResults saved to {results_file}")
        
        print(f"\n{'='*80}")
        print(f"TEMPORAL GNN TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Experiment: {args.experiment_name}")
        if results.get('test_metrics'):
            test_r2 = results['test_metrics'].get('r2', 0)
            test_rmse = results['test_metrics'].get('rmse', 0)
            print(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main() 