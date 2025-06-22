#!/usr/bin/env python3
"""
Enhanced Graph Neural Network Training Script for EcoBici Demand Prediction

This script provides a comprehensive interface for training GNN models with multi-GPU support
on EcoBici demand prediction data. It includes various model architectures, extensive 
hyperparameter configuration, data handling, and experiment tracking.

Multi-GPU support is automatically enabled when multiple GPUs are available.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.gnn_models import create_gnn_model
from src.training.gnn_trainer import GNNTrainer, get_device_info


def set_seed(seed: int = 42):
    """Set seeds for reproducibility across all libraries and operations"""
    
    # python random
    random.seed(seed)
    
    # numpy
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # additional environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # use deterministic algorithms when available
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Set global seed to {seed} for reproducibility")


def load_gnn_data(data_dir: str) -> Dict[str, Any]:
    """
    Load preprocessed GNN data.
    
    Args:
        data_dir: Directory containing GNN data files
        
    Returns:
        Dictionary containing train/val/test data and metadata
    """
    data_path = Path(data_dir)
    
    # load data
    train_data = torch.load(data_path / 'train_data.pt')
    val_data = torch.load(data_path / 'val_data.pt')
    test_data = torch.load(data_path / 'test_data.pt')
    
    # load metadata
    with open(data_path / 'train_feature_names.json', 'r') as f:
        train_metadata = json.load(f)
    
    with open(data_path / 'processing_config.json', 'r') as f:
        processing_config = json.load(f)
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'feature_names': train_metadata['features'],
        'target_names': train_metadata['targets'],
        'n_features': train_metadata['n_features'],
        'n_targets': train_metadata['n_targets'],
        'processing_config': processing_config
    }


def get_default_model_config(model_type: str, num_features: int, num_targets: int) -> Dict[str, Any]:
    """Get default configuration for a specific model type"""
    
    base_config = {
        'num_features': num_features,
        'num_targets': num_targets,
        'hidden_dim': 64,
        'dropout': 0.2,
    }
    
    if model_type == 'gcn':
        base_config.update({
            'num_layers': 3,
            'use_batch_norm': True,
        })
    elif model_type == 'gat':
        base_config.update({
            'num_layers': 2,
            'num_heads': 4,
            'use_batch_norm': True,
            'attention_dropout': 0.1,
        })
    elif model_type == 'sage':
        base_config.update({
            'num_layers': 3,
            'aggregation': 'mean',
            'use_batch_norm': True,
        })
    elif model_type == 'transformer':
        base_config.update({
            'num_layers': 2,
            'num_heads': 8,
            'use_batch_norm': True,
        })
    elif model_type == 'hybrid':
        base_config.update({
            'use_temporal': True,
            'temporal_dim': 64,
        })
    
    return base_config


def create_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create comprehensive training configuration from arguments"""
    
    config = {
        'trainer_params': {
            'device': args.device,
            'save_dir': args.output_dir
        },
        'training_params': {
            'optimizer_name': args.optimizer,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'scheduler_name': args.scheduler,
            'loss_function': args.loss_function,
        },
        'fit_params': {
            'epochs': args.epochs,
            'early_stopping_patience': args.early_stopping_patience,
            'save_best_model': args.save_best_model,
            'save_checkpoints': args.save_checkpoints,
            'checkpoint_frequency': args.checkpoint_frequency,
            'verbose': args.verbose
        },
        'model_params': {
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
        }
    }
    
    # add model-specific parameters
    if args.model_type == 'gcn':
        config['model_params'].update({
            'num_layers': args.num_layers,
            'use_batch_norm': args.use_batch_norm,
        })
    elif args.model_type == 'gat':
        config['model_params'].update({
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'use_batch_norm': args.use_batch_norm,
            'attention_dropout': args.attention_dropout,
        })
    elif args.model_type == 'sage':
        config['model_params'].update({
            'num_layers': args.num_layers,
            'aggregation': args.aggregation,
            'use_batch_norm': args.use_batch_norm,
        })
    elif args.model_type == 'transformer':
        config['model_params'].update({
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'use_batch_norm': args.use_batch_norm,
        })
    elif args.model_type == 'hybrid':
        config['model_params'].update({
            'use_temporal': args.use_temporal,
            'temporal_dim': args.temporal_dim,
        })
    
    return config


def load_and_prepare_data(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    """Load and prepare GNN datasets"""
    
    print(f"\nLoading GNN dataset from: {args.data_path}")
    
    # load preprocessed data directly
    data = load_gnn_data(args.data_path)
    
    print(f"Dataset loaded successfully:")
    print(f"  Training samples: {data['train_data'].x.shape[0]}")
    print(f"  Validation samples: {data['val_data'].x.shape[0]}")
    print(f"  Test samples: {data['test_data'].x.shape[0]}")
    print(f"  Features per node: {data['n_features']}")
    print(f"  Edges in training graph: {data['train_data'].edge_index.shape[1]}")
    
    return data['train_data'], data['val_data'], data['test_data']


def display_device_info(device_spec: str):
    """Display comprehensive device information"""
    
    device_info = get_device_info()
    
    print(f"\n{'='*60}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Requested device: {device_spec}")
    print(f"CUDA available: {device_info['cuda_available']}")
    print(f"Number of GPUs: {device_info['num_gpus']}")
    
    if device_info['cuda_available']:
        print(f"Primary device: {device_info['primary_device']}")
        
        print(f"\nGPU Information:")
        for i in range(device_info['num_gpus']):
            name = device_info['gpu_names'][i]
            memory_gb = device_info['total_memory'][i] / (1024**3)
            print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
        
        if device_info['use_multi_gpu']:
            print(f"\n🚀 Multi-GPU training will be enabled!")
            print(f"   Using {device_info['num_gpus']} GPUs for accelerated training")
        else:
            print(f"\n⚡ Single GPU training enabled")
    else:
        print(f"🐌 CPU training mode")
    
    print(f"{'='*60}\n")


def main():
    """Main training function with multi-GPU support"""
    
    parser = argparse.ArgumentParser(
        description='Train GNN models for EcoBici demand prediction with multi-GPU support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model-type', type=str, default='gcn',
                            choices=['gcn', 'gat', 'sage', 'transformer', 'hybrid'],
                            help='Type of GNN model to train')
    model_group.add_argument('--hidden-dim', type=int, default=64,
                            help='Hidden dimension size')
    model_group.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout rate')
    model_group.add_argument('--num-layers', type=int, default=3,
                            help='Number of GNN layers (gcn, gat, sage, transformer)')
    model_group.add_argument('--num-heads', type=int, default=4,
                            help='Number of attention heads (gat, transformer)')
    model_group.add_argument('--attention-dropout', type=float, default=0.1,
                            help='Attention dropout rate (gat)')
    model_group.add_argument('--aggregation', type=str, default='mean',
                            choices=['mean', 'max', 'sum'],
                            help='Aggregation type for GraphSAGE')
    model_group.add_argument('--use-batch-norm', action='store_true', default=True,
                            help='Use batch normalization')
    model_group.add_argument('--use-temporal', action='store_true', default=True,
                            help='Use temporal components (hybrid model)')
    model_group.add_argument('--temporal-dim', type=int, default=64,
                            help='Temporal dimension (hybrid model)')
    
    # data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-path', type=str, default='data/93k_gnn_ready',
                           help='Path to processed GNN data')
    data_group.add_argument('--spatial-threshold', type=float, default=1000.0,
                           help='Spatial threshold for graph edges (meters)')
    data_group.add_argument('--temporal-window', type=int, default=24,
                           help='Temporal window size (hours)')
    data_group.add_argument('--target-column', type=str, default='demand',
                           help='Target column for prediction')
    data_group.add_argument('--val-split', type=float, default=0.2,
                           help='Validation split ratio')
    data_group.add_argument('--test-split', type=float, default=0.1,
                           help='Test split ratio')
    
    # training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_group.add_argument('--learning-rate', type=float, default=0.001,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=1e-5,
                            help='Weight decay for regularization')
    train_group.add_argument('--optimizer', type=str, default='adam',
                            choices=['adam', 'adamw', 'sgd'],
                            help='Optimizer type')
    train_group.add_argument('--scheduler', type=str, default='plateau',
                            choices=['plateau', 'cosine', 'none'],
                            help='Learning rate scheduler')
    train_group.add_argument('--loss-function', type=str, default='mse',
                            choices=['mse', 'mae', 'huber'],
                            help='Loss function')
    train_group.add_argument('--early-stopping-patience', type=int, default=15,
                            help='Early stopping patience')
    
    # system configuration
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', type=str, default='auto',
                             choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'multi-gpu'],
                             help='Device to use (auto/cpu/cuda/multi-gpu)')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    
    # experiment configuration
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument('--experiment-name', type=str, default=None,
                          help='Name for this experiment')
    exp_group.add_argument('--output-dir', type=str, default='experiments',
                          help='Directory to save results')
    exp_group.add_argument('--save-best-model', action='store_true', default=True,
                          help='Save best model during training')
    exp_group.add_argument('--save-checkpoints', action='store_true', default=True,
                          help='Save periodic checkpoints')
    exp_group.add_argument('--checkpoint-frequency', type=int, default=10,
                          help='Checkpoint saving frequency (epochs)')
    exp_group.add_argument('--verbose', action='store_true', default=True,
                          help='Verbose training output')
    exp_group.add_argument('--config-file', type=str, default=None,
                          help='JSON config file to load parameters from')
    
    args = parser.parse_args()
    
    # load configuration from file if provided
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
        
        # update args with config file values
        for key, value in config_data.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
        print(f"Loaded configuration from: {args.config_file}")
    
    # set reproducibility seed
    set_seed(args.seed)
    
    # display device information
    display_device_info(args.device)
    
    # create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_experiment_{args.seed}"
    
    print(f"Starting GNN training experiment: {args.experiment_name}")
    print(f"Model type: {args.model_type}")
    print(f"Device configuration: {args.device}")
    
    # load and prepare data
    train_data, val_data, test_data = load_and_prepare_data(args)
    
    # create model configuration
    num_features = train_data.x.shape[1]
    num_targets = train_data.y.shape[1] if len(train_data.y.shape) > 1 else 1
    
    # create model
    print(f"\nCreating {args.model_type.upper()} model...")
    model_config = get_default_model_config(args.model_type, num_features, num_targets)
    
    # update with custom parameters
    training_config = create_training_config(args)
    model_config.update(training_config['model_params'])
    
    model = create_gnn_model(
        model_type=args.model_type,
        **model_config
    )
    
    print(f"Model created: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # create trainer with multi-GPU support
    print(f"\nInitializing trainer with multi-GPU support...")
    trainer = GNNTrainer(
        model=model,
        device=args.device,
        experiment_name=args.experiment_name,
        save_dir=args.output_dir
    )
    
    # setup training components
    trainer.setup_training(**training_config['training_params'])
    
    # save configuration
    config_path = trainer.save_dir / 'config.json'
    config_to_save = {
        'model_type': args.model_type,
        'model_config': model_config,
        'training_config': training_config,
        'data_config': {
            'data_path': args.data_path,
            'spatial_threshold': args.spatial_threshold,
            'temporal_window': args.temporal_window,
            'target_column': args.target_column,
            'val_split': args.val_split,
            'test_split': args.test_split,
        },
        'experiment_config': {
            'seed': args.seed,
            'experiment_name': args.experiment_name,
        },
        'device_info': trainer.device_info,
        'multi_gpu_enabled': trainer.use_multi_gpu
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    # train model
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        **training_config['fit_params']
    )
    
    # evaluate on test set
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION")
    print(f"{'='*60}")
    
    test_metrics = trainer.evaluate(test_data)
    
    # save final results
    results = {
        'model_type': args.model_type,
        'experiment_name': args.experiment_name,
        'config': config_to_save,
        'training_history': history,
        'test_metrics': test_metrics,
        'device_info': trainer.device_info,
        'multi_gpu_used': trainer.use_multi_gpu
    }
    
    results_path = trainer.save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"Best validation R²: {max([m.get('r2', 0) for m in history['val_metrics']]):.4f}")
    print(f"Final test R²: {test_metrics.get('r2', 0):.4f}")
    
    if trainer.use_multi_gpu:
        print(f"Multi-GPU training was used with {trainer.device_info['num_gpus']} GPUs")
    
    return trainer, results


if __name__ == '__main__':
    trainer, results = main() 