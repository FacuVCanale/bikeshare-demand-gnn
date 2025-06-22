"""
Model loading utilities for EcoBici GNN experiments.

This module provides functions to load trained models from experiment directories,
including both the model weights and architecture information.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .gnn_models import create_gnn_model


def load_model_from_experiment(
    experiment_path: str,
    model_filename: str = 'final_model.pt',
    device: str = 'auto'
) -> torch.nn.Module:
    """
    Load a trained model from an experiment directory.
    
    Args:
        experiment_path: Path to the experiment directory
        model_filename: Name of the model file (default: 'final_model.pt')
        device: Device to load model on ('cuda', 'cpu', or 'auto')
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If model file or architecture file not found
        ValueError: If model cannot be reconstructed
    """
    experiment_path = Path(experiment_path)
    model_path = experiment_path / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # extract model information
    model_class = checkpoint['model_class']
    model_init_params = checkpoint.get('model_init_params', {})
    
    # filter out None values from initialization parameters
    model_init_params = {k: v for k, v in model_init_params.items() if v is not None}
    
    # determine model type from class name
    model_type_mapping = {
        'TemporalGCN': 'gcn',
        'SpatialGAT': 'gat',
        'GraphSAGE': 'sage',
        'GraphTransformer': 'transformer',
        'HybridSpatioTemporalGNN': 'hybrid'
    }
    
    model_type = model_type_mapping.get(model_class)
    if not model_type:
        raise ValueError(f"Unknown model class: {model_class}")
    
    # create model with saved parameters
    model = create_gnn_model(
        model_type=model_type,
        **model_init_params
    )
    
    # load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_model_architecture(experiment_path: str, filename: str = 'model_architecture.json') -> Dict[str, Any]:
    """
    Load model architecture information from an experiment directory.
    
    Args:
        experiment_path: Path to the experiment directory
        filename: Name of the architecture file (default: 'model_architecture.json')
        
    Returns:
        Dictionary containing model architecture information
        
    Raises:
        FileNotFoundError: If architecture file not found
    """
    experiment_path = Path(experiment_path)
    arch_path = experiment_path / filename
    
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture file not found: {arch_path}")
    
    with open(arch_path, 'r') as f:
        architecture_info = json.load(f)
    
    return architecture_info


def load_training_history(experiment_path: str, filename: str = 'training_history.json') -> Dict[str, Any]:
    """
    Load training history from an experiment directory.
    
    Args:
        experiment_path: Path to the experiment directory
        filename: Name of the history file (default: 'training_history.json')
        
    Returns:
        Dictionary containing training history
        
    Raises:
        FileNotFoundError: If history file not found
    """
    experiment_path = Path(experiment_path)
    history_path = experiment_path / filename
    
    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def get_experiment_summary(experiment_path: str) -> Dict[str, Any]:
    """
    Get a comprehensive summary of an experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        Dictionary containing experiment summary
    """
    experiment_path = Path(experiment_path)
    
    summary = {
        'experiment_path': str(experiment_path),
        'experiment_name': experiment_path.name,
        'files_present': []
    }
    
    # check for standard files
    standard_files = [
        'final_model.pt',
        'best_model.pt',
        'model_architecture.json',
        'training_history.json',
        'training.log'
    ]
    
    for file in standard_files:
        file_path = experiment_path / file
        if file_path.exists():
            summary['files_present'].append({
                'filename': file,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'exists': True
            })
        else:
            summary['files_present'].append({
                'filename': file,
                'size_mb': 0,
                'exists': False
            })
    
    # load architecture if available
    try:
        architecture = load_model_architecture(experiment_path)
        summary['model_info'] = {
            'model_class': architecture['model_class'],
            'total_parameters': architecture['total_parameters'],
            'trainable_parameters': architecture['trainable_parameters'],
            'initialization_parameters': architecture['initialization_parameters']
        }
    except FileNotFoundError:
        summary['model_info'] = None
    
    # load final training metrics if available
    try:
        history = load_training_history(experiment_path)
        if history.get('val_loss'):
            summary['final_metrics'] = {
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
                'total_epochs': len(history['train_loss']) if history['train_loss'] else None
            }
        else:
            summary['final_metrics'] = None
    except FileNotFoundError:
        summary['final_metrics'] = None
    
    return summary


def recreate_model_from_architecture(
    architecture_path: str,
    device: str = 'auto'
) -> torch.nn.Module:
    """
    Recreate a model from its architecture file (without loading weights).
    
    Args:
        architecture_path: Path to the model_architecture.json file
        device: Device to create model on ('cuda', 'cpu', or 'auto')
        
    Returns:
        Newly created model with same architecture
        
    Raises:
        FileNotFoundError: If architecture file not found
        ValueError: If model cannot be reconstructed
    """
    architecture_path = Path(architecture_path)
    
    if not architecture_path.exists():
        raise FileNotFoundError(f"Architecture file not found: {architecture_path}")
    
    # set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # load architecture
    with open(architecture_path, 'r') as f:
        architecture_info = json.load(f)
    
    # extract model information
    model_class = architecture_info['model_class']
    init_params = architecture_info['initialization_parameters']
    
    # filter out None values
    init_params = {k: v for k, v in init_params.items() if v is not None}
    
    # determine model type
    model_type_mapping = {
        'TemporalGCN': 'gcn',
        'SpatialGAT': 'gat',
        'GraphSAGE': 'sage',
        'GraphTransformer': 'transformer',
        'HybridSpatioTemporalGNN': 'hybrid'
    }
    
    model_type = model_type_mapping.get(model_class)
    if not model_type:
        raise ValueError(f"Unknown model class: {model_class}")
    
    # create model
    model = create_gnn_model(
        model_type=model_type,
        **init_params
    )
    
    model.to(device)
    
    return model 