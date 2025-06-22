"""
Script simple para probar modelos GNN entrenados en el conjunto de test.

Usa exactamente la misma lógica de carga de datos y evaluación que el script de entrenamiento.
Includes multi-GPU support for consistency with training scripts.

Uso:
    python scripts/test_model.py --model_path final_model.pt
    python scripts/test_model.py --model_path final_model.pt --data_dir data/gnn_ready
    python scripts/test_model.py --model_path final_model.pt --model_type gat
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_models import create_gnn_model, calculate_gnn_metrics
from src.training.gnn_trainer import get_device_info, setup_device


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
            print(f"\n🚀 Multi-GPU inference available!")
            print(f"   {device_info['num_gpus']} GPUs detected for accelerated inference")
        else:
            print(f"\n⚡ Single GPU inference enabled")
    else:
        print(f"🐌 CPU inference mode")
    
    print(f"{'='*60}\n")


def load_gnn_data(data_dir: str):
    """
    Load preprocessed GNN data - same function as train_gnn.py
    
    Args:
        data_dir: Directory containing GNN data files
        
    Returns:
        Dictionary containing train/val/test data and metadata
    """
    data_path = Path(data_dir)
    
    # load data
    train_data = torch.load(data_path / 'train_data.pt', weights_only=False)
    val_data = torch.load(data_path / 'val_data.pt', weights_only=False)
    test_data = torch.load(data_path / 'test_data.pt', weights_only=False)
    
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


def load_model_from_checkpoint(
    model_path: str,
    num_features: int,
    num_targets: int,
    model_type: str = None,
    device: str = 'auto'
):
    """
    Load a trained GNN model from checkpoint with multi-GPU support.
    
    Args:
        model_path: Path to the saved model (.pt file)
        num_features: Number of input features
        num_targets: Number of target variables
        model_type: Type of model ('gat', 'gcn', etc.). If None, tries to infer from checkpoint
        device: Device specification ('auto', 'cpu', 'cuda', 'multi-gpu')
        
    Returns:
        Tuple of (loaded_model, use_multi_gpu, device_info)
    """
    # setup device configuration
    primary_device, use_multi_gpu, device_info = setup_device(device)
    
    print(f"Loading model from: {model_path}")
    
    # load checkpoint
    checkpoint = torch.load(model_path, map_location=primary_device, weights_only=False)
    
    # get model info from checkpoint if available
    if 'model_class' in checkpoint:
        model_class = checkpoint['model_class']
        print(f"Model class from checkpoint: {model_class}")
        
        # map class names to model types
        class_to_type = {
            'TemporalGCN': 'gcn',
            'SpatialGAT': 'gat', 
            'GraphSAGE': 'sage',
            'GraphTransformer': 'transformer',
            'HybridSpatioTemporalGNN': 'hybrid'
        }
        
        if model_type is None:
            model_type = class_to_type.get(model_class, 'gat')  # default to gat
    
    if model_type is None:
        raise ValueError("Could not determine model type. Please specify --model_type")
    
    print(f"Creating {model_type.upper()} model with {num_features} features, {num_targets} targets")
    
    # check if model was trained with multi-GPU
    multi_gpu_trained = checkpoint.get('multi_gpu_trained', False)
    if multi_gpu_trained:
        print("Model was trained with multi-GPU")
    
    # get default model configurations (same as train_gnn.py)
    model_configs = {
        'gcn': {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'use_batch_norm': True,
            'activation': 'relu'
        },
        'gat': {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.2,
            'attention_dropout': 0.1,
            'use_batch_norm': True
        },
        'sage': {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'aggregation': 'mean',
            'use_batch_norm': True
        },
        'transformer': {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.2,
            'use_batch_norm': True
        },
        'hybrid': {
            'hidden_dim': 128,
            'dropout': 0.2,
            'use_temporal': True,
            'temporal_dim': 64
        }
    }
    
    config = model_configs.get(model_type, {})
    
    # create model
    model = create_gnn_model(
        model_type=model_type,
        num_features=num_features,
        num_targets=num_targets,
        **config
    )
    
    # load state dict (handle both single-GPU and multi-GPU trained models)
    state_dict = checkpoint['model_state_dict']
    
    # if model was trained with DataParallel but we're loading for single GPU,
    # remove 'module.' prefix from keys
    if multi_gpu_trained and not use_multi_gpu:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # if model was trained single-GPU but we want to use multi-GPU,
    # add 'module.' prefix
    elif not multi_gpu_trained and use_multi_gpu:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    
    # setup multi-GPU if requested and available
    if use_multi_gpu:
        print(f"Setting up multi-GPU inference on {device_info['num_gpus']} GPUs")
        model = nn.DataParallel(model)
    
    # load state dict and move to device
    model.load_state_dict(state_dict)
    model = model.to(primary_device)
    model.eval()
    
    print(f"Model loaded successfully on {primary_device}")
    if use_multi_gpu:
        print(f"Multi-GPU inference enabled using {device_info['num_gpus']} GPUs")
    
    return model, use_multi_gpu, device_info


def evaluate_model_on_test(model, test_data, device='auto'):
    """
    Evaluate model on test data using same logic as gnn_trainer.py
    
    Args:
        model: Trained model (potentially wrapped with DataParallel)
        test_data: Test data
        device: Device specification
        
    Returns:
        Dictionary of metrics and predictions
    """
    # setup device configuration
    primary_device, use_multi_gpu, device_info = setup_device(device)
    
    model.eval()
    test_data = test_data.to(primary_device)
    
    # setup loss function (same as trainer)
    criterion = nn.MSELoss()
    
    print("\nRunning evaluation on test set...")
    if use_multi_gpu:
        print(f"Using {device_info['num_gpus']} GPUs for inference")
    
    with torch.no_grad():
        # forward pass
        predictions = model(test_data)
        targets = test_data.y
        
        # calculate metrics using same function as trainer
        metrics = calculate_gnn_metrics(predictions, targets)
        
        # add test loss (same as trainer evaluate method)
        metrics['test_loss'] = criterion(predictions, targets).item()
    
    return metrics, predictions.cpu().numpy(), targets.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Test a trained GNN model with multi-GPU support')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model (.pt file)')
    parser.add_argument('--data_dir', type=str, default='data/gnn_ready',
                        help='Directory containing preprocessed GNN data')
    parser.add_argument('--model_type', type=str, 
                        choices=['gcn', 'gat', 'sage', 'transformer', 'hybrid'],
                        help='Model type (if not inferable from checkpoint)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'multi-gpu'],
                        help='Device to use (auto/cpu/cuda/multi-gpu)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GNN Model Testing with Multi-GPU Support")
    print("="*60)
    
    # display device information
    display_device_info(args.device)
    
    try:
        # load GNN data using same function as train_gnn.py
        print("Loading GNN data...")
        data = load_gnn_data(args.data_dir)
        print(f"Data loaded successfully from {args.data_dir}")
        
        # print data info (same as train_gnn.py)
        print(f"\nDataset Information:")
        print(f"  Train nodes: {data['train_data'].x.shape[0]}")
        print(f"  Val nodes: {data['val_data'].x.shape[0]}")
        print(f"  Test nodes: {data['test_data'].x.shape[0]}")
        print(f"  Features: {data['n_features']}")
        print(f"  Targets: {data['n_targets']}")
        print(f"  Graph edges: {data['test_data'].edge_index.shape[1]}")
        
        # load model
        model, use_multi_gpu, device_info = load_model_from_checkpoint(
            model_path=args.model_path,
            num_features=data['n_features'],
            num_targets=data['n_targets'],
            model_type=args.model_type,
            device=args.device
        )
        
        # evaluate model using same logic as trainer
        metrics, predictions, targets = evaluate_model_on_test(
            model=model,
            test_data=data['test_data'],
            device=args.device
        )
        
        # print results (same format as trainer)
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # save predictions if requested
        if args.save_predictions:
            output_file = Path(args.model_path).stem + '_predictions.npz'
            np.savez(
                output_file,
                predictions=predictions,
                targets=targets,
                metrics=metrics,
                device_info=device_info,
                multi_gpu_used=use_multi_gpu
            )
            print(f"\nPredictions saved to: {output_file}")
        
        print(f"\n{'='*60}")
        print("Testing completed successfully!")
        if use_multi_gpu:
            print(f"Multi-GPU inference was used with {device_info['num_gpus']} GPUs")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Make sure you have run the data preparation script first:")
        print("  python scripts/prepare_gnn_dataset.py")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 