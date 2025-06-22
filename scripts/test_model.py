"""
Script simple para probar modelos GNN entrenados en el conjunto de test.

Uso:
    python scripts/test_model.py --model_path final_model.pt --data_dir data/gnn_ready
    python scripts/test_model.py --model_path final_model.pt --data_dir data/gnn_ready --model_type gat
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_models import create_gnn_model, calculate_gnn_metrics


def load_model_from_checkpoint(
    model_path: str,
    model_type: str = None,
    num_features: int = None,
    num_targets: int = 1,
    device: str = 'auto'
):
    """
    Load a trained GNN model from checkpoint.
    
    Args:
        model_path: Path to the saved model (.pt file)
        model_type: Type of model ('gat', 'gcn', etc.). If None, tries to infer from checkpoint
        num_features: Number of input features. If None, tries to infer from data
        num_targets: Number of target variables
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Loading model from: {model_path}")
    
    # load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    # get model config from checkpoint if available
    model_config = checkpoint.get('model_config', {})
    
    # infer num_features from state_dict if not provided
    if num_features is None:
        state_dict = checkpoint['model_state_dict']
        # look for first layer input size
        for key, tensor in state_dict.items():
            if 'convs.0' in key and 'weight' in key:
                if len(tensor.shape) >= 2:
                    num_features = tensor.shape[1]  # input dim
                    break
        
        if num_features is None:
            raise ValueError("Could not infer num_features. Please specify in data or provide manually")
    
    print(f"Creating {model_type.upper()} model with {num_features} features, {num_targets} targets")
    
    # create model with default config (will be overridden by state_dict)
    default_configs = {
        'gat': {'hidden_dim': 128, 'num_heads': 8, 'num_layers': 3},
        'gcn': {'hidden_dim': 128, 'num_layers': 3},
        'sage': {'hidden_dim': 128, 'num_layers': 3},
        'transformer': {'hidden_dim': 128, 'num_heads': 8, 'num_layers': 3},
        'hybrid': {'hidden_dim': 128}
    }
    
    config = default_configs.get(model_type, {})
    config.update(model_config)  # use saved config if available
    
    model = create_gnn_model(
        model_type=model_type,
        num_features=num_features,
        num_targets=num_targets,
        **config
    )
    
    # load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def load_test_data(data_dir: str):
    """
    Load test data and metadata.
    
    Args:
        data_dir: Directory containing GNN data files
        
    Returns:
        Dictionary with test data and metadata
    """
    data_path = Path(data_dir)
    
    print(f"Loading test data from: {data_dir}")
    
    # load test data
    test_data = torch.load(data_path / 'test_data.pt')
    
    # load metadata
    with open(data_path / 'train_feature_names.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Test data loaded:")
    print(f"  Nodes: {test_data.x.shape[0]}")
    print(f"  Features: {test_data.x.shape[1]}")
    print(f"  Edges: {test_data.edge_index.shape[1]}")
    print(f"  Targets: {test_data.y.shape}")
    
    return {
        'test_data': test_data,
        'n_features': metadata['n_features'],
        'n_targets': metadata['n_targets'],
        'feature_names': metadata['features'],
        'target_names': metadata['targets']
    }


def evaluate_model(model, test_data, device='auto'):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_data: Test data
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.eval()
    test_data = test_data.to(device)
    
    print("\nRunning evaluation...")
    
    with torch.no_grad():
        # get predictions
        predictions = model(test_data)
        targets = test_data.y
        
        # calculate metrics
        metrics = calculate_gnn_metrics(predictions, targets)
        
        # additional metrics
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)
        metrics['test_loss'] = mse_loss.item()
    
    return metrics, predictions.cpu().numpy(), targets.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Test a trained GNN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model (.pt file)')
    parser.add_argument('--data_dir', type=str, default='data/gnn_ready',
                        help='Directory containing test data')
    parser.add_argument('--model_type', type=str, 
                        choices=['gcn', 'gat', 'sage', 'transformer', 'hybrid'],
                        help='Model type (if not inferable from checkpoint)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GNN Model Testing")
    print("="*60)
    
    try:
        # load test data
        data_info = load_test_data(args.data_dir)
        
        # load model
        model = load_model_from_checkpoint(
            model_path=args.model_path,
            model_type=args.model_type,
            num_features=data_info['n_features'],
            num_targets=data_info['n_targets'],
            device=args.device
        )
        
        # evaluate model
        metrics, predictions, targets = evaluate_model(
            model=model,
            test_data=data_info['test_data'],
            device=args.device
        )
        
        # print results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Test samples: {len(predictions)}")
        print(f"Features: {data_info['n_features']}")
        print(f"Targets: {data_info['n_targets']}")
        print()
        
        print("Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # save predictions if requested
        if args.save_predictions:
            output_file = Path(args.model_path).stem + '_predictions.npz'
            np.savez(
                output_file,
                predictions=predictions,
                targets=targets,
                metrics=metrics
            )
            print(f"\nPredictions saved to: {output_file}")
        
        print(f"\n{'='*60}")
        print("Testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 