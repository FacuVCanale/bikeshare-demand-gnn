"""
Training script for Graph Neural Networks on EcoBici clustered data.

This script loads the preprocessed GNN data and trains various GNN architectures
for bike demand prediction, including model comparison and evaluation.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import sys
from typing import Dict, Any, List

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_models import create_gnn_model, print_model_summary
from src.training.gnn_trainer import GNNTrainer, train_gnn_experiment


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


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configurations for different GNN models.
    
    Returns:
        Dictionary of model configurations
    """
    base_config = {
        'trainer_params': {
            'device': 'auto',
            'save_dir': 'experiments/gnn'
        },
        'training_params': {
            'optimizer_name': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler_name': 'plateau',
            'loss_function': 'mse'
        },
        'fit_params': {
            'epochs': 100,
            'early_stopping_patience': 15,
            'save_best_model': True,
            'verbose': True
        }
    }
    
    configs = {
        'gcn': {
            **base_config,
            'model_params': {
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'use_batch_norm': True,
                'activation': 'relu'
            }
        },
        
        'gat': {
            **base_config,
            'model_params': {
                'hidden_dim': 128,
                'num_layers': 3,
                'num_heads': 8,
                'dropout': 0.2,
                'attention_dropout': 0.1,
                'use_batch_norm': True
            }
        },
        
        'sage': {
            **base_config,
            'model_params': {
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'aggregation': 'mean',
                'use_batch_norm': True
            }
        },
        
        'transformer': {
            **base_config,
            'model_params': {
                'hidden_dim': 128,
                'num_layers': 3,
                'num_heads': 8,
                'dropout': 0.2,
                'use_batch_norm': True
            },
            'training_params': {
                **base_config['training_params'],
                'learning_rate': 0.0005,  # lower lr for transformer
            }
        },
        
        'hybrid': {
            **base_config,
            'model_params': {
                'hidden_dim': 128,
                'dropout': 0.2,
                'use_temporal': True,
                'temporal_dim': 64
            },
            'training_params': {
                **base_config['training_params'],
                'learning_rate': 0.0008,
                'weight_decay': 1e-4
            }
        }
    }
    
    return configs


def run_single_experiment(
    model_type: str,
    data: Dict[str, Any],
    config: Dict[str, Any],
    experiment_name: str = None
) -> Dict[str, Any]:
    """
    Run a single GNN training experiment.
    
    Args:
        model_type: Type of GNN model
        data: Data dictionary
        config: Model configuration
        experiment_name: Name for the experiment
        
    Returns:
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # print data info
    print(f"Dataset Information:")
    print(f"  Train nodes: {data['train_data'].x.shape[0]}")
    print(f"  Val nodes: {data['val_data'].x.shape[0]}")
    print(f"  Test nodes: {data['test_data'].x.shape[0]}")
    print(f"  Features: {data['n_features']}")
    print(f"  Targets: {data['n_targets']}")
    print(f"  Graph edges: {data['train_data'].edge_index.shape[1]}")
    
    # create and print model summary
    model = create_gnn_model(
        model_type=model_type,
        num_features=data['n_features'],
        num_targets=data['n_targets'],
        **config['model_params']
    )
    
    print(f"\nModel Summary:")
    print_model_summary(model, (data['train_data'].x.shape[0], data['n_features']))
    
    # train model
    trained_model, results = train_gnn_experiment(
        model_type=model_type,
        train_data=data['train_data'],
        val_data=data['val_data'],
        test_data=data['test_data'],
        config=config,
        experiment_name=experiment_name
    )
    
    # print results
    print(f"\nFinal Results:")
    test_metrics = results['test_metrics']
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return results


def run_model_comparison(
    data: Dict[str, Any],
    models_to_compare: List[str] = None,
    save_results: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run comparison between different GNN models.
    
    Args:
        data: Data dictionary
        models_to_compare: List of model types to compare
        save_results: Whether to save comparison results
        
    Returns:
        Dictionary of results for each model
    """
    if models_to_compare is None:
        models_to_compare = ['gcn', 'gat', 'sage', 'transformer', 'hybrid']
    
    configs = get_model_configs()
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"RUNNING MODEL COMPARISON")
    print(f"Models: {', '.join(models_to_compare)}")
    print(f"{'='*80}")
    
    for model_type in models_to_compare:
        if model_type not in configs:
            print(f"Warning: No configuration found for {model_type}, skipping...")
            continue
            
        try:
            experiment_name = f"comparison_{model_type}"
            results = run_single_experiment(
                model_type=model_type,
                data=data,
                config=configs[model_type],
                experiment_name=experiment_name
            )
            all_results[model_type] = results
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue
    
    # create comparison summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # create comparison table
        comparison_data = []
        for model_type, results in all_results.items():
            metrics = results['test_metrics']
            comparison_data.append({
                'Model': model_type.upper(),
                'Test Loss': f"{metrics.get('test_loss', float('nan')):.4f}",
                'RMSE': f"{metrics.get('rmse', float('nan')):.4f}",
                'MAE': f"{metrics.get('mae', float('nan')):.4f}",
                'R²': f"{metrics.get('r2', float('nan')):.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # find best model
        best_model = min(all_results.items(), key=lambda x: x[1]['test_metrics'].get('test_loss', float('inf')))
        print(f"\nBest Model: {best_model[0].upper()} (Test Loss: {best_model[1]['test_metrics']['test_loss']:.4f})")
        
        # save results
        if save_results:
            results_dir = Path('experiments/gnn/comparison_results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # save detailed results
            with open(results_dir / 'detailed_results.json', 'w') as f:
                # convert any numpy arrays to lists for JSON serialization
                serializable_results = {}
                for model_type, results in all_results.items():
                    serializable_results[model_type] = {
                        'test_metrics': results['test_metrics'],
                        'experiment_name': results['experiment_name'],
                        'config': results['config']
                    }
                json.dump(serializable_results, f, indent=2)
            
            # save comparison table
            df_comparison.to_csv(results_dir / 'model_comparison.csv', index=False)
            
            print(f"\nResults saved to {results_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Train GNN models for EcoBici demand prediction')
    parser.add_argument('--data_dir', type=str, default='data/gnn_ready',
                        help='Directory containing preprocessed GNN data')
    parser.add_argument('--model', type=str, choices=['gcn', 'gat', 'sage', 'transformer', 'hybrid', 'all'],
                        default='all', help='Model type to train')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (for GAT/Transformer)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Attention dropout rate (for GAT)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'elu', 'leaky_relu', 'gelu'],
                        help='Activation function')
    parser.add_argument('--batch_norm', action='store_true', default=True,
                        help='Use batch normalization')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save detailed results')
    
    args = parser.parse_args()
    
    print("Loading GNN data...")
    try:
        data = load_gnn_data(args.data_dir)
        print(f"Data loaded successfully from {args.data_dir}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Make sure you have run the data preparation script first:")
        print("  python scripts/prepare_gnn_dataset.py")
        return
    
    # get configurations
    configs = get_model_configs()
    
    # override config with command line arguments
    for model_type in configs:
        # fit params
        configs[model_type]['fit_params']['epochs'] = args.epochs
        
        # training params
        configs[model_type]['training_params']['learning_rate'] = args.learning_rate
        configs[model_type]['training_params']['weight_decay'] = args.weight_decay
        configs[model_type]['training_params']['optimizer_name'] = args.optimizer
        configs[model_type]['training_params']['scheduler_name'] = args.scheduler
        
        # model params
        configs[model_type]['model_params']['hidden_dim'] = args.hidden_dim
        configs[model_type]['model_params']['num_layers'] = args.num_layers
        configs[model_type]['model_params']['dropout'] = args.dropout
        configs[model_type]['model_params']['use_batch_norm'] = args.batch_norm
        
        # model-specific params
        if model_type in ['gat', 'transformer']:
            configs[model_type]['model_params']['num_heads'] = args.num_heads
        if model_type == 'gat':
            configs[model_type]['model_params']['attention_dropout'] = args.attention_dropout
        if model_type in ['gcn', 'sage']:
            configs[model_type]['model_params']['activation'] = args.activation
        
        # trainer params
        configs[model_type]['trainer_params']['device'] = args.device
    
    if args.model == 'all':
        # run comparison
        results = run_model_comparison(
            data=data,
            save_results=args.save_results
        )
    else:
        # run single model
        if args.model not in configs:
            print(f"Error: Unknown model type '{args.model}'")
            print(f"Available models: {list(configs.keys())}")
            return
        
        results = run_single_experiment(
            model_type=args.model,
            data=data,
            config=configs[args.model],
            experiment_name=args.experiment_name
        )
        
        if args.save_results:
            results_dir = Path('experiments/gnn/single_results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / f'{args.model}_results.json', 'w') as f:
                serializable_results = {
                    'test_metrics': results['test_metrics'],
                    'experiment_name': results['experiment_name'],
                    'config': results['config']
                }
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    main() 