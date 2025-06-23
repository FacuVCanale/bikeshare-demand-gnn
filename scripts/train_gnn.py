"""
Training script for Graph Neural Networks on EcoBici clustered data.

This script loads the preprocessed GNN data and trains various GNN architectures
for bike demand prediction, including model comparison and evaluation.

Updated to handle temporal snapshots and data without leakage.
"""

import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import json
import argparse
import sys
from typing import Dict, Any, List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_models import create_gnn_model, print_model_summary


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


def load_gnn_data(data_dir: str) -> Dict[str, Any]:
    """
    Load preprocessed GNN data with support for temporal snapshots.
    
    Args:
        data_dir: Directory containing GNN data files
        
    Returns:
        Dictionary containing train/val/test data and metadata
    """
    data_path = Path(data_dir)
    
    # Try to load temporal snapshot data first (new format)
    train_list_path = data_path / 'train_data_list.pt'
    if train_list_path.exists():
        print("Loading temporal snapshot data (new format)...")
        train_data_list = torch.load(train_list_path, weights_only=False)
        val_data_list = torch.load(data_path / 'val_data_list.pt', weights_only=False)
        test_data_list = torch.load(data_path / 'test_data_list.pt', weights_only=False)
        
        # Also load single snapshots for compatibility
        train_data = torch.load(data_path / 'train_data.pt', weights_only=False)
        val_data = torch.load(data_path / 'val_data.pt', weights_only=False)
        test_data = torch.load(data_path / 'test_data.pt', weights_only=False)
        
        data_format = 'temporal_snapshots'
        print(f"  Loaded {len(train_data_list)} train snapshots")
        print(f"  Loaded {len(val_data_list)} val snapshots")
        print(f"  Loaded {len(test_data_list)} test snapshots")
    else:
        print("Loading single snapshot data (legacy format)...")
        # Fallback to old format
        train_data = torch.load(data_path / 'train_data.pt', weights_only=False)
        val_data = torch.load(data_path / 'val_data.pt', weights_only=False)
        test_data = torch.load(data_path / 'test_data.pt', weights_only=False)
        
        train_data_list = [train_data]
        val_data_list = [val_data]
        test_data_list = [test_data]
        data_format = 'single_snapshot'
    
    # Load metadata
    with open(data_path / 'train_feature_names.json', 'r') as f:
        train_metadata = json.load(f)
    
    with open(data_path / 'processing_config.json', 'r') as f:
        processing_config = json.load(f)
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_data_list': train_data_list,
        'val_data_list': val_data_list,
        'test_data_list': test_data_list,
        'feature_names': train_metadata['features'],
        'target_names': train_metadata['targets'],
        'n_features': train_metadata['n_features'],
        'n_targets': train_metadata['n_targets'],
        'processing_config': processing_config,
        'data_format': data_format,
        'sequence_length': train_metadata.get('sequence_length', 1),
        'n_snapshots': {
            'train': len(train_data_list),
            'val': len(val_data_list),
            'test': len(test_data_list)
        }
    }


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, 
                   valid_mask: torch.Tensor = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True values [n_samples, n_targets]
        y_pred: Predicted values [n_samples, n_targets]
        valid_mask: Mask for valid predictions [n_samples]
        
    Returns:
        Dictionary of metrics
    """
    # Apply valid mask if provided
    if valid_mask is not None:
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Handle multi-target case
    if y_true_np.ndim > 1 and y_true_np.shape[1] > 1:
        # Average metrics across targets
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        for i in range(y_true_np.shape[1]):
            mse_scores.append(mean_squared_error(y_true_np[:, i], y_pred_np[:, i]))
            mae_scores.append(mean_absolute_error(y_true_np[:, i], y_pred_np[:, i]))
            r2_scores.append(r2_score(y_true_np[:, i], y_pred_np[:, i]))
        
        mse = np.mean(mse_scores)
        mae = np.mean(mae_scores)
        r2 = np.mean(r2_scores)
    else:
        # Single target
        if y_true_np.ndim > 1:
            y_true_np = y_true_np.flatten()
            y_pred_np = y_pred_np.flatten()
            
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }


def train_temporal_gnn(
    model: torch.nn.Module,
    train_data_list: List[Data],
    val_data_list: List[Data],
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train GNN model on temporal snapshots.
    
    Args:
        model: GNN model to train
        train_data_list: List of training snapshots
        val_data_list: List of validation snapshots
        device: Device to train on
        config: Training configuration
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("Training GNN on temporal snapshots...")
    
    # Extract parameters
    epochs = config['fit_params']['epochs']
    learning_rate = config['training_params']['learning_rate']
    weight_decay = config['training_params']['weight_decay']
    patience = config['fit_params']['early_stopping_patience']
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        train_masks = []
        
        # Shuffle training snapshots
        train_indices = torch.randperm(len(train_data_list))
        
        for idx in tqdm(train_indices, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            data = train_data_list[idx].to(device)
            
            optimizer.zero_grad()
            out = model(data)
            
            # Apply valid mask if available
            if hasattr(data, 'valid_mask'):
                valid_mask = data.valid_mask
                loss = criterion(out[valid_mask], data.y[valid_mask])
                train_predictions.append(out[valid_mask].detach())
                train_targets.append(data.y[valid_mask])
                train_masks.append(valid_mask.cpu())
            else:
                loss = criterion(out, data.y)
                train_predictions.append(out.detach())
                train_targets.append(data.y)
                train_masks.append(torch.ones(len(data.y), dtype=torch.bool))
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        val_masks = []
        
        with torch.no_grad():
            for data in val_data_list:
                data = data.to(device)
                out = model(data)
                
                if hasattr(data, 'valid_mask'):
                    valid_mask = data.valid_mask
                    loss = criterion(out[valid_mask], data.y[valid_mask])
                    val_predictions.append(out[valid_mask].cpu())
                    val_targets.append(data.y[valid_mask].cpu())
                    val_masks.append(valid_mask.cpu())
                else:
                    loss = criterion(out, data.y)
                    val_predictions.append(out.cpu())
                    val_targets.append(data.y.cpu())
                    val_masks.append(torch.ones(len(data.y), dtype=torch.bool))
                
                val_losses.append(loss.item())
        
        # Compute epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Concatenate predictions for metrics
        all_train_pred = torch.cat(train_predictions, dim=0)
        all_train_true = torch.cat(train_targets, dim=0)
        all_val_pred = torch.cat(val_predictions, dim=0)
        all_val_true = torch.cat(val_targets, dim=0)
        
        train_metrics = compute_metrics(all_train_true, all_train_pred)
        val_metrics = compute_metrics(all_val_true, all_val_pred)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % config['fit_params'].get('log_frequency', 10) == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, "
                  f"Val R²={val_metrics['r2']:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss - config['fit_params'].get('min_improvement', 0.001):
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    return model, history


def evaluate_temporal_gnn(
    model: torch.nn.Module,
    data_list: List[Data],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate GNN model on temporal snapshots.
    
    Args:
        model: Trained GNN model
        data_list: List of data snapshots
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            out = model(data)
            
            if hasattr(data, 'valid_mask'):
                valid_mask = data.valid_mask
                all_predictions.append(out[valid_mask].cpu())
                all_targets.append(data.y[valid_mask].cpu())
            else:
                all_predictions.append(out.cpu())
                all_targets.append(data.y.cpu())
    
    # Concatenate all predictions
    all_pred = torch.cat(all_predictions, dim=0)
    all_true = torch.cat(all_targets, dim=0)
    
    return compute_metrics(all_true, all_pred)


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configurations for different GNN models.
    
    Returns:
        Dictionary of model configurations
    """
    base_config = {
        'trainer_params': {
            'device': 'auto',
            'save_dir': 'experiments/gnn_temporal'
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
            'verbose': True,
            'log_frequency': 10,
            'min_improvement': 0.001
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
    Run a single GNN training experiment with temporal snapshots.
    
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
    
    # Setup device
    if config['trainer_params']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['trainer_params']['device'])
    
    print(f"Using device: {device}")
    
    # Print data info
    print(f"Dataset Information:")
    if data['data_format'] == 'temporal_snapshots':
        print(f"  Train snapshots: {data['n_snapshots']['train']}")
        print(f"  Val snapshots: {data['n_snapshots']['val']}")
        print(f"  Test snapshots: {data['n_snapshots']['test']}")
        print(f"  Sequence length: {data['sequence_length']}")
    else:
        print(f"  Train nodes: {data['train_data'].x.shape[0]}")
        print(f"  Val nodes: {data['val_data'].x.shape[0]}")
        print(f"  Test nodes: {data['test_data'].x.shape[0]}")
    
    print(f"  Features: {data['n_features']}")
    print(f"  Targets: {data['n_targets']}")
    print(f"  Graph edges: {data['train_data'].edge_index.shape[1]}")
    print(f"  Data format: {data['data_format']}")
    
    # Check for data leakage fix
    if data['processing_config'].get('leakage_fixed', False):
        print(f"  ✅ Data leakage: FIXED")
    else:
        print(f"  ⚠️  Data leakage: NOT FIXED (legacy data)")
    
    # Create and print model summary
    model = create_gnn_model(
        model_type=model_type,
        num_features=data['n_features'],
        num_targets=data['n_targets'],
        **config['model_params']
    )
    
    print(f"\nModel Summary:")
    print_model_summary(model, (data['train_data'].x.shape[0], data['n_features']))
    
    # Train model using temporal approach
    print(f"\nStarting training...")
    trained_model, history = train_temporal_gnn(
        model=model,
        train_data_list=data['train_data_list'],
        val_data_list=data['val_data_list'],
        device=device,
        config=config
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = evaluate_temporal_gnn(
        model=trained_model,
        data_list=data['test_data_list'],
        device=device
    )
    
    # Add test prefix to metrics
    test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
    test_metrics_prefixed['test_loss'] = test_metrics['mse']  # For compatibility
    
    # Print results
    print(f"\nFinal Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model if requested
    if config['fit_params'].get('save_best_model', False):
        save_dir = Path(config['trainer_params']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"{experiment_name or model_type}_best_model.pt"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': config['model_params'],
            'test_metrics': test_metrics,
            'data_format': data['data_format']
        }, model_path)
        print(f"  Model saved to: {model_path}")
    
    return {
        'model': trained_model,
        'test_metrics': test_metrics_prefixed,
        'training_history': history,
        'experiment_name': experiment_name or model_type,
        'config': config,
        'data_format': data['data_format']
    }


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
    print(f"Data format: {data['data_format']}")
    if data['data_format'] == 'temporal_snapshots':
        print(f"Total training snapshots: {data['n_snapshots']['train']}")
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
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Create comparison table
        comparison_data = []
        for model_type, results in all_results.items():
            metrics = results['test_metrics']
            comparison_data.append({
                'Model': model_type.upper(),
                'Test Loss': f"{metrics.get('test_loss', float('nan')):.4f}",
                'RMSE': f"{metrics.get('test_rmse', float('nan')):.4f}",
                'MAE': f"{metrics.get('test_mae', float('nan')):.4f}",
                'R²': f"{metrics.get('test_r2', float('nan')):.4f}",
                'Data Format': results['data_format']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model = min(all_results.items(), key=lambda x: x[1]['test_metrics'].get('test_loss', float('inf')))
        print(f"\nBest Model: {best_model[0].upper()} (Test Loss: {best_model[1]['test_metrics']['test_loss']:.4f})")
        
        # Save results
        if save_results:
            results_dir = Path('experiments/gnn_temporal/comparison_results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            with open(results_dir / 'detailed_results.json', 'w') as f:
                # Convert any numpy arrays to lists for JSON serialization
                serializable_results = {}
                for model_type, results in all_results.items():
                    serializable_results[model_type] = {
                        'test_metrics': results['test_metrics'],
                        'experiment_name': results['experiment_name'],
                        'config': results['config'],
                        'data_format': results['data_format']
                    }
                json.dump(serializable_results, f, indent=2)
            
            # Save comparison table
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement)')
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
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='How often to log progress (every N epochs)')
    parser.add_argument('--min_improvement', type=float, default=0.001,
                        help='Minimum improvement required to save best model')
    parser.add_argument('--save_results', action='store_true',
                        help='Save detailed results')
    
    args = parser.parse_args()
    
    # set random seed for reproducibility
    set_seed(args.seed)
    
    print("Loading GNN data...")
    try:
        data = load_gnn_data(args.data_dir)
        print(f"Data loaded successfully from {args.data_dir}")
        print(f"Data format: {data['data_format']}")
        
        if data['processing_config'].get('leakage_fixed', False):
            print("✅ Using data WITHOUT leakage - results will be realistic!")
        else:
            print("⚠️  Using data WITH potential leakage - results may be overly optimistic!")
            
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
        configs[model_type]['fit_params']['early_stopping_patience'] = args.patience
        configs[model_type]['fit_params']['log_frequency'] = args.log_frequency
        configs[model_type]['fit_params']['min_improvement'] = args.min_improvement
        
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
            results_dir = Path('experiments/gnn_temporal/single_results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / f'{args.model}_results.json', 'w') as f:
                serializable_results = {
                    'test_metrics': results['test_metrics'],
                    'experiment_name': results['experiment_name'],
                    'config': results['config'],
                    'data_format': results['data_format']
                }
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    main() 