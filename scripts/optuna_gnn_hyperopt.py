"""
Comprehensive Optuna hyperparameter optimization for GNN models in EcoBici demand prediction.

This script uses Optuna to systematically search for optimal hyperparameters across all
available GNN architectures (GCN, GAT, SAGE, Transformer, Hybrid) with extensive
customization options via terminal arguments.

Key features:
- Multi-model optimization with model-specific parameter spaces
- Flexible objective functions (validation loss, test metrics)
- Parallel optimization support
- Study persistence and resumption
- Comprehensive result analysis and visualization
- Integration with existing training infrastructure
- Organized directory structure with timestamps and performance metrics
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
import pandas as pd
import random
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings
import shutil

# suppress optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# set early deterministic configuration
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_models import create_gnn_model
from src.training.gnn_trainer import train_gnn_experiment, get_device_info
from scripts.train_gnn import load_gnn_data, set_seed


class GNNOptunaObjective:
    """
    Optuna objective function for GNN hyperparameter optimization.
    
    Handles model-specific parameter spaces, training, and evaluation
    for systematic hyperparameter search across different GNN architectures.
    """
    
    def __init__(
        self,
        model_type: str,
        data: Dict[str, Any],
        base_config: Dict[str, Any],
        objective_metric: str = 'val_loss',
        use_test_metric: bool = False,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        base_save_dir: str = 'experiments/optuna',
        timestamp: str = None,
        seed: int = 42
    ):
        """
        Initialize optimization objective.
        
        Args:
            model_type: GNN model type to optimize
            data: Training/validation/test data dictionary
            base_config: Base configuration template
            objective_metric: Metric to optimize ('val_loss', 'val_r2', 'test_rmse', etc.)
            use_test_metric: Whether to use test set for final evaluation
            max_epochs: Maximum training epochs per trial
            early_stopping_patience: Early stopping patience
            device: Training device
            base_save_dir: Base directory to save trial results
            timestamp: Timestamp string for organizing results
            seed: Random seed base
        """
        self.model_type = model_type
        self.data = data
        self.base_config = base_config.copy()
        self.objective_metric = objective_metric
        self.use_test_metric = use_test_metric
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # create organized directory structure
        self.experiment_dir = Path(base_save_dir) / f"optuna_{self.timestamp}"
        self.model_dir = self.experiment_dir / model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed_base = seed
        
        # trial counter for unique seeds
        self.trial_count = 0
        
        # configure base training parameters
        self.base_config['fit_params'].update({
            'epochs': max_epochs,
            'early_stopping_patience': early_stopping_patience,
            'save_best_model': True,  # enable to save individual trial models
            'save_checkpoints': False,
            'verbose': False
        })
        self.base_config['trainer_params']['device'] = device
        
        # setup logging
        self.setup_logging()
        
        # track all trials for summary
        self.trials_summary = []
        
    def setup_logging(self):
        """Setup logging for optimization trials"""
        log_file = self.model_dir / 'optimization.log'
        
        self.logger = logging.getLogger(f'optuna_{self.model_type}_{self.timestamp}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on model type.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        config = self.base_config.copy()
        
        # common parameters for all models
        common_params = {
            'hidden_dim': trial.suggest_categorical(
                'hidden_dim', [64, 96, 128, 160, 192, 256, 320, 384, 512]
            ),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False])
        }
        
        # training parameters
        training_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True),
            'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'adamw']),
            'scheduler_name': trial.suggest_categorical('scheduler_name', ['plateau', 'cosine', 'none'])
        }
        
        # model-specific parameters
        model_specific_params = {}
        
        if self.model_type == 'gcn':
            model_specific_params.update({
                'num_layers': trial.suggest_int('num_layers', 2, 6),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu'])
            })
            
        elif self.model_type == 'gat':
            model_specific_params.update({
                'num_layers': trial.suggest_int('num_layers', 2, 5),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4, 6, 8, 12, 16]),
                'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3, step=0.05)
            })
            
        elif self.model_type == 'sage':
            model_specific_params.update({
                'num_layers': trial.suggest_int('num_layers', 2, 6),
                'aggregation': trial.suggest_categorical('aggregation', ['mean', 'max', 'add']),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu'])
            })
            
        elif self.model_type == 'transformer':
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 6, 8, 12, 16])
            # ensure hidden_dim is divisible by num_heads
            valid_hidden_dims = [h for h in [64, 96, 128, 160, 192, 256, 320, 384, 512] if h % num_heads == 0]
            if not valid_hidden_dims:
                # fallback to a safe combination
                if num_heads <= 8:
                    common_params['hidden_dim'] = 128
                else:
                    common_params['hidden_dim'] = 256
            else:
                common_params['hidden_dim'] = trial.suggest_categorical(
                    'hidden_dim_adjusted', valid_hidden_dims
                )
            
            model_specific_params.update({
                'num_layers': trial.suggest_int('num_layers', 2, 5),
                'num_heads': num_heads
            })
            
        elif self.model_type == 'hybrid':
            model_specific_params.update({
                'use_temporal': trial.suggest_categorical('use_temporal', [True, False]),
                'temporal_dim': trial.suggest_categorical('temporal_dim', [32, 48, 64, 96, 128])
            })
            # adjust learning rate for hybrid model complexity
            training_params['learning_rate'] = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True)
        
        # update configuration
        config['model_params'].update(common_params)
        config['model_params'].update(model_specific_params)
        config['training_params'].update(training_params)
        
        return config
    
    def create_trial_directory(self, trial_number: int, r2_score: float) -> Path:
        """
        Create a directory for this trial with R² score in the name.
        
        Args:
            trial_number: Trial number
            r2_score: R² score achieved
            
        Returns:
            Path to trial directory
        """
        # format R² score for directory name
        r2_str = f"r2_{r2_score:.4f}".replace('.', '_')
        trial_dir_name = f"trial_{trial_number:03d}_{r2_str}"
        trial_dir = self.model_dir / trial_dir_name
        trial_dir.mkdir(exist_ok=True)
        
        return trial_dir
    
    def save_trial_results(
        self, 
        trial_number: int, 
        trial_params: Dict[str, Any],
        results: Dict[str, Any],
        model,
        trial_dir: Path
    ):
        """
        Save comprehensive trial results in the trial directory.
        
        Args:
            trial_number: Trial number
            trial_params: Trial hyperparameters
            results: Training results
            model: Trained model
            trial_dir: Directory to save results
        """
        # save trial summary
        trial_summary = {
            'trial_number': trial_number,
            'hyperparameters': trial_params,
            'test_metrics': results['test_metrics'],
            'final_val_loss': results['history']['val_loss'][-1],
            'training_epochs': len(results['history']['train_loss']),
            'experiment_name': results['experiment_name'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(trial_dir / 'trial_summary.json', 'w') as f:
            json.dump(trial_summary, f, indent=2)
        
        # save training history
        with open(trial_dir / 'training_history.json', 'w') as f:
            history_serializable = {}
            for key, value in results['history'].items():
                if isinstance(value, list):
                    history_serializable[key] = [
                        item.tolist() if hasattr(item, 'tolist') else item 
                        for item in value
                    ]
                else:
                    history_serializable[key] = value
            json.dump(history_serializable, f, indent=2)
        
        # save configuration
        with open(trial_dir / 'config.json', 'w') as f:
            json.dump(results['config'], f, indent=2)
        
        # save model architecture info
        architecture_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_structure': str(model)
        }
        
        with open(trial_dir / 'model_architecture.json', 'w') as f:
            json.dump(architecture_info, f, indent=2, default=str)
        
        # save model weights
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'hyperparameters': trial_params,
            'test_metrics': results['test_metrics']
        }, trial_dir / 'model.pt')
        
        # add to trials summary
        self.trials_summary.append({
            'trial_number': trial_number,
            'r2_score': results['test_metrics'].get('r2', 0.0),
            'rmse': results['test_metrics'].get('rmse', float('inf')),
            'val_loss': results['history']['val_loss'][-1],
            'directory': trial_dir.name,
            'hyperparameters': trial_params
        })
        
        # save updated trials summary
        self.save_trials_summary()
    
    def save_trials_summary(self):
        """Save summary of all trials for this model"""
        summary_file = self.model_dir / 'trials_summary.json'
        
        # sort by R² score (descending)
        sorted_trials = sorted(self.trials_summary, key=lambda x: x['r2_score'], reverse=True)
        
        summary_data = {
            'model_type': self.model_type,
            'total_trials': len(self.trials_summary),
            'best_r2': sorted_trials[0]['r2_score'] if sorted_trials else 0.0,
            'best_trial_dir': sorted_trials[0]['directory'] if sorted_trials else None,
            'timestamp': self.timestamp,
            'trials': sorted_trials
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function - train and evaluate model with suggested hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to minimize/maximize
        """
        # set unique seed for this trial
        trial_seed = self.seed_base + self.trial_count
        set_seed(trial_seed)
        self.trial_count += 1
        
        try:
            # get hyperparameters for this trial
            config = self.suggest_hyperparameters(trial)
            
            # create experiment name
            experiment_name = f"optuna_trial_{trial.number}_{self.model_type}_{self.timestamp}"
            
            # temporarily save to a temp directory
            temp_save_dir = self.model_dir / f"temp_trial_{trial.number}"
            config['trainer_params']['save_dir'] = str(temp_save_dir)
            
            # train model
            model, results = train_gnn_experiment(
                model_type=self.model_type,
                train_data=self.data['train_data'],
                val_data=self.data['val_data'],
                test_data=self.data['test_data'],
                config=config,
                experiment_name=experiment_name
            )
            
            # extract objective value
            if self.use_test_metric:
                metrics_dict = results['test_metrics']
            else:
                # use last validation metrics
                if results['history']['val_metrics']:
                    metrics_dict = results['history']['val_metrics'][-1]
                else:
                    # fallback to validation loss
                    metrics_dict = {'val_loss': results['history']['val_loss'][-1]}
            
            # get objective value
            if self.objective_metric == 'val_loss':
                objective_value = results['history']['val_loss'][-1]
                direction = 'minimize'
            elif self.objective_metric in metrics_dict:
                objective_value = metrics_dict[self.objective_metric]
                # determine direction based on metric name
                if any(x in self.objective_metric.lower() for x in ['loss', 'mse', 'mae', 'rmse']):
                    direction = 'minimize'
                else:  # r2, accuracy, etc.
                    direction = 'maximize'
                    objective_value = -objective_value  # optuna minimizes by default
            else:
                raise ValueError(f"Metric {self.objective_metric} not found in results")
            
            # get R² score for directory naming
            r2_score = results['test_metrics'].get('r2', 0.0)
            
            # create proper trial directory with R² in name
            trial_dir = self.create_trial_directory(trial.number, r2_score)
            
            # save comprehensive trial results
            self.save_trial_results(
                trial_number=trial.number,
                trial_params=trial.params,
                results=results,
                model=model,
                trial_dir=trial_dir
            )
            
            # clean up temporary directory
            if temp_save_dir.exists():
                shutil.rmtree(temp_save_dir)
            
            # log trial result
            self.logger.info(
                f"Trial {trial.number}: R²={r2_score:.4f}, {self.objective_metric}={abs(objective_value):.4f}, "
                f"saved to {trial_dir.name}"
            )
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            # clean up temp directory on failure
            temp_save_dir = self.model_dir / f"temp_trial_{trial.number}"
            if temp_save_dir.exists():
                shutil.rmtree(temp_save_dir)
            # return a penalty value
            return float('inf') if 'loss' in self.objective_metric.lower() else float('-inf')


def create_base_config(device: str = 'auto') -> Dict[str, Any]:
    """Create base configuration template for optimization"""
    return {
        'trainer_params': {
            'device': device,
            'save_dir': 'experiments/optuna'
        },
        'training_params': {
            'optimizer_name': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler_name': 'plateau',
            'loss_function': 'mse'
        },
        'fit_params': {
            'epochs': 50,
            'early_stopping_patience': 10,
            'save_best_model': True,
            'save_checkpoints': False,
            'verbose': False
        },
        'model_params': {
            'hidden_dim': 128,
            'dropout': 0.2,
            'use_batch_norm': True
        }
    }


def optimize_model_hyperparameters(
    model_type: str,
    data: Dict[str, Any],
    n_trials: int = 100,
    objective_metric: str = 'val_loss',
    study_name: str = None,
    storage_url: str = None,
    n_jobs: int = 1,
    timeout: int = None,
    seed: int = 42,
    timestamp: str = None,
    device: str = 'auto',
    **objective_kwargs
) -> optuna.Study:
    """
    Optimize hyperparameters for a specific GNN model.
    
    Args:
        model_type: GNN model type to optimize
        data: Training data dictionary
        n_trials: Number of optimization trials
        objective_metric: Metric to optimize
        study_name: Name for the optimization study
        storage_url: Database URL for study persistence
        n_jobs: Number of parallel jobs
        timeout: Maximum optimization time in seconds
        seed: Random seed
        timestamp: Timestamp for organizing results
        **objective_kwargs: Additional arguments for objective function
        
    Returns:
        Completed Optuna study
    """
    # create study name if not provided
    if study_name is None:
        study_name = f"gnn_optuna_{model_type}_{timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # determine study direction
    direction = 'minimize' if any(x in objective_metric.lower() for x in ['loss', 'mse', 'mae', 'rmse']) else 'maximize'
    
    # create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        load_if_exists=True
    )
    
    # create objective
    base_config = create_base_config(device=device)
    objective = GNNOptunaObjective(
        model_type=model_type,
        data=data,
        base_config=base_config,
        objective_metric=objective_metric,
        device=device,
        seed=seed,
        timestamp=timestamp,
        **objective_kwargs
    )
    
    print(f"\nStarting hyperparameter optimization for {model_type.upper()}")
    print(f"Study: {study_name}")
    print(f"Objective: {objective_metric} ({direction})")
    print(f"Trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Results directory: {objective.experiment_dir}")
    if timeout:
        print(f"Timeout: {timeout}s ({timeout//3600}h {(timeout%3600)//60}m)")
    print(f"{'='*60}")
    
    # optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs if n_jobs > 1 else None,
        show_progress_bar=True
    )
    
    return study


def analyze_optimization_results(
    study: optuna.Study,
    model_type: str,
    timestamp: str,
    base_save_dir: str = 'experiments/optuna'
) -> Dict[str, Any]:
    """
    Analyze and save optimization results.
    
    Args:
        study: Completed Optuna study
        model_type: Model type that was optimized
        timestamp: Timestamp for organizing results
        base_save_dir: Base directory for saving results
        
    Returns:
        Analysis results dictionary
    """
    experiment_dir = Path(base_save_dir) / f"optuna_{timestamp}"
    model_dir = experiment_dir / model_type
    
    # best trial analysis
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    analysis = {
        'model_type': model_type,
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'best_value': best_value,
        'best_params': best_params,
        'best_trial_number': best_trial.number,
        'optimization_direction': study.direction.name,
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'experiment_directory': str(experiment_dir),
        'model_directory': str(model_dir)
    }
    
    print(f"\nOptimization Results for {model_type.upper()}:")
    print(f"{'='*50}")
    print(f"Best {study.direction.name}: {best_value:.4f}")
    print(f"Best trial: {best_trial.number}")
    print(f"Completed trials: {analysis['completed_trials']}/{analysis['n_trials']}")
    print(f"Failed trials: {analysis['failed_trials']}")
    print(f"Pruned trials: {analysis['pruned_trials']}")
    print(f"Results saved in: {model_dir}")
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # parameter importance analysis
    try:
        importance = optuna.importance.get_param_importances(study)
        analysis['param_importance'] = importance
        
        print(f"\nParameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.3f}")
            
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
        analysis['param_importance'] = {}
    
    # save detailed analysis
    analysis_file = model_dir / 'optimization_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # save trials dataframe
    df_trials = study.trials_dataframe()
    df_trials.to_csv(model_dir / 'trials_dataframe.csv', index=False)
    
    # save best configuration in training format
    best_config = create_base_config()
    
    # map parameters back to config structure
    model_params = {}
    training_params = {}
    
    for param, value in best_params.items():
        if param in ['learning_rate', 'weight_decay', 'optimizer_name', 'scheduler_name']:
            training_params[param] = value
        else:
            model_params[param] = value
    
    best_config['model_params'].update(model_params)
    best_config['training_params'].update(training_params)
    
    with open(model_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    return analysis


def create_optimization_summary(
    results: Dict[str, Dict[str, Any]],
    timestamp: str,
    base_save_dir: str = 'experiments/optuna'
) -> pd.DataFrame:
    """
    Create summary comparison of optimization results across models.
    
    Args:
        results: Dictionary of model results
        timestamp: Timestamp for organizing results
        base_save_dir: Base directory for saving results
        
    Returns:
        Summary DataFrame
    """
    experiment_dir = Path(base_save_dir) / f"optuna_{timestamp}"
    summary_data = []
    
    for model_type, analysis in results.items():
        # try to get best R² from model summary
        model_dir = experiment_dir / model_type
        trials_summary_file = model_dir / 'trials_summary.json'
        
        best_r2 = 0.0
        if trials_summary_file.exists():
            with open(trials_summary_file, 'r') as f:
                trials_summary = json.load(f)
                best_r2 = trials_summary.get('best_r2', 0.0)
        
        summary_data.append({
            'Model': model_type.upper(),
            'Best Score': f"{analysis['best_value']:.4f}",
            'Best R²': f"{best_r2:.4f}",
            'Trials': analysis['n_trials'],
            'Completed': analysis['completed_trials'],
            'Failed': analysis['failed_trials'],
            'Success Rate': f"{analysis['completed_trials']/analysis['n_trials']*100:.1f}%",
            'Best Trial': analysis['best_trial_number'],
            'Directory': analysis['model_directory']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # save summary
    summary_path = experiment_dir / 'optimization_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results in: {experiment_dir}")
    
    return df_summary


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
            print(f"\n🚀 Multi-GPU optimization will be enabled!")
            print(f"   Using {device_info['num_gpus']} GPUs for accelerated training")
        else:
            print(f"\n⚡ Single GPU optimization enabled")
    else:
        print(f"🐌 CPU optimization mode")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter optimization for GNN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize all models with default settings
  python scripts/optuna_gnn_hyperopt.py --models all --trials 100

  # Optimize specific models with custom settings  
  python scripts/optuna_gnn_hyperopt.py --models gcn gat --trials 200 --objective val_r2

  # Parallel optimization with database storage
  python scripts/optuna_gnn_hyperopt.py --models transformer --trials 500 --jobs 4 --storage sqlite:///optuna.db

  # Quick optimization with test metric
  python scripts/optuna_gnn_hyperopt.py --models sage --trials 50 --objective test_rmse --use_test_metric

Results Structure:
  experiments/optuna_YYYYMMDD_HHMMSS/
  ├── gcn/
  │   ├── trial_001_r2_0_8534/
  │   │   ├── model.pt
  │   │   ├── training_history.json
  │   │   ├── model_architecture.json
  │   │   └── config.json
  │   ├── trial_002_r2_0_8621/
  │   └── trials_summary.json
  └── optimization_summary.csv
        """
    )
    
    # data and model selection
    parser.add_argument('--data_dir', type=str, default='data/gnn_ready',
                        help='Directory containing preprocessed GNN data')
    parser.add_argument('--models', nargs='+', 
                        choices=['gcn', 'gat', 'sage', 'transformer', 'hybrid', 'all'],
                        default=['all'], help='GNN models to optimize')
    
    # optimization parameters
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials per model')
    parser.add_argument('--objective', type=str, default='val_loss',
                        choices=['val_loss', 'val_r2', 'val_rmse', 'val_mae', 'test_rmse', 'test_r2', 'test_mae'],
                        help='Objective metric to optimize')
    parser.add_argument('--use_test_metric', action='store_true',
                        help='Use test set metrics instead of validation (use carefully)')
    
    # training configuration
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum epochs per trial')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'multi-gpu'],
                        help='Training device (auto/cpu/cuda/multi-gpu)')
    
    # parallel and persistence
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of parallel optimization jobs')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database URL for study persistence (e.g., sqlite:///optuna.db)')
    parser.add_argument('--study_prefix', type=str, default='gnn_optuna',
                        help='Prefix for study names')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum optimization time in seconds')
    
    # reproducibility and output
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='experiments/optuna',
                        help='Base directory to save optimization results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing studies if found')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Custom timestamp for organizing results (default: auto-generated)')
    
    args = parser.parse_args()
    
    # create timestamp for this optimization run
    timestamp = args.timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # expand 'all' models
    if 'all' in args.models:
        models_to_optimize = ['gcn', 'gat', 'sage', 'transformer', 'hybrid']
    else:
        models_to_optimize = args.models
    
    # set global seed
    set_seed(args.seed)
    
    print(f"{'='*80}")
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Timestamp: {timestamp}")
    print(f"Models: {', '.join(models_to_optimize)}")
    print(f"Trials per model: {args.trials}")
    print(f"Objective: {args.objective}")
    print(f"Results will be saved in: {args.save_dir}/optuna_{timestamp}")
    print(f"{'='*80}")
    
    print("\nLoading GNN data...")
    try:
        data = load_gnn_data(args.data_dir)
        print(f"Data loaded successfully from {args.data_dir}")
        print(f"Train samples: {data['train_data'].x.shape[0]}")
        print(f"Validation samples: {data['val_data'].x.shape[0]}")
        print(f"Test samples: {data['test_data'].x.shape[0]}")
        print(f"Features: {data['n_features']}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Make sure you have run the data preparation script first:")
        print("  python scripts/prepare_gnn_dataset.py")
        return
    
    # display device information
    display_device_info(args.device)
    
    # run optimization for each model
    results = {}
    
    for model_type in models_to_optimize:
        print(f"\n{'='*80}")
        print(f"OPTIMIZING {model_type.upper()} MODEL")
        print(f"{'='*80}")
        
        try:
            study_name = f"{args.study_prefix}_{model_type}_{timestamp}"
            if args.resume and args.storage:
                # try to load existing study
                try:
                    existing_study = optuna.load_study(
                        study_name=f"{args.study_prefix}_{model_type}",
                        storage=args.storage
                    )
                    print(f"Resuming existing study with {len(existing_study.trials)} trials")
                    study_name = existing_study.study_name
                except:
                    print("No existing study found, creating new one")
            
            study = optimize_model_hyperparameters(
                model_type=model_type,
                data=data,
                n_trials=args.trials,
                objective_metric=args.objective,
                study_name=study_name,
                storage_url=args.storage,
                n_jobs=args.jobs,
                timeout=args.timeout,
                seed=args.seed,
                timestamp=timestamp,
                device=args.device,
                use_test_metric=args.use_test_metric,
                max_epochs=args.max_epochs,
                early_stopping_patience=args.patience,
                base_save_dir=args.save_dir
            )
            
            # analyze results
            analysis = analyze_optimization_results(
                study=study,
                model_type=model_type,
                timestamp=timestamp,
                base_save_dir=args.save_dir
            )
            
            results[model_type] = analysis
            
        except Exception as e:
            print(f"Error optimizing {model_type}: {str(e)}")
            continue
    
    # create overall summary
    if results:
        summary_df = create_optimization_summary(results, timestamp, args.save_dir)
        
        # save optimization metadata
        experiment_dir = Path(args.save_dir) / f"optuna_{timestamp}"
        metadata = {
            'optimization_date': datetime.now().isoformat(),
            'timestamp': timestamp,
            'models_optimized': list(results.keys()),
            'trials_per_model': args.trials,
            'objective_metric': args.objective,
            'max_epochs': args.max_epochs,
            'early_stopping_patience': args.patience,
            'seed': args.seed,
            'parallel_jobs': args.jobs,
            'total_trials': sum(r['n_trials'] for r in results.values()),
            'total_completed': sum(r['completed_trials'] for r in results.values()),
            'experiment_directory': str(experiment_dir),
            'command_line_args': vars(args)
        }
        
        with open(experiment_dir / 'optimization_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {experiment_dir}")
        print(f"Total trials run: {metadata['total_trials']}")
        print(f"Total successful: {metadata['total_completed']}")
        print(f"Success rate: {metadata['total_completed']/metadata['total_trials']*100:.1f}%")
        print(f"\nEach trial saved in organized structure:")
        print(f"  {experiment_dir}/{{model}}/trial_{{number}}_r2_{{score}}/")
        print(f"{'='*80}")
    
    else:
        print("No successful optimizations completed.")


if __name__ == '__main__':
    main() 