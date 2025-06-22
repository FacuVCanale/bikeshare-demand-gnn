#!/usr/bin/env python3
"""
OPTUNA hyperparameter optimization script for Gate models.
Compatible with the existing gate training system.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training.gate_trainer import GateTrainer


class GateHyperparameterOptimizer:
    """
    OPTUNA-based hyperparameter optimizer for Gate models.
    """
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        model_type: str,
        output_dir: str = "models/gate",
        target_col: str = 'has_activity',
        feature_groups: Optional[list] = None,
        force_cpu: bool = False,
        optimization_metric: str = 'f1_score',
        optimization_direction: str = 'maximize'
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_type = model_type.lower()
        self.output_dir = Path(output_dir)
        self.target_col = target_col
        self.feature_groups = feature_groups
        self.force_cpu = force_cpu
        self.optimization_metric = optimization_metric
        self.optimization_direction = optimization_direction
        
        # create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize trainer
        self.trainer = GateTrainer(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            output_dir=str(output_dir),
            target_col=target_col,
            feature_groups=feature_groups
        )
        
        # prepare data once
        print("=== Preparing data for optimization ===")
        self.trainer.prepare_data()
        
        # store trials results
        self.trial_results = []
        self.best_model = None
        self.best_score = None
        self.best_params = None
        
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the given model type.
        """
        params = {}
        
        if self.model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 500, 3000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
                'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255]),
                'random_seed': 42,
                'eval_metric': 'AUC',
                'early_stopping_rounds': 100,
                'verbose': False
            }
            
            if not self.force_cpu:
                # try to use GPU
                params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })
            else:
                params['task_type'] = 'CPU'
        
        elif self.model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'random_state': 42,
                'early_stopping_rounds': 100,
                'metric': 'auc'
            }
            
            if not self.force_cpu:
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
            else:
                params['device'] = 'cpu'
        
        elif self.model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'early_stopping_rounds': 100,
                'eval_metric': 'auc'
            }
            
            if not self.force_cpu:
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0
                })
            else:
                params['tree_method'] = 'hist'
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for OPTUNA optimization.
        """
        try:
            # get hyperparameters for this trial
            hyperparameters = self._suggest_hyperparameters(trial)
            
            # train model
            self.trainer.train(
                model_type=self.model_type,
                hyperparameters=hyperparameters
            )
            
            # get validation metrics
            val_metrics = self.trainer.training_results['metrics']['validation']
            score = val_metrics[self.optimization_metric]
            
            # store trial result
            trial_result = {
                'trial_number': trial.number,
                'hyperparameters': hyperparameters,
                'score': score,
                'all_metrics': val_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            # update best model if this is better
            if self.best_score is None or (
                (self.optimization_direction == 'maximize' and score > self.best_score) or
                (self.optimization_direction == 'minimize' and score < self.best_score)
            ):
                self.best_score = score
                self.best_params = hyperparameters.copy()
                self.best_model = self.trainer.model
                
                print(f"🎯 New best {self.optimization_metric}: {score:.4f}")
            
            # log trial results
            print(f"Trial {trial.number}: {self.optimization_metric}={score:.4f}")
            print(f"  Params: {json.dumps({k: v for k, v in hyperparameters.items() if k not in ['verbose', 'eval_metric', 'early_stopping_rounds']}, indent=None)}")
            
            return score
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {str(e)}")
            # return worst possible score so optuna can continue
            return float('-inf') if self.optimization_direction == 'maximize' else float('inf')
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            study_name: Name for the study (optional)
            
        Returns:
            Dictionary with optimization results
        """
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"gate_{self.model_type}_optimization_{timestamp}"
        
        print(f"=== Starting OPTUNA Optimization ===")
        print(f"Model: {self.model_type.upper()}")
        print(f"Optimization metric: {self.optimization_metric} ({self.optimization_direction})")
        print(f"Trials: {n_trials}")
        print(f"GPU: {'No (forced CPU)' if self.force_cpu else 'Yes (if available)'}")
        
        # create study
        study = optuna.create_study(
            direction=self.optimization_direction,
            study_name=study_name,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # run optimization
        start_time = time.time()
        
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # get best results
        best_trial = study.best_trial
        
        print(f"\n=== Optimization Completed ===")
        print(f"Total time: {optimization_time:.2f} seconds")
        print(f"Best {self.optimization_metric}: {best_trial.value:.4f}")
        print(f"Best hyperparameters: {json.dumps(best_trial.params, indent=2)}")
        
        # prepare results
        results = {
            'study_name': study_name,
            'model_type': self.model_type,
            'optimization_metric': self.optimization_metric,
            'optimization_direction': self.optimization_direction,
            'n_trials': len(study.trials),
            'best_score': best_trial.value,
            'best_params': best_trial.params,
            'optimization_time': optimization_time,
            'all_trials': self.trial_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def save_best_model(self, model_name: Optional[str] = None) -> str:
        """
        Save the best model found during optimization.
        
        Args:
            model_name: Custom name for the model (optional)
            
        Returns:
            Path to saved model
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run optimization first.")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"gate_{self.model_type}_optuna_best_{timestamp}"
        
        # temporarily set the best model and params in trainer
        original_model = self.trainer.model
        original_results = self.trainer.training_results.copy()
        
        self.trainer.model = self.best_model
        self.trainer.training_results.update({
            'hyperparameters': self.best_params,
            'optimization_score': self.best_score,
            'optimization_metric': self.optimization_metric
        })
        
        # save using trainer's method
        model_path = self.trainer.save_model(model_name)
        
        # restore original state
        self.trainer.model = original_model
        self.trainer.training_results = original_results
        
        return model_path
    
    def save_optimization_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save optimization results to JSON file.
        
        Args:
            results: Optimization results dictionary
            filename: Custom filename (optional)
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_optimization_{self.model_type}_{timestamp}.json"
        
        results_path = self.output_dir / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Optimization results saved to: {results_path}")
        return str(results_path)
    
    def evaluate_best_model_on_test(self) -> Dict[str, Any]:
        """
        Evaluate the best model on test set.
        
        Returns:
            Test metrics dictionary
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run optimization first.")
        
        print("\n=== Evaluating Best Model on Test Set ===")
        
        # temporarily set best model in trainer
        original_model = self.trainer.model
        self.trainer.model = self.best_model
        
        # get test data
        test_data = self.trainer.dataset.get_test_data()
        test_metrics = self.best_model.get_metrics(test_data)
        
        # restore original model
        self.trainer.model = original_model
        
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return test_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OPTUNA hyperparameter optimization for Gate models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # data paths
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/clustered/train_cluster_features.parquet',
        help='Path to training data parquet file'
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default='data/clustered/val_cluster_features.parquet',
        help='Path to validation data parquet file'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default='data/clustered/test_cluster_features.parquet',
        help='Path to test data parquet file'
    )
    
    # model configuration
    parser.add_argument(
        '--model-type',
        type=str,
        default='catboost',
        choices=['catboost', 'lightgbm', 'xgboost'],
        help='Type of model to optimize'
    )
    
    # optimization settings
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        help='Timeout in seconds (optional)'
    )
    parser.add_argument(
        '--optimization-metric',
        type=str,
        default='f1_score',
        choices=['f1_score', 'auc', 'accuracy', 'precision', 'recall'],
        help='Metric to optimize'
    )
    parser.add_argument(
        '--optimization-direction',
        type=str,
        default='maximize',
        choices=['maximize', 'minimize'],
        help='Direction of optimization'
    )
    
    # feature selection
    parser.add_argument(
        '--feature-groups',
        type=str,
        nargs='+',
        help='Feature groups to include'
    )
    
    # output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/gate',
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name for the best model (auto-generated if not provided)'
    )
    parser.add_argument(
        '--results-filename',
        type=str,
        help='Filename for optimization results (auto-generated if not provided)'
    )
    
    # other options
    parser.add_argument(
        '--target-col',
        type=str,
        default='has_activity',
        help='Name of target column'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU training (disable GPU)'
    )
    parser.add_argument(
        '--no-save-model',
        action='store_true',
        help='Do not save the best model'
    )
    parser.add_argument(
        '--no-test-eval',
        action='store_true',
        help='Do not evaluate best model on test set'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        help='Custom study name for OPTUNA'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=== EcoBici Gate Model Hyperparameter Optimization ===")
    print(f"Model type: {args.model_type}")
    print(f"Trials: {args.n_trials}")
    print(f"Optimization metric: {args.optimization_metric}")
    print(f"Training data: {args.train_path}")
    print(f"Validation data: {args.val_path}")
    print(f"Test data: {args.test_path}")
    print(f"Output directory: {args.output_dir}")
    
    # create optimizer
    optimizer = GateHyperparameterOptimizer(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        target_col=args.target_col,
        feature_groups=args.feature_groups,
        force_cpu=args.force_cpu,
        optimization_metric=args.optimization_metric,
        optimization_direction=args.optimization_direction
    )
    
    # run optimization
    results = optimizer.optimize(
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name
    )
    
    # save optimization results
    optimizer.save_optimization_results(results, args.results_filename)
    
    # save best model
    if not args.no_save_model:
        model_path = optimizer.save_best_model(args.model_name)
        print(f"Best model saved to: {model_path}")
    
    # evaluate best model on test set
    if not args.no_test_eval:
        test_metrics = optimizer.evaluate_best_model_on_test()
        
        # add test metrics to results
        results['test_metrics'] = test_metrics
        # re-save results with test metrics
        optimizer.save_optimization_results(results, args.results_filename)
    
    print("\n✅ Hyperparameter optimization completed!")


if __name__ == '__main__':
    main() 