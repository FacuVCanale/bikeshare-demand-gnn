#!/usr/bin/env python3
"""
Optimized OPTUNA hyperparameter optimization script for Gate models.
Enhanced with parallel trials, memory management, and intelligent pruning.
"""
import argparse
import json
import logging
import sys
import time
import gc
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
import numpy as np
import psutil

# add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.training.gate_trainer import GateTrainer


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))
    
    # remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # create console handler with custom format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # custom format for better readability
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger


class OptimizedGateHyperparameterOptimizer:
    """
    Enhanced OPTUNA-based hyperparameter optimizer for Gate models with:
    - Parallel trial execution
    - Intelligent memory management  
    - Advanced pruning strategies
    - Data caching and reuse
    - Convergence detection
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
        optimization_direction: str = 'maximize',
        n_jobs: int = 1,
        memory_limit_gb: float = None,
        logger: Optional[logging.Logger] = None
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
        self.n_jobs = min(n_jobs, mp.cpu_count()) if n_jobs > 1 else 1
        self.memory_limit_gb = memory_limit_gb or (psutil.virtual_memory().total / (1024**3) * 0.8)
        self.logger = logger or setup_logging()
        
        # create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # cached data to avoid reloading
        self._cached_trainer = None
        self._data_prepared = False
        
        # optimization state
        self.trial_results = []
        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.convergence_history = []
        self.memory_usage_history = []
        
        # performance tracking
        self.gpu_available = self._detect_gpu_availability()
        
        self.logger.info(f"Optimizer initialized with {self.n_jobs} parallel jobs")
        self.logger.info(f"Memory limit: {self.memory_limit_gb:.1f} GB")
        if self.gpu_available:
            self.logger.info("GPU acceleration detected and available")
        
    def _detect_gpu_availability(self) -> bool:
        """Detect if GPU is available for training."""
        if self.force_cpu:
            return False
            
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
            
        # check specific framework GPU support
        if self.model_type == 'catboost':
            try:
                from catboost.utils import get_gpu_device_count
                return get_gpu_device_count() > 0
            except:
                return False
        
        return False
    
    def _prepare_cached_data(self):
        """Prepare and cache training data to avoid reloading in each trial."""
        if self._data_prepared:
            return
            
        self.logger.info("=== PREPARING AND CACHING DATA ===")
        
        # create trainer instance
        self._cached_trainer = GateTrainer(
            train_path=self.train_path,
            val_path=self.val_path,
            test_path=self.test_path,
            output_dir=str(self.output_dir),
            target_col=self.target_col,
            feature_groups=self.feature_groups
        )
        
        # prepare data once
        self._cached_trainer.prepare_data()
        
        # log memory usage after data loading
        memory_usage = psutil.virtual_memory().percent
        self.logger.info(f"Data loaded - Memory usage: {memory_usage:.1f}%")
        
        if memory_usage > 85:
            self.logger.warning("High memory usage detected - consider reducing dataset size")
        
        self._data_prepared = True
        
    def _monitor_memory_usage(self) -> float:
        """Monitor current memory usage."""
        memory_usage = psutil.virtual_memory().percent
        self.memory_usage_history.append(memory_usage)
        
        if memory_usage > 90:
            self.logger.warning(f"High memory usage: {memory_usage:.1f}% - forcing garbage collection")
            gc.collect()
            
        return memory_usage
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Enhanced hyperparameter suggestion with smarter ranges.
        """
        params = {}
        
        if self.model_type == 'catboost':
            # catboost with intelligent defaults
            params = {
                'iterations': trial.suggest_int('iterations', 100, 2000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
                'depth': trial.suggest_int('depth', 4, 20),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 20, log=True),
                'border_count': trial.suggest_categorical('border_count', [64, 128, 255]),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 2),
                'random_seed': 42,
                'eval_metric': 'AUC',
                'early_stopping_rounds': 50,  # more aggressive early stopping
                'verbose': False,
                'use_best_model': True
            }
            
            if not self.force_cpu and self.gpu_available:
                params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })
            else:
                params['task_type'] = 'CPU'
        
        elif self.model_type == 'lightgbm':
            # lightgbm with intelligent defaults
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'random_state': 42,
                'early_stopping_rounds': 50,
                'metric': 'auc',
                'verbosity': -1
            }
            
            if not self.force_cpu and self.gpu_available:
                params.update({
                    'device': 'gpu',
                    'gpu_use_dp': True
                })
            else:
                params['device'] = 'cpu'
        
        elif self.model_type == 'xgboost':
            # xgboost with intelligent defaults
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 2),
                'random_state': 42,
                'early_stopping_rounds': 50,
                'eval_metric': 'auc',
                'verbosity': 0
            }
            
            if not self.force_cpu and self.gpu_available:
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
        Optimized objective function with memory management and error handling.
        """
        try:
            # ensure data is cached
            if not self._data_prepared:
                self._prepare_cached_data()
            
            # monitor memory before training
            memory_before = self._monitor_memory_usage()
            
            # get hyperparameters for this trial
            hyperparameters = self._suggest_hyperparameters(trial)
            
            self.logger.info(f"=== TRIAL {trial.number} ===")
            
            # log key hyperparameters more concisely
            key_params = {k: v for k, v in hyperparameters.items() 
                         if k not in ['verbose', 'eval_metric', 'early_stopping_rounds', 'random_seed', 'random_state', 'devices', 'gpu_platform_id', 'gpu_device_id', 'gpu_id', 'use_best_model', 'verbosity']}
            self.logger.info(f"Params: {json.dumps(key_params, indent=None)}")
            
            # create fresh trainer instance to avoid state issues
            trainer = GateTrainer(
                train_path=self.train_path,
                val_path=self.val_path,
                test_path=self.test_path,
                output_dir=str(self.output_dir),
                target_col=self.target_col,
                feature_groups=self.feature_groups
            )
            
            # reuse cached data preparation
            trainer.dataset = self._cached_trainer.dataset
            
            # train model
            start_time = time.time()
            trainer.train(
                model_type=self.model_type,
                hyperparameters=hyperparameters
            )
            training_time = time.time() - start_time
            
            # get validation metrics
            val_metrics = trainer.training_results['metrics']['validation']
            score = val_metrics[self.optimization_metric]
            
            # log results concisely
            self.logger.info(f"Time: {training_time:.1f}s, {self.optimization_metric}: {score:.4f}")
            
            # store trial result
            trial_result = {
                'trial_number': trial.number,
                'hyperparameters': hyperparameters,
                'score': score,
                'training_time': training_time,
                'all_metrics': val_metrics,
                'memory_usage': memory_before,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            # update best model if this is better
            is_better = False
            if self.best_score is None:
                is_better = True
            elif self.optimization_direction == 'maximize' and score > self.best_score:
                is_better = True
            elif self.optimization_direction == 'minimize' and score < self.best_score:
                is_better = True
                
            if is_better:
                improvement = "N/A" if self.best_score is None else f"(+{score - self.best_score:.4f})"
                self.best_score = score
                self.best_params = hyperparameters.copy()
                self.best_model = trainer.model
                
                self.logger.info(f"🏆 NEW BEST: {score:.4f} {improvement}")
            
            # update convergence history
            self.convergence_history.append(score)
            
            # cleanup to free memory
            del trainer
            gc.collect()
            
            return score
            
        except optuna.TrialPruned:
            # trial was pruned, this is expected
            self.logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            # cleanup on error
            gc.collect()
            # return worst possible score so optuna can continue
            return float('-inf') if self.optimization_direction == 'maximize' else float('inf')
    
    def _check_convergence(self, n_trials_lookback: int = 20) -> bool:
        """Check if optimization has converged."""
        if len(self.convergence_history) < n_trials_lookback:
            return False
        
        recent_scores = self.convergence_history[-n_trials_lookback:]
        improvement_threshold = 0.001  # 0.1% improvement threshold
        
        if self.optimization_direction == 'maximize':
            best_recent = max(recent_scores)
            if self.best_score and (best_recent - self.best_score) < improvement_threshold:
                return True
        else:
            best_recent = min(recent_scores)
            if self.best_score and (self.best_score - best_recent) < improvement_threshold:
                return True
        
        return False
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        enable_pruning: bool = True,
        convergence_patience: int = 30
    ) -> Dict[str, Any]:
        """
        Run enhanced hyperparameter optimization.
        
        Args:
            n_trials: Maximum number of trials to run
            timeout: Timeout in seconds (optional)
            study_name: Name for the study (optional)
            enable_pruning: Enable intelligent pruning
            convergence_patience: Stop if no improvement for N trials
            
        Returns:
            Dictionary with optimization results
        """
        # prepare cached data
        self._prepare_cached_data()
        
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"gate_{self.model_type}_opt_{timestamp}"
        
        self.logger.info("=== STARTING OPTIMIZED HYPERPARAMETER SEARCH ===")
        self.logger.info(f"Model: {self.model_type.upper()}")
        self.logger.info(f"Metric: {self.optimization_metric} ({self.optimization_direction})")
        self.logger.info(f"Max trials: {n_trials}")
        self.logger.info(f"Parallel jobs: {self.n_jobs}")
        self.logger.info(f"Pruning: {'Enabled' if enable_pruning else 'Disabled'}")
        if timeout:
            self.logger.info(f"Timeout: {timeout} seconds")
        
        # choose sampler based on number of trials
        if n_trials > 50:
            sampler = CmaEsSampler(seed=42)  # better for many trials
            self.logger.info("Using CMA-ES sampler for extensive search")
        else:
            sampler = TPESampler(seed=42)  # better for few trials  
            self.logger.info("Using TPE sampler for targeted search")
        
        # choose pruner
        if enable_pruning:
            pruner = HyperbandPruner(min_resource=10, max_resource=200, reduction_factor=3)
        else:
            pruner = MedianPruner(n_startup_trials=5)
        
        # create study
        study = optuna.create_study(
            direction=self.optimization_direction,
            study_name=study_name,
            sampler=sampler,
            pruner=pruner
        )
        
        # run optimization
        start_time = time.time()
        
        try:
            if self.n_jobs > 1:
                # parallel optimization
                self.logger.info(f"Running {self.n_jobs} parallel optimization jobs")
                study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=self.n_jobs,
                    show_progress_bar=True
                )
            else:
                # sequential optimization with convergence checking
                trials_without_improvement = 0
                
                for trial_idx in range(n_trials):
                    if timeout and (time.time() - start_time) > timeout:
                        self.logger.info("Optimization stopped due to timeout")
                        break
                    
                    study.optimize(self._objective, n_trials=1)
                    
                    # check convergence
                    if len(self.convergence_history) > convergence_patience:
                        if self._check_convergence(convergence_patience):
                            trials_without_improvement += 1
                            if trials_without_improvement >= convergence_patience // 3:
                                self.logger.info(f"Early stopping: no improvement for {trials_without_improvement} trials")
                                break
                        else:
                            trials_without_improvement = 0
                    
                    # memory check
                    if len(self.convergence_history) % 10 == 0:
                        memory_usage = self._monitor_memory_usage()
                        if memory_usage > 85:
                            self.logger.warning("High memory usage - consider stopping optimization")
                            
        except KeyboardInterrupt:
            self.logger.warning("Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        # get best results
        if study.trials:
            best_trial = study.best_trial
            
            self.logger.info("=== OPTIMIZATION COMPLETED ===")
            self.logger.info(f"Total time: {optimization_time:.1f}s")
            self.logger.info(f"Completed trials: {len(study.trials)}")
            self.logger.info(f"Best {self.optimization_metric}: {best_trial.value:.4f}")
            self.logger.info("Best hyperparameters:")
            for param, value in best_trial.params.items():
                self.logger.info(f"  {param}: {value}")
            
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
                'parallel_jobs': self.n_jobs,
                'convergence_history': self.convergence_history,
                'memory_usage_history': self.memory_usage_history,
                'all_trials': self.trial_results,
                'timestamp': datetime.now().isoformat()
            }
            
            return results
        else:
            raise RuntimeError("No trials completed successfully")
    
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
            model_name = f"gate_{self.model_type}_optimized_{timestamp}"
        
        # create temporary trainer to save model
        temp_trainer = GateTrainer(
            train_path=self.train_path,
            val_path=self.val_path,
            test_path=self.test_path,
            output_dir=str(self.output_dir),
            target_col=self.target_col,
            feature_groups=self.feature_groups
        )
        
        temp_trainer.model = self.best_model
        temp_trainer.training_results = {
            'model_type': self.model_type,
            'hyperparameters': self.best_params,
            'optimization_score': self.best_score,
            'optimization_metric': self.optimization_metric,
            'timestamp': datetime.now().isoformat()
        }
        
        # save using trainer's method
        model_path = temp_trainer.save_model(model_name)
        
        self.logger.info(f"Best model saved to: {model_path}")
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
            filename = f"optimized_results_{self.model_type}_{timestamp}.json"
        
        results_path = self.output_dir / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Optimization results saved to: {results_path}")
        return str(results_path)
    
    def evaluate_best_model_on_test(self) -> Dict[str, Any]:
        """
        Evaluate the best model on test set.
        
        Returns:
            Test metrics dictionary
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run optimization first.")
        
        self.logger.info("=== EVALUATING BEST MODEL ON TEST SET ===")
        
        # use cached trainer
        if not self._data_prepared:
            self._prepare_cached_data()
        
        # get test data
        test_data = self._cached_trainer.dataset.get_test_data()
        test_metrics = self.best_model.get_metrics(test_data)
        
        self.logger.info("Test set performance:")
        for metric, value in test_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimized OPTUNA hyperparameter optimization for Gate models",
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
        help='Maximum number of optimization trials'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        help='Timeout in seconds (optional)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (1 for sequential)'
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
    parser.add_argument(
        '--disable-pruning',
        action='store_true',
        help='Disable intelligent pruning'
    )
    parser.add_argument(
        '--convergence-patience',
        type=int,
        default=30,
        help='Stop if no improvement for N trials'
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
        '--memory-limit-gb',
        type=float,
        help='Memory limit in GB (auto-detected if not provided)'
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
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=== OPTIMIZED ECOBICI GATE MODEL HYPERPARAMETER OPTIMIZATION ===")
    logger.info(f"Model type: {args.model_type.upper()}")
    logger.info(f"Max trials: {args.n_trials}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info(f"Metric: {args.optimization_metric} ({args.optimization_direction})")
    logger.info(f"GPU: {'Disabled' if args.force_cpu else 'Auto-detect'}")
    if args.memory_limit_gb:
        logger.info(f"Memory limit: {args.memory_limit_gb} GB")
    
    # create optimizer
    optimizer = OptimizedGateHyperparameterOptimizer(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        target_col=args.target_col,
        feature_groups=args.feature_groups,
        force_cpu=args.force_cpu,
        optimization_metric=args.optimization_metric,
        optimization_direction=args.optimization_direction,
        n_jobs=args.n_jobs,
        memory_limit_gb=args.memory_limit_gb,
        logger=logger
    )
    
    # run optimization
    try:
        results = optimizer.optimize(
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=args.study_name,
            enable_pruning=not args.disable_pruning,
            convergence_patience=args.convergence_patience
        )
        
        # save optimization results
        results_path = optimizer.save_optimization_results(results, args.results_filename)
        
        # save best model
        if not args.no_save_model:
            model_path = optimizer.save_best_model(args.model_name)
            logger.info(f"✅ Best model saved to: {model_path}")
        
        # evaluate best model on test set
        if not args.no_test_eval:
            test_metrics = optimizer.evaluate_best_model_on_test()
            
            # add test metrics to results
            results['test_metrics'] = test_metrics
            # re-save results with test metrics
            optimizer.save_optimization_results(results, args.results_filename)
        
        logger.info("🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Results: {results_path}")
        
    except KeyboardInterrupt:
        logger.warning("❌ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Optimization failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 