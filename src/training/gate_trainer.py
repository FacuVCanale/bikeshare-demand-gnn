"""
Optimized Gate model training pipeline with enhanced performance and memory management.
"""
import json
import time
import gc
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from src.dataset.gate_dataset import GateDataset
from src.models.catboost_model import CatBoostModel
from src.models.gbdt_models import LightGBMModel, XGBoostModel


def detect_gpu_availability():
    """
    Enhanced GPU detection with detailed capability reporting.
    
    Returns:
        Dict with comprehensive GPU availability information
    """
    gpu_info = {
        'catboost': False,
        'lightgbm': False,  
        'xgboost': False,
        'cuda_available': False,
        'mps_available': False,
        'device_count': 0,
        'device_names': [],
        'memory_gb': 0
    }
    
    # check CUDA availability with memory info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_names'] = [torch.cuda.get_device_name(i) for i in range(gpu_info['device_count'])]
            gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA detected: {gpu_info['device_count']} device(s)")
            print(f"   Primary GPU: {gpu_info['device_names'][0]} ({gpu_info['memory_gb']:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['mps_available'] = True
            print("✅ Apple Metal Performance Shaders (MPS) detected")
        else:
            print("⚠️  No GPU acceleration detected")
    except ImportError:
        print("⚠️  PyTorch not available for GPU detection")
    
    # check framework-specific GPU support
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            gpu_info['catboost'] = True
            print(f"✅ CatBoost GPU support: {get_gpu_device_count()} devices")
    except Exception:
        print("⚠️  CatBoost GPU support not available")
    
    # lightgbm and xgboost gpu depends on compilation and CUDA
    if gpu_info['cuda_available']:
        gpu_info['lightgbm'] = True  # assume available if CUDA exists
        gpu_info['xgboost'] = True
        print("✅ LightGBM/XGBoost GPU support assumed available (CUDA detected)")
    else:
        print("⚠️  LightGBM/XGBoost GPU support not available (no CUDA)")
    
    return gpu_info


class ModelFactory:
    """Enhanced factory for creating different model types with optimized configurations."""
    
    @staticmethod
    def create_model(model_type: str, model_config: Dict[str, Any]):
        """
        Create an optimized model instance based on type and configuration.
        
        Args:
            model_type: Type of model ('catboost', 'lightgbm', 'xgboost')
            model_config: Configuration dictionary for the model
            
        Returns:
            Model instance
        """
        model_type = model_type.lower()
        
        if model_type == 'catboost':
            return CatBoostModel(
                model_type=model_config.get('task_type', 'classifier'),
                cat_features=model_config.get('cat_features', None)
            )
        elif model_type == 'lightgbm':
            return LightGBMModel(
                model_type=model_config.get('task_type', 'classifier')
            )
        elif model_type == 'xgboost':
            return XGBoostModel(
                model_type=model_config.get('task_type', 'classifier')
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: catboost, lightgbm, xgboost")


class OptimizedGateTrainer:
    """
    Enhanced Gate model trainer with performance optimizations:
    - Improved memory management
    - Better GPU utilization  
    - Faster data loading
    - Enhanced error handling
    - Training monitoring
    """
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        output_dir: str = "models/gate",
        target_col: str = 'has_activity',
        feature_groups: Optional[List[str]] = None,
        memory_efficient: bool = True,
        enable_early_stopping: bool = True
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.output_dir = Path(output_dir)
        self.target_col = target_col
        self.feature_groups = feature_groups
        self.memory_efficient = memory_efficient
        self.enable_early_stopping = enable_early_stopping
        
        # create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize dataset
        self.dataset = GateDataset(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            target_col=target_col,
            feature_groups=feature_groups
        )
        
        # training state
        self.model = None
        self.training_results = {}
        self.gpu_info = None
        
        # performance tracking
        self.training_start_time = None
        self.memory_usage_before = None
        self.memory_usage_after = None
        
    def prepare_data(self, verbose: bool = True):
        """Load and prepare the dataset with memory monitoring."""
        if verbose:
            print("=== Preparing Data ===")
        
        # monitor memory before data loading
        if self.memory_efficient:
            import psutil
            self.memory_usage_before = psutil.virtual_memory().percent
            if verbose:
                print(f"Memory usage before loading: {self.memory_usage_before:.1f}%")
        
        # check GPU availability
        if verbose:
            print("\n=== GPU Detection ===")
        self.gpu_info = detect_gpu_availability()
        
        # load raw data
        self.dataset.load_data()
        
        # create binary targets
        self.dataset.create_targets()
        
        # select features
        self.dataset.select_features()
        
        # monitor memory after data loading
        if self.memory_efficient:
            self.memory_usage_after = psutil.virtual_memory().percent
            if verbose:
                print(f"Memory usage after loading: {self.memory_usage_after:.1f}%")
                print(f"Memory increase: {self.memory_usage_after - self.memory_usage_before:.1f}%")
            
            if self.memory_usage_after > 85:
                warnings.warn("High memory usage detected. Consider reducing dataset size or enabling memory optimizations.")
        
        # print summary
        if verbose:
            self.dataset.print_data_summary()
        
        # force garbage collection if memory efficient mode is enabled
        if self.memory_efficient:
            gc.collect()
        
        return self
    
    def train(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Train a Gate model with enhanced performance monitoring.
        
        Args:
            model_type: Type of model to train ('catboost', 'lightgbm', 'xgboost')
            hyperparameters: Hyperparameters for the model
            model_config: Model configuration (task_type, etc.)
            verbose: Enable verbose output
        """
        if verbose:
            print(f"\n=== Training {model_type.upper()} Gate Model ===")
        
        self.training_start_time = time.time()
        
        # default configurations
        if model_config is None:
            model_config = {'task_type': 'classifier'}
        
        if hyperparameters is None:
            hyperparameters = self._get_optimized_hyperparameters(model_type)
        
        # create model
        self.model = ModelFactory.create_model(model_type, model_config)
        
        # set hyperparameters
        self.model.set_hps(hyperparameters)
        
        # prepare data for the specific model type
        try:
            if model_type.lower() == 'catboost':
                train_data, val_data, test_data = self.dataset.prepare_for_catboost(self.model)
            else:
                train_data, val_data, test_data = self.dataset.prepare_for_gbdt(self.model)
        except Exception as e:
            raise RuntimeError(f"Data preparation failed for {model_type}: {str(e)}")
        
        # train model with error handling
        start_time = time.time()
        if verbose:
            print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            self.model.train(train_data, val_data)
        except Exception as e:
            # try fallback to CPU if GPU training fails
            if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                warnings.warn(f"GPU training failed ({str(e)}), attempting CPU fallback")
                # modify hyperparameters for CPU
                cpu_hyperparameters = self._force_cpu_hyperparameters(hyperparameters, model_type)
                self.model.set_hps(cpu_hyperparameters)
                self.model.train(train_data, val_data)
            else:
                raise RuntimeError(f"Model training failed: {str(e)}")
        
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        # evaluate on all splits
        if verbose:
            print("\n=== Evaluation Results ===")
        
        try:
            train_metrics = self.model.get_metrics(train_data)
            val_metrics = self.model.get_metrics(val_data)
            test_metrics = self.model.get_metrics(test_data)
        except Exception as e:
            raise RuntimeError(f"Model evaluation failed: {str(e)}")
        
        # store comprehensive results
        self.training_results = {
            'model_type': model_type,
            'model_config': model_config,
            'hyperparameters': hyperparameters,
            'training_time': training_time,
            'feature_info': self.dataset.get_feature_info(),
            'metrics': {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            },
            'gpu_info': self.gpu_info,
            'memory_usage': {
                'before_loading': self.memory_usage_before,
                'after_loading': self.memory_usage_after
            } if self.memory_efficient else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # print metrics
        if verbose:
            self._print_metrics(train_metrics, val_metrics, test_metrics)
        
        # memory cleanup
        if self.memory_efficient:
            gc.collect()
        
        return self
    
    def _get_optimized_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get optimized default hyperparameters for each model type."""
        # ensure GPU info is available
        if self.gpu_info is None:
            self.gpu_info = detect_gpu_availability()
        
        defaults = {
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'border_count': 128,  # optimized border count
                'random_seed': 42,
                'eval_metric': 'AUC',
                'early_stopping_rounds': 100 if self.enable_early_stopping else None,
                'verbose': False,
                'use_best_model': True
            },
            'lightgbm': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 63,  # 2^depth - 1
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'early_stopping_rounds': 100 if self.enable_early_stopping else None,
                'metric': 'auc',
                'verbosity': -1
            },
            'xgboost': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'early_stopping_rounds': 100 if self.enable_early_stopping else None,
                'eval_metric': 'auc',
                'verbosity': 0
            }
        }
        
        model_defaults = defaults.get(model_type.lower(), {})
        
        # add optimized GPU configurations based on detection
        if model_type.lower() == 'catboost' and self.gpu_info.get('catboost', False):
            model_defaults.update({
                'task_type': 'GPU',
                'devices': '0',
                'gpu_ram_part': 0.9  # use 90% of GPU memory
            })
            print("🚀 CatBoost configured for optimized GPU training")
        
        elif model_type.lower() == 'lightgbm' and self.gpu_info.get('lightgbm', False):
            model_defaults.update({
                'device': 'gpu',
                'gpu_use_dp': True,  # use double precision for better accuracy
                'max_bin': 255  # optimized for GPU
            })
            print("🚀 LightGBM configured for optimized GPU training")
        
        elif model_type.lower() == 'xgboost' and self.gpu_info.get('xgboost', False):
            model_defaults.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'max_bin': 256  # optimized for GPU
            })
            print("🚀 XGBoost configured for optimized GPU training")
        
        return model_defaults
    
    def _force_cpu_hyperparameters(self, hyperparameters: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Force CPU training by modifying hyperparameters."""
        cpu_hyperparameters = hyperparameters.copy()
        
        if model_type.lower() == 'catboost':
            cpu_hyperparameters['task_type'] = 'CPU'
            cpu_hyperparameters.pop('devices', None)
            cpu_hyperparameters.pop('gpu_ram_part', None)
        
        elif model_type.lower() == 'lightgbm':
            cpu_hyperparameters['device'] = 'cpu'
            cpu_hyperparameters.pop('gpu_use_dp', None)
            cpu_hyperparameters.pop('gpu_platform_id', None)
            cpu_hyperparameters.pop('gpu_device_id', None)
        
        elif model_type.lower() == 'xgboost':
            cpu_hyperparameters['tree_method'] = 'hist'
            cpu_hyperparameters.pop('gpu_id', None)
        
        return cpu_hyperparameters
    
    def _print_metrics(self, train_metrics: Dict, val_metrics: Dict, test_metrics: Dict):
        """Print formatted metrics with highlights."""
        print("📊 Training Results:")
        
        print("\n📈 Train Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n🎯 Validation Metrics:")
        for metric, value in val_metrics.items():
            if metric in ['f1_score', 'auc']:
                print(f"  {metric}: {value:.4f} ⭐")  # highlight key metrics
            else:
                print(f"  {metric}: {value:.4f}")
        
        print("\n🧪 Test Metrics:")
        for metric, value in test_metrics.items():
            if metric in ['f1_score', 'auc']:
                print(f"  {metric}: {value:.4f} ⭐")
            else:
                print(f"  {metric}: {value:.4f}")
    
    def save_model(self, model_name: Optional[str] = None):
        """Save the trained model and comprehensive results."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"gate_{self.training_results['model_type']}_{timestamp}"
        
        # save model
        model_path = self.output_dir / f"{model_name}.model"
        self.model.save(str(model_path))
        
        # save comprehensive training results
        results_path = self.output_dir / f"{model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # save feature info
        feature_info_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_info_path, 'w') as f:
            json.dump(self.dataset.get_feature_info(), f, indent=2)
        
        print(f"\n💾 Model saved:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        print(f"  Features: {feature_info_path}")
        
        return str(model_path)
    
    def get_feature_importance(self, top_n: int = 20, plot: bool = False) -> Optional[pd.DataFrame]:
        """Get feature importance from the trained model with optional plotting."""
        if self.model is None:
            print("No trained model available.")
            return None
        
        importance = self.model.get_feature_importance()
        if importance is None:
            print("Feature importance not available for this model.")
            return None
        
        feature_names = self.dataset.get_feature_info()['feature_names']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        # optional plotting
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                top_features = importance_df.head(top_n)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n} Feature Importance - {self.training_results["model_type"]}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                # save plot
                plot_path = self.output_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"📊 Feature importance plot saved to: {plot_path}")
                plt.show()
            except ImportError:
                print("⚠️  Matplotlib not available for plotting")
        
        return importance_df
    
    def run_experiment(
        self,
        experiments_config: Dict[str, Dict[str, Any]],
        save_models: bool = True,
        verbose: bool = True
    ):
        """
        Run multiple training experiments with enhanced reporting.
        
        Args:
            experiments_config: Dictionary with experiment configurations
            save_models: Whether to save trained models
            verbose: Enable verbose output
            
        Example:
            experiments = {
                'catboost_baseline': {
                    'model_type': 'catboost',
                    'hyperparameters': {'iterations': 500}
                },
                'lightgbm_tuned': {
                    'model_type': 'lightgbm',
                    'hyperparameters': {'n_estimators': 800, 'learning_rate': 0.05}
                }
            }
        """
        if verbose:
            print("=== Running Enhanced Experiments ===")
        
        all_results = {}
        experiment_times = {}
        
        for exp_name, exp_config in experiments_config.items():
            if verbose:
                print(f"\n--- Experiment: {exp_name} ---")
            
            exp_start_time = time.time()
            
            # train model
            self.train(
                model_type=exp_config['model_type'],
                hyperparameters=exp_config.get('hyperparameters'),
                model_config=exp_config.get('model_config'),
                verbose=verbose
            )
            
            exp_time = time.time() - exp_start_time
            experiment_times[exp_name] = exp_time
            
            # store results
            all_results[exp_name] = self.training_results.copy()
            
            # save model if requested
            if save_models:
                self.save_model(exp_name)
            
            # get feature importance
            if verbose:
                self.get_feature_importance()
            
            # memory cleanup between experiments
            if self.memory_efficient:
                gc.collect()
        
        # save comprehensive experiment summary
        summary_data = {
            'experiments': all_results,
            'experiment_times': experiment_times,
            'total_experiments': len(experiments_config),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / f"experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        if verbose:
            print(f"\n📋 Experiment summary saved to: {summary_path}")
        
        # print enhanced comparison
        if verbose:
            self._print_enhanced_experiment_comparison(all_results, experiment_times)
        
        return all_results
    
    def _print_enhanced_experiment_comparison(self, all_results: Dict, experiment_times: Dict):
        """Print enhanced comparison of experiment results."""
        print("\n=== Enhanced Experiment Comparison ===")
        
        comparison_data = []
        for exp_name, results in all_results.items():
            val_metrics = results['metrics']['validation']
            test_metrics = results['metrics']['test']
            
            comparison_data.append({
                'Experiment': exp_name,
                'Model': results['model_type'].upper(),
                'Time(s)': f"{experiment_times.get(exp_name, 0):.1f}",
                'Val_F1': f"{val_metrics.get('f1_score', 0):.4f}",
                'Val_AUC': f"{val_metrics.get('auc', 0):.4f}",
                'Val_Recall': f"{val_metrics.get('recall', 0):.4f}",
                'Test_F1': f"{test_metrics.get('f1_score', 0):.4f}",
                'Test_AUC': f"{test_metrics.get('auc', 0):.4f}",
                'Test_Recall': f"{test_metrics.get('recall', 0):.4f}",
                'GPU': "✅" if results.get('gpu_info', {}).get(results['model_type'], False) else "❌"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # highlight best performers
        if len(comparison_data) > 1:
            print("\n🏆 Best Performers:")
            val_f1_scores = [float(row['Val_F1']) for row in comparison_data]
            test_f1_scores = [float(row['Test_F1']) for row in comparison_data]
            
            best_val_idx = np.argmax(val_f1_scores)
            best_test_idx = np.argmax(test_f1_scores)
            
            print(f"  Best Validation F1: {comparison_data[best_val_idx]['Experiment']} ({comparison_data[best_val_idx]['Val_F1']})")
            print(f"  Best Test F1: {comparison_data[best_test_idx]['Experiment']} ({comparison_data[best_test_idx]['Test_F1']})")


# backward compatibility alias
GateTrainer = OptimizedGateTrainer 