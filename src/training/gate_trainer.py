"""
Gate model training pipeline with model factory pattern.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from src.dataset.gate_dataset import GateDataset
from src.models.catboost_model import CatBoostModel
from src.models.gbdt_models import LightGBMModel, XGBoostModel


class ModelFactory:
    """Factory for creating different model types."""
    
    @staticmethod
    def create_model(model_type: str, model_config: Dict[str, Any]):
        """
        Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model ('catboost', 'lightgbm', 'xgboost')
            model_config: Configuration dictionary for the model
            
        Returns:
            Model instance
        """
        if model_type.lower() == 'catboost':
            return CatBoostModel(
                model_type=model_config.get('task_type', 'classifier'),
                cat_features=model_config.get('cat_features', None)
            )
        elif model_type.lower() == 'lightgbm':
            return LightGBMModel(
                model_type=model_config.get('task_type', 'classifier')
            )
        elif model_type.lower() == 'xgboost':
            return XGBoostModel(
                model_type=model_config.get('task_type', 'classifier')
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: catboost, lightgbm, xgboost")


class GateTrainer:
    """
    Main training class for Gate models.
    """
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        output_dir: str = "models/gate",
        target_col: str = 'has_activity',
        feature_groups: Optional[List[str]] = None
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.output_dir = Path(output_dir)
        self.target_col = target_col
        self.feature_groups = feature_groups
        
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
        
    def prepare_data(self):
        """Load and prepare the dataset."""
        print("=== Preparing Data ===")
        
        # load raw data
        self.dataset.load_data()
        
        # create binary targets
        self.dataset.create_targets()
        
        # select features
        self.dataset.select_features()
        
        # print summary
        self.dataset.print_data_summary()
        
        return self
    
    def train(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Train a Gate model.
        
        Args:
            model_type: Type of model to train ('catboost', 'lightgbm', 'xgboost')
            hyperparameters: Hyperparameters for the model
            model_config: Model configuration (task_type, etc.)
        """
        print(f"\n=== Training {model_type.upper()} Gate Model ===")
        
        # default configurations
        if model_config is None:
            model_config = {'task_type': 'classifier'}
        
        if hyperparameters is None:
            hyperparameters = self._get_default_hyperparameters(model_type)
        
        # create model
        self.model = ModelFactory.create_model(model_type, model_config)
        
        # set hyperparameters
        self.model.set_hps(hyperparameters)
        
        # prepare data for the specific model type
        train_data, val_data, test_data = self.dataset.prepare_dataset(self.model)
        
        # train model
        start_time = time.time()
        print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.model.train(train_data, val_data)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # evaluate on all splits
        print("\n=== Evaluation Results ===")
        
        train_metrics = self.model.get_metrics(train_data)
        val_metrics = self.model.get_metrics(val_data)
        test_metrics = self.model.get_metrics(test_data)
        
        # store results
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
            'timestamp': datetime.now().isoformat()
        }
        
        # print metrics
        self._print_metrics(train_metrics, val_metrics, test_metrics)
        
        return self
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for each model type."""
        defaults = {
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'eval_metric': 'AUC',
                'early_stopping_rounds': 50,
                'verbose': False
            },
            'lightgbm': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'early_stopping_rounds': 50,
                'metric': 'auc'
            },
            'xgboost': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'early_stopping_rounds': 50,
                'eval_metric': 'auc'
            }
        }
        
        return defaults.get(model_type.lower(), {})
    
    def _print_metrics(self, train_metrics: Dict, val_metrics: Dict, test_metrics: Dict):
        """Print formatted metrics."""
        print("Train Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    def save_model(self, model_name: Optional[str] = None):
        """Save the trained model and results."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"gate_{self.training_results['model_type']}_{timestamp}"
        
        # save model
        model_path = self.output_dir / f"{model_name}.model"
        self.model.save(str(model_path))
        
        # save training results
        results_path = self.output_dir / f"{model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # save feature info
        feature_info_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_info_path, 'w') as f:
            json.dump(self.dataset.get_feature_info(), f, indent=2)
        
        print(f"\nModel saved:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        print(f"  Features: {feature_info_path}")
        
        return model_path
    
    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Get feature importance from the trained model."""
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
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def run_experiment(
        self,
        experiments_config: Dict[str, Dict[str, Any]],
        save_models: bool = True
    ):
        """
        Run multiple training experiments.
        
        Args:
            experiments_config: Dictionary with experiment configurations
            save_models: Whether to save trained models
            
        Example:
            experiments = {
                'catboost_default': {
                    'model_type': 'catboost',
                    'hyperparameters': {'iterations': 500}
                },
                'lightgbm_tuned': {
                    'model_type': 'lightgbm',
                    'hyperparameters': {'n_estimators': 800, 'learning_rate': 0.05}
                }
            }
        """
        print("=== Running Multiple Experiments ===")
        
        all_results = {}
        
        for exp_name, exp_config in experiments_config.items():
            print(f"\n--- Experiment: {exp_name} ---")
            
            # train model
            self.train(
                model_type=exp_config['model_type'],
                hyperparameters=exp_config.get('hyperparameters'),
                model_config=exp_config.get('model_config')
            )
            
            # save results
            all_results[exp_name] = self.training_results.copy()
            
            # save model if requested
            if save_models:
                self.save_model(exp_name)
            
            # get feature importance
            self.get_feature_importance()
        
        # save experiment summary
        summary_path = self.output_dir / f"experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nExperiment summary saved to: {summary_path}")
        
        # print comparison
        self._print_experiment_comparison(all_results)
        
        return all_results
    
    def _print_experiment_comparison(self, all_results: Dict):
        """Print comparison of experiment results."""
        print("\n=== Experiment Comparison ===")
        
        comparison_data = []
        for exp_name, results in all_results.items():
            val_metrics = results['metrics']['validation']
            test_metrics = results['metrics']['test']
            
            comparison_data.append({
                'Experiment': exp_name,
                'Model': results['model_type'],
                'Val_F1': val_metrics.get('f1_score', 0),
                'Val_AUC': val_metrics.get('auc', 0),
                'Val_Recall': val_metrics.get('recall', 0),
                'Test_F1': test_metrics.get('f1_score', 0),
                'Test_AUC': test_metrics.get('auc', 0),
                'Test_Recall': test_metrics.get('recall', 0),
                'Training_Time': results['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.4f')) 