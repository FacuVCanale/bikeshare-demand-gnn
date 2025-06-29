"""
Training Module for Individual Station Bike Prediction

Trains XGBoost and LightGBM models on processed delta T data.
Handles both single and multi-target regression for arrivals and departures.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing GBDT models from the project
import sys
sys.path.append('../src')
from models.gbdt_models import XGBoostModel, LightGBMModel


class ModelTrainer:
    """
    Training class for XGBoost and LightGBM models on bike sharing data.
    """
    
    def __init__(self, random_state=42, use_gpu=True):
        """
        Initialize the trainer.
        
        Args:
            random_state (int): Random state for reproducibility
            use_gpu (bool): Whether to enable GPU acceleration (default: True)
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        
    def prepare_data(self, X, y, metadata, test_size=0.2, validation_size=0.2, 
                    scale_features=True, use_time_split=True):
        """
        Prepare data for training with proper time-aware splits.
        
        Args:
            X (pd.DataFrame or pl.DataFrame): Features
            y (pd.DataFrame or pl.DataFrame): Targets  
            metadata (pd.DataFrame or pl.DataFrame): Datetime and station info
            test_size (float): Test set proportion
            validation_size (float): Validation set proportion
            scale_features (bool): Whether to scale features
            use_time_split (bool): Whether to use time-based splitting
            
        Returns:
            dict: Dictionary with train/val/test splits
        """
        print("Preparing data for training...")
        
        # Convert polars to pandas if needed for scikit-learn compatibility
        if isinstance(X, pl.DataFrame):
            print("  Converting polars DataFrames to pandas for sklearn compatibility...")
            X = X.to_pandas()
            y = y.to_pandas()
            metadata = metadata.to_pandas()
        
        if use_time_split:
            # Time-based split to avoid data leakage
            X_with_time = X.copy()
            X_with_time['datetime'] = metadata['datetime']
            X_with_time = X_with_time.sort_values('datetime')
            
            # Split data chronologically
            n_samples = len(X_with_time)
            train_end = int(n_samples * (1 - test_size - validation_size))
            val_end = int(n_samples * (1 - test_size))
            
            train_idx = X_with_time.index[:train_end]
            val_idx = X_with_time.index[train_end:val_end]
            test_idx = X_with_time.index[val_end:]
            
            X_train = X.loc[train_idx].drop(['datetime'], axis=1, errors='ignore')
            X_val = X.loc[val_idx].drop(['datetime'], axis=1, errors='ignore')
            X_test = X.loc[test_idx].drop(['datetime'], axis=1, errors='ignore')
            
            y_train = y.loc[train_idx]
            y_val = y.loc[val_idx]
            y_test = y.loc[test_idx]
            
            meta_train = metadata.loc[train_idx]
            meta_val = metadata.loc[val_idx]
            meta_test = metadata.loc[test_idx]
            
        else:
            # Random split
            X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
                X, y, metadata, test_size=test_size, random_state=self.random_state
            )
            
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
                X_temp, y_temp, meta_temp, test_size=val_size_adjusted, 
                random_state=self.random_state
            )
        
        # Feature scaling
        if scale_features:
            print("Scaling features...")
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val), 
                columns=X_val.columns, 
                index=X_val.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
            self.scalers['feature_scaler'] = scaler
            X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
        
        # Prepare data splits
        data_splits = {
            'train': {
                'X': X_train,
                'y': y_train,
                'metadata': meta_train
            },
            'val': {
                'X': X_val,
                'y': y_val,
                'metadata': meta_val
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'metadata': meta_test
            }
        }
        
        print(f"Data splits prepared:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        print(f"  Features: {X_train.shape[1]}")
        
        return data_splits
    
    def get_model_hyperparameters(self, model_type, target_type='arrivals', use_gpu=True):
        """
        Get optimized hyperparameters for different models.
        
        Args:
            model_type (str): 'xgboost' or 'lightgbm'
            target_type (str): 'arrivals' or 'departures'
            use_gpu (bool): Whether to enable GPU acceleration (default: True)
            
        Returns:
            dict: Hyperparameters
        """
        if model_type == 'xgboost':
            base_params = {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'early_stopping_rounds': 100,
                'n_jobs': -1
            }
            # GPU-specific parameters for XGBoost
            if use_gpu:
                base_params.update({
                    'tree_method': 'gpu_hist',  # Enable GPU histogram algorithm
                    'gpu_id': 0,  # Use first GPU
                })
            return base_params
            
        elif model_type == 'lightgbm':
            base_params = {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'early_stopping_rounds': 100,
                'n_jobs': -1,
                'verbosity': -1
            }
            # GPU-specific parameters for LightGBM
            if use_gpu:
                base_params.update({
                    'device': 'gpu',  # Enable GPU device
                    'gpu_use_dp': True,  # Use double precision on GPU
                })
            return base_params
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_single_target(self, data_splits, target_col, model_types=['xgboost', 'lightgbm']):
        """
        Train models for a single target (arrivals or departures).
        
        Args:
            data_splits (dict): Data splits from prepare_data
            target_col (str): Target column name ('arrivals' or 'departures')
            model_types (list): List of model types to train
            
        Returns:
            dict: Trained models and their performance
        """
        print(f"\nTraining models for target: {target_col}")
        
        results = {}
        
        # Extract single target
        y_train = data_splits['train']['y'][target_col]
        y_val = data_splits['val']['y'][target_col]
        y_test = data_splits['test']['y'][target_col]
        
        X_train = data_splits['train']['X']
        X_val = data_splits['val']['X']
        X_test = data_splits['test']['X']
        
        for model_type in model_types:
            print(f"  Training {model_type}...")
            
            # Initialize model
            if model_type == 'xgboost':
                model = XGBoostModel(model_type='regressor')
            elif model_type == 'lightgbm':
                model = LightGBMModel(model_type='regressor')
            else:
                print(f"    Unknown model type: {model_type}, skipping...")
                continue
            
            # Set hyperparameters (with GPU configuration)
            hps = self.get_model_hyperparameters(model_type, target_col, self.use_gpu)
            model.set_hps(hps)
            
            # Print GPU status
            if self.use_gpu:
                gpu_status = "GPU enabled" if hasattr(model, '_gpu_available') and model._gpu_available else "GPU requested but not available"
                print(f"    {gpu_status}")
            
            # Train model
            start_time = datetime.now()
            try:
                model.train(
                    train_data=(X_train, y_train),
                    valid_data=(X_val, y_val)
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate on all splits
                train_metrics = model.get_metrics((X_train, y_train))
                val_metrics = model.get_metrics((X_val, y_val))
                test_metrics = model.get_metrics((X_test, y_test))
                
                # Store results
                model_key = f"{model_type}_{target_col}"
                results[model_key] = {
                    'model': model,
                    'target': target_col,
                    'training_time': training_time,
                    'metrics': {
                        'train': train_metrics,
                        'val': val_metrics,
                        'test': test_metrics
                    },
                    'hyperparameters': hps
                }
                
                # Store in main models dict
                self.models[model_key] = model
                
                print(f"    {model_type} completed in {training_time:.2f}s")
                print(f"    Validation RMSE: {val_metrics['rmse']:.4f}")
                print(f"    Test RMSE: {test_metrics['rmse']:.4f}")
                
            except Exception as e:
                print(f"    Error training {model_type}: {str(e)}")
                continue
        
        return results
    
    def train_multi_target(self, data_splits, target_cols=['arrivals', 'departures'], 
                          model_types=['xgboost', 'lightgbm']):
        """
        Train models for multiple targets.
        
        Args:
            data_splits (dict): Data splits from prepare_data
            target_cols (list): Target column names
            model_types (list): List of model types to train
            
        Returns:
            dict: All trained models and their performance
        """
        print("Starting multi-target training...")
        
        all_results = {}
        
        # Train separate model for each target
        for target_col in target_cols:
            target_results = self.train_single_target(
                data_splits, target_col, model_types
            )
            all_results.update(target_results)
        
        # Store training history
        self.training_history = {
            'timestamp': datetime.now().isoformat(),
            'targets': target_cols,
            'model_types': model_types,
            'results': {}
        }
        
        # Process results for history
        for model_key, result in all_results.items():
            self.training_history['results'][model_key] = {
                'target': result['target'],
                'training_time': result['training_time'],
                'final_metrics': result['metrics'],
                'hyperparameters': result['hyperparameters']
            }
        
        print(f"\nTraining completed for {len(all_results)} models")
        return all_results
    
    def get_best_models(self, metric='rmse', split='val'):
        """
        Get the best performing models for each target.
        
        Args:
            metric (str): Metric to use for comparison
            split (str): Data split to evaluate on
            
        Returns:
            dict: Best models for each target
        """
        if not self.training_history:
            print("No training history available")
            return {}
        
        best_models = {}
        targets = set()
        
        # Group by target
        for model_key, result in self.training_history['results'].items():
            targets.add(result['target'])
        
        for target in targets:
            target_models = {}
            for model_key, result in self.training_history['results'].items():
                if result['target'] == target:
                    target_models[model_key] = result['final_metrics'][split][metric]
            
            if target_models:
                best_model_key = min(target_models.keys(), key=lambda k: target_models[k])
                best_models[target] = {
                    'model_key': best_model_key,
                    'model': self.models[best_model_key],
                    'metric_value': target_models[best_model_key]
                }
                
                print(f"Best model for {target}: {best_model_key} "
                      f"({metric}={target_models[best_model_key]:.4f})")
        
        return best_models
    
    def save_models(self, save_dir='models', include_scalers=True):
        """
        Save trained models and scalers.
        
        Args:
            save_dir (str): Directory to save models
            include_scalers (bool): Whether to save scalers
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_key, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_key}.joblib")
            model.save(model_path)
            print(f"Saved model: {model_path}")
        
        # Save scalers
        if include_scalers and self.scalers:
            for scaler_key, scaler in self.scalers.items():
                scaler_path = os.path.join(save_dir, f"{scaler_key}.joblib")
                joblib.dump(scaler, scaler_path)
                print(f"Saved scaler: {scaler_path}")
        
        # Save training history
        if self.training_history:
            history_path = os.path.join(save_dir, "training_history.json")
            import json
            with open(history_path, 'w') as f:
                # Convert any non-serializable objects to strings
                history_copy = self.training_history.copy()
                json.dump(history_copy, f, indent=2, default=str)
            print(f"Saved training history: {history_path}")
    
    def load_models(self, save_dir='models'):
        """
        Load saved models and scalers.
        
        Args:
            save_dir (str): Directory containing saved models
        """
        if not os.path.exists(save_dir):
            print(f"Save directory {save_dir} does not exist")
            return
        
        # Load models
        for filename in os.listdir(save_dir):
            if filename.endswith('.joblib') and not filename.endswith('_scaler.joblib'):
                model_key = filename.replace('.joblib', '')
                model_path = os.path.join(save_dir, filename)
                
                # Determine model type
                if 'xgboost' in model_key:
                    model = XGBoostModel(model_type='regressor')
                elif 'lightgbm' in model_key:
                    model = LightGBMModel(model_type='regressor')
                else:
                    continue
                
                model.load(model_path)
                self.models[model_key] = model
                print(f"Loaded model: {model_key}")
        
        # Load scalers
        scaler_files = [f for f in os.listdir(save_dir) if f.endswith('_scaler.joblib')]
        for filename in scaler_files:
            scaler_key = filename.replace('.joblib', '')
            scaler_path = os.path.join(save_dir, filename)
            scaler = joblib.load(scaler_path)
            self.scalers[scaler_key] = scaler
            print(f"Loaded scaler: {scaler_key}")
        
        # Load training history
        history_path = os.path.join(save_dir, "training_history.json")
        if os.path.exists(history_path):
            import json
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            print("Loaded training history")


def train_pipeline(X, y, metadata, target_cols=['arrivals', 'departures'], 
                  model_types=['xgboost', 'lightgbm'], save_models=True, 
                  save_dir='models', use_gpu=True):
    """
    Complete training pipeline function.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Targets
        metadata (pd.DataFrame): Metadata
        target_cols (list): Target columns to train on
        model_types (list): Model types to train
        save_models (bool): Whether to save trained models
        save_dir (str): Directory to save models
        use_gpu (bool): Whether to enable GPU acceleration (default: True)
        
    Returns:
        tuple: (trainer, results, best_models)
    """
    print("=== STARTING TRAINING PIPELINE ===")
    gpu_status = "with GPU acceleration" if use_gpu else "CPU only"
    print(f"Training models {gpu_status}")
    
    # Initialize trainer
    trainer = ModelTrainer(use_gpu=use_gpu)
    
    # Prepare data
    data_splits = trainer.prepare_data(X, y, metadata)
    
    # Train models
    results = trainer.train_multi_target(data_splits, target_cols, model_types)
    
    # Get best models
    best_models = trainer.get_best_models()
    
    # Save models if requested
    if save_models:
        trainer.save_models(save_dir)
    
    print("=== TRAINING PIPELINE COMPLETE ===")
    
    return trainer, results, best_models


if __name__ == "__main__":
    print("Training Module for Individual Station Bike Prediction")
    print("Use train_pipeline() to train models on your processed data") 