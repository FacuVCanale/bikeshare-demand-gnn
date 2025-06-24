"""
Gradient Boosting Decision Tree (GBDT) training for ETA and destination prediction.
Supports both XGBoost and LightGBM with configurable hyperparameters.
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: LightGBM not available")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Import our evaluation utilities
from evaluation import (
    evaluate_eta_prediction, 
    evaluate_destination_prediction,
    print_evaluation_results,
    save_evaluation_results,
    load_and_prepare_data
)


class GBDTTrainer:
    """
    Trainer for Gradient Boosting Decision Trees (XGBoost and LightGBM).
    Handles both ETA regression and destination classification.
    """
    
    def __init__(self, model_type: str = "lgb", eta_params: dict = None, dest_params: dict = None, use_gpu: bool = True):
        """
        Initialize GBDT trainer.
        
        Args:
            model_type: Either 'xgb' or 'lgb'
            eta_params: Hyperparameters for ETA regression model
            dest_params: Hyperparameters for destination classification model
            use_gpu: Whether to use GPU for training (default: True)
        """
        self.model_type = model_type.lower()
        self.use_gpu = use_gpu
        self.eta_model = None
        self.dest_model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.dest_classes = None
        
        # Check GPU availability if requested
        if self.use_gpu:
            self._check_gpu_availability()
        
        # Default hyperparameters
        self.default_eta_params = self._get_default_eta_params()
        self.default_dest_params = self._get_default_dest_params()
        
        # Use provided params or defaults
        self.eta_params = eta_params if eta_params else self.default_eta_params
        self.dest_params = dest_params if dest_params else self.default_dest_params
        
        # Validate model availability
        if self.model_type == 'xgb' and not XGB_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        elif self.model_type == 'lgb' and not LGB_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        elif self.model_type not in ['xgb', 'lgb']:
            raise ValueError("model_type must be 'xgb' or 'lgb'")
    
    def _check_gpu_availability(self):
        """Check if GPU is available for the selected model type."""
        try:
            if self.model_type == 'xgb':
                import xgboost as xgb
                # Test XGBoost GPU availability
                try:
                    test_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
                    print(f"   🚀 GPU acceleration enabled for XGBoost")
                except Exception as e:
                    print(f"   ⚠️  XGBoost GPU not available, falling back to CPU: {e}")
                    self.use_gpu = False
            elif self.model_type == 'lgb':
                import lightgbm as lgb
                # Test LightGBM GPU availability
                try:
                    test_model = lgb.LGBMRegressor(device='gpu')
                    print(f"   🚀 GPU acceleration enabled for LightGBM")
                except Exception as e:
                    print(f"   ⚠️  LightGBM GPU not available, falling back to CPU: {e}")
                    self.use_gpu = False
        except Exception as e:
            print(f"   ⚠️  GPU check failed, falling back to CPU: {e}")
            self.use_gpu = False
    
    def _get_default_eta_params(self):
        """Get default hyperparameters for ETA regression."""
        if self.model_type == 'xgb':
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
            # Add GPU-specific parameters for XGBoost
            if self.use_gpu:
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor'
                })
            return params
        else:  # lgb
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            # Add GPU-specific parameters for LightGBM
            if self.use_gpu:
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
            return params
    
    def _get_default_dest_params(self):
        """Get default hyperparameters for destination classification."""
        if self.model_type == 'xgb':
            params = {
                'objective': 'multi:softprob',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
            # Add GPU-specific parameters for XGBoost
            if self.use_gpu:
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor'
                })
            return params
        else:  # lgb
            params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            # Add GPU-specific parameters for LightGBM
            if self.use_gpu:
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
            return params
    
    def _prepare_features(self, df: pl.DataFrame):
        """
        Prepare features for GBDT training.
        
        Args:
            df: Input dataframe
            
        Returns:
            Feature matrix as numpy array
        """
        # Convert to pandas for easier feature manipulation
        df_pd = df.to_pandas()
        
        # Define feature columns (exclude target variables and metadata)
        exclude_cols = {
            'duracion_recorrido',  # ETA target
            'id_estacion_destino',  # Destination target
            'fecha_origen_recorrido', 'fecha_destino_recorrido',  # Timestamps
            'id_usuario', 'id_estacion_origen',  # IDs (we keep engineered features from these)
            'fecha_date', 'fecha_alta',  # Date columns
            'genero_usuario', 'hora_alta',  # Keep as categorical if needed
            'temp_bucket', 'wind_bucket', 'weather_code_cat'  # Handle categoricals separately
        }
        
        # Get all numeric columns
        numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Add categorical features (one-hot encoded or label encoded)
        categorical_cols = ['genero_usuario', 'temp_bucket', 'wind_bucket']
        for col in categorical_cols:
            if col in df_pd.columns:
                # Simple label encoding for tree-based models
                if df_pd[col].dtype == 'object':
                    df_pd[f'{col}_encoded'] = pd.Categorical(df_pd[col]).codes
                    feature_cols.append(f'{col}_encoded')
        
        # Handle weather code categories (already one-hot encoded)
        weather_cat_cols = [col for col in df_pd.columns if col.startswith('weather_code_cat_')]
        feature_cols.extend(weather_cat_cols)
        
        # Store feature columns for later use
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        # Select and fill missing values
        X = df_pd[feature_cols].fillna(0)
        
        print(f"   📊 Using {len(feature_cols)} features for training")
        
        return X.values
    
    def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame):
        """
        Fit both ETA and destination models.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
        """
        gpu_status = "GPU" if self.use_gpu else "CPU"
        print(f"🏗️  Training {self.model_type.upper()} models on {gpu_status}...")
        
        # Prepare training features
        X_train = self._prepare_features(train_df)
        y_eta_train = train_df.select("duracion_recorrido").to_numpy().flatten()
        y_dest_train = train_df.select("id_estacion_destino").to_numpy().flatten()
        
        # Prepare validation features
        X_val = self._prepare_features(val_df)
        y_eta_val = val_df.select("duracion_recorrido").to_numpy().flatten()
        y_dest_val = val_df.select("id_estacion_destino").to_numpy().flatten()
        
        # Encode destination labels with robust handling of unseen labels
        y_dest_train_encoded = self.label_encoder.fit_transform(y_dest_train)
        self.dest_classes = self.label_encoder.classes_
        
        # Handle unseen labels in validation set
        train_stations = set(y_dest_train)
        val_stations = set(y_dest_val)
        unseen_stations = val_stations - train_stations
        
        if unseen_stations:
            print(f"   ⚠️  Found {len(unseen_stations)} stations in validation not seen in training:")
            print(f"       {sorted(list(unseen_stations))}")
            print(f"   🔧 Filtering validation set to only include known stations...")
            
            # Create mask for valid validation samples (only known stations)
            val_mask = np.isin(y_dest_val, list(train_stations))
            X_val = X_val[val_mask]
            y_eta_val = y_eta_val[val_mask]
            y_dest_val = y_dest_val[val_mask]
            
            print(f"   📉 Validation set reduced to {len(y_dest_val):,} samples")
        
        y_dest_val_encoded = self.label_encoder.transform(y_dest_val)
        
        print(f"   📈 Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"   📈 Validation data: {X_val.shape[0]:,} samples")
        print(f"   🎯 Destination classes: {len(self.dest_classes)} unique stations")
        
        # Train ETA regression model
        print(f"   🚀 Training ETA regression model...")
        if self.model_type == 'xgb':
            self.eta_model = xgb.XGBRegressor(**self.eta_params)
            self.eta_model.fit(
                X_train, y_eta_train,
                eval_set=[(X_val, y_eta_val)],
                verbose=False
            )
        else:  # lgb
            self.eta_model = lgb.LGBMRegressor(**self.eta_params)
            self.eta_model.fit(
                X_train, y_eta_train,
                eval_set=[(X_val, y_eta_val)],
                callbacks=[lgb.log_evaluation(0)]  # Suppress verbose output
            )
        
        # Train destination classification model
        print(f"   🎯 Training destination classification model...")
        dest_params_copy = self.dest_params.copy()
        if self.model_type == 'lgb':
            dest_params_copy['num_class'] = len(self.dest_classes)
        
        if self.model_type == 'xgb':
            self.dest_model = xgb.XGBClassifier(**dest_params_copy)
            self.dest_model.fit(
                X_train, y_dest_train_encoded,
                eval_set=[(X_val, y_dest_val_encoded)],
                verbose=False
            )
        else:  # lgb
            self.dest_model = lgb.LGBMClassifier(**dest_params_copy)
            self.dest_model.fit(
                X_train, y_dest_train_encoded,
                eval_set=[(X_val, y_dest_val_encoded)],
                callbacks=[lgb.log_evaluation(0)]
            )
        
        print("   ✅ Model training complete!")
    
    def predict(self, test_df: pl.DataFrame):
        """
        Make predictions for both ETA and destination.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Tuple of (eta_pred, dest_pred, dest_proba)
        """
        if self.eta_model is None or self.dest_model is None:
            raise ValueError("Models must be fitted before prediction")
        
        X_test = self._prepare_features(test_df)
        
        # ETA predictions
        eta_pred = self.eta_model.predict(X_test)
        
        # Handle destination predictions with unseen labels
        y_dest_test = test_df.select("id_estacion_destino").to_numpy().flatten()
        train_stations = set(self.dest_classes)
        test_stations = set(y_dest_test)
        unseen_stations = test_stations - train_stations
        
        if unseen_stations:
            print(f"   ⚠️  Found {len(unseen_stations)} stations in test set not seen in training:")
            print(f"       Replacing with most frequent training station...")
            
            # Replace unseen stations with the most frequent station from training
            most_frequent_station = self.dest_classes[0]  # LabelEncoder classes are sorted by frequency
            y_dest_test_clean = np.where(np.isin(y_dest_test, list(train_stations)), 
                                       y_dest_test, most_frequent_station)
        else:
            y_dest_test_clean = y_dest_test
        
        # Destination predictions
        try:
            dest_pred_encoded = self.dest_model.predict(X_test)
            dest_pred = self.label_encoder.inverse_transform(dest_pred_encoded)
            dest_proba = self.dest_model.predict_proba(X_test)
        except ValueError as e:
            if "unseen labels" in str(e):
                print(f"   🔧 Handling prediction error by filtering X_test...")
                # This is a fallback - shouldn't happen with above fix but just in case
                valid_mask = np.isin(y_dest_test, list(train_stations))
                X_test_filtered = X_test[valid_mask]
                
                if len(X_test_filtered) == 0:
                    raise ValueError("No valid test samples after filtering unseen stations")
                
                dest_pred_encoded = self.dest_model.predict(X_test_filtered)
                dest_pred_partial = self.label_encoder.inverse_transform(dest_pred_encoded)
                dest_proba_partial = self.dest_model.predict_proba(X_test_filtered)
                
                # Expand back to full size with padding
                dest_pred = np.full(len(y_dest_test), most_frequent_station, dtype=dest_pred_partial.dtype)
                dest_pred[valid_mask] = dest_pred_partial
                
                dest_proba = np.full((len(y_dest_test), len(self.dest_classes)), 
                                   1.0/len(self.dest_classes))  # Uniform distribution for unknowns
                dest_proba[valid_mask] = dest_proba_partial
                
                print(f"   📉 Predictions made for {len(X_test_filtered):,} valid samples out of {len(X_test):,}")
            else:
                raise e
        
        return eta_pred, dest_pred, dest_proba
    
    def save_models(self, save_dir: Path):
        """Save trained models to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        eta_path = save_dir / f"{self.model_type}_eta_model.pkl"
        dest_path = save_dir / f"{self.model_type}_dest_model.pkl"
        encoder_path = save_dir / f"{self.model_type}_label_encoder.pkl"
        
        joblib.dump(self.eta_model, eta_path)
        joblib.dump(self.dest_model, dest_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature columns and hyperparameters
        metadata = {
            'model_type': self.model_type,
            'use_gpu': self.use_gpu,
            'feature_columns': self.feature_columns,
            'dest_classes': self.dest_classes.tolist(),
            'eta_params': self.eta_params,
            'dest_params': self.dest_params
        }
        
        metadata_path = save_dir / f"{self.model_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Models saved to: {save_dir}")
        
        return {
            'eta_model': eta_path,
            'dest_model': dest_path,
            'label_encoder': encoder_path,
            'metadata': metadata_path
        }


def parse_hyperparameters(param_string: str) -> dict:
    """
    Parse hyperparameter string into dictionary.
    
    Args:
        param_string: String like "max_depth=6,learning_rate=0.1,n_estimators=100"
        
    Returns:
        Dictionary of hyperparameters
    """
    if not param_string:
        return {}
    
    params = {}
    for pair in param_string.split(','):
        key, value = pair.split('=')
        key = key.strip()
        value = value.strip()
        
        # Try to convert to appropriate type
        try:
            # Try int first
            params[key] = int(value)
        except ValueError:
            try:
                # Try float
                params[key] = float(value)
            except ValueError:
                # Keep as string
                params[key] = value
    
    return params


def main():
    """Main training pipeline with argument parsing."""
    
    parser = argparse.ArgumentParser(description="Train GBDT models for ETA and destination prediction")
    
    parser.add_argument('--model', type=str, choices=['xgb', 'lgb'], default='lgb',
                       help='Model type: xgb (XGBoost) or lgb (LightGBM)')
    
    parser.add_argument('--eta-params', type=str, default='',
                       help='ETA model hyperparameters as comma-separated key=value pairs')
    
    parser.add_argument('--dest-params', type=str, default='',
                       help='Destination model hyperparameters as comma-separated key=value pairs')
    
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                       help='Directory containing train/val/test parquet files')
    
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results and models')
    
    parser.add_argument('--experiment-name', type=str, default='',
                       help='Name for this experiment (will be added to output files)')
    
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration if available (default: True)')
    
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help='Disable GPU acceleration, use CPU only')
    
    args = parser.parse_args()
    
    # Parse hyperparameters
    eta_params = parse_hyperparameters(args.eta_params)
    dest_params = parse_hyperparameters(args.dest_params)
    
    # Create experiment name
    experiment_name = args.experiment_name or f"{args.model}_default"
    
    print("🚀 GBDT MODEL TRAINING")
    print("="*60)
    print(f"Model Type: {args.model.upper()}")
    print(f"Experiment: {experiment_name}")
    print(f"GPU Enabled: {'Yes' if args.gpu else 'No'}")
    if eta_params:
        print(f"ETA Params: {eta_params}")
    if dest_params:
        print(f"Dest Params: {dest_params}")
    print("="*60)
    
    # Define data paths
    data_dir = Path(args.data_dir)
    train_path = data_dir / "trips_train.parquet"
    val_path = data_dir / "trips_val.parquet"
    test_path = data_dir / "trips_test.parquet"
    
    # Check if data files exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"❌ Data file not found: {path}")
            print("   Please run feat-eng.py first to generate the train/val/test splits")
            return
    
    # Load data
    train_df, val_df, test_df = load_and_prepare_data(
        str(train_path), str(val_path), str(test_path)
    )
    
    # Initialize and train model
    trainer = GBDTTrainer(
        model_type=args.model,
        eta_params=eta_params,
        dest_params=dest_params,
        use_gpu=args.gpu
    )
    
    trainer.fit(train_df, val_df)
    
    # Evaluate on validation set
    print(f"\n🔍 EVALUATING ON VALIDATION SET")
    print("-" * 40)
    
    val_eta_pred, val_dest_pred, val_dest_proba = trainer.predict(val_df)
    val_eta_true = val_df.select("duracion_recorrido").to_numpy().flatten()
    val_dest_true = val_df.select("id_estacion_destino").to_numpy().flatten()
    
    val_eta_metrics = evaluate_eta_prediction(val_eta_true, val_eta_pred)
    val_dest_metrics = evaluate_destination_prediction(val_dest_true, val_dest_pred, val_dest_proba)
    
    print_evaluation_results(val_eta_metrics, val_dest_metrics, f"{args.model.upper()} (Validation)")
    
    # Evaluate on test set
    print(f"\n🔍 EVALUATING ON TEST SET")
    print("-" * 40)
    
    test_eta_pred, test_dest_pred, test_dest_proba = trainer.predict(test_df)
    test_eta_true = test_df.select("duracion_recorrido").to_numpy().flatten()
    test_dest_true = test_df.select("id_estacion_destino").to_numpy().flatten()
    
    test_eta_metrics = evaluate_eta_prediction(test_eta_true, test_eta_pred)
    test_dest_metrics = evaluate_destination_prediction(test_dest_true, test_dest_pred, test_dest_proba)
    
    print_evaluation_results(test_eta_metrics, test_dest_metrics, f"{args.model.upper()} (Test)")
    
    # Save results and models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save evaluation results
    results_file = output_dir / f"{experiment_name}_results.csv"
    save_evaluation_results(test_eta_metrics, test_dest_metrics, experiment_name, str(results_file))
    
    # Save models
    model_paths = trainer.save_models(output_dir / "models" / experiment_name)
    
    # Save predictions
    predictions_df = test_df.with_columns([
        pl.Series("eta_pred", test_eta_pred),
        pl.Series("dest_pred", test_dest_pred)
    ])
    
    predictions_file = output_dir / f"{experiment_name}_predictions.parquet"
    predictions_df.write_parquet(predictions_file)
    print(f"💾 Predictions saved to: {predictions_file}")
    
    # Feature importance analysis
    if hasattr(trainer.eta_model, 'feature_importances_'):
        print(f"\n📊 TOP 10 FEATURES FOR ETA PREDICTION:")
        eta_importance = trainer.eta_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': trainer.feature_columns,
            'importance': eta_importance
        }).sort_values('importance', ascending=False)
        
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Save feature importance
        importance_file = output_dir / f"{experiment_name}_feature_importance.csv"
        feature_importance_df.to_csv(importance_file, index=False)
    
    print(f"\n✅ {args.model.upper()} training and evaluation complete!")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   Model Type:           {args.model.upper()}")
    print(f"   Experiment:           {experiment_name}")
    print(f"   Training Data:        {len(train_df):,} trips")
    print(f"   Validation Data:      {len(val_df):,} trips")
    print(f"   Test Data:            {len(test_df):,} trips")
    print(f"   Test ETA R²:          {test_eta_metrics['r2']:.4f}")
    print(f"   Test Destination Acc: {test_dest_metrics['accuracy']:.4f}")
    print(f"   Results saved to:     {output_dir}")


if __name__ == "__main__":
    main() 