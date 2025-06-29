from .base_model import BaseModel
from abc import abstractmethod
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, mean_absolute_error
)
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import gc
import warnings


class GBDTModel(BaseModel):
    """
    Abstract base class for GBDT models.
    """
    def prepare_dataset(self, dataset):
        """
        Prepares the dataset for a GBDT model.
        """
        return dataset.prepare_for_gbdt(self)

    @abstractmethod
    def train(self, train_data, valid_data=None):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def get_metrics(self, data) -> dict:
        pass

    @abstractmethod
    def set_hps(self, hps: dict):
        pass

    def predict_proba(self, data):
        """
        Make probability predictions (for classifiers).
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")
        
        X = data[0] if isinstance(data, tuple) else data
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class LightGBMModel(GBDTModel):
    """
    Optimized LightGBM model implementation with complete metrics and GPU support.
    """
    def __init__(self, model_type='regressor'):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'regressor':
            self.model = lgb.LGBMRegressor(verbose=-1)
        elif self.model_type == 'classifier':
            self.model = lgb.LGBMClassifier(verbose=-1)
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'")
        self.hps = {}
        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if LightGBM GPU is available."""
        try:
            # test gpu availability by creating a small model
            test_model = lgb.LGBMClassifier(device='gpu', verbosity=-1)
            return True
        except Exception:
            return False

    def set_hps(self, hps: dict):
        """
        Set hyperparameters for the LightGBM model with GPU optimization.
        """
        self.hps = hps.copy()
        
        # optimize GPU settings
        if not hps.get('device') == 'cpu' and self._gpu_available:
            self.hps.update({
                'device': 'gpu',
                'gpu_use_dp': True,  # use double precision on GPU
                'verbosity': -1
            })
        else:
            self.hps['device'] = 'cpu'
            self.hps['verbosity'] = -1
        
        self.model.set_params(**self.hps)

    def train(self, train_data, valid_data=None):
        """
        Train the LightGBM model with optimized settings.
        """
        X_train, y_train = train_data
        
        eval_set = []
        eval_names = []
        if valid_data:
            X_valid, y_valid = valid_data
            eval_set = [(X_train, y_train), (X_valid, y_valid)]
            eval_names = ['train', 'valid']

        # train with optimized callbacks
        early_stopping_rounds = self.hps.get('early_stopping_rounds', 100)
        
        callbacks = [lgb.early_stopping(early_stopping_rounds)]
        if self.hps.get('verbosity', -1) >= 0:
            callbacks.append(lgb.log_evaluation(period=100))

        try:
            self.model.fit(
                X_train, y_train, 
                eval_set=eval_set,
                eval_names=eval_names,
                callbacks=callbacks
            )
        except Exception as e:
            # fallback to CPU if GPU fails
            if 'gpu' in str(e).lower() and self.hps.get('device') == 'gpu':
                warnings.warn("GPU training failed, falling back to CPU")
                self.hps['device'] = 'cpu'
                self.model.set_params(**self.hps)
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    eval_names=eval_names,
                    callbacks=callbacks
                )
            else:
                raise e
        
        # force garbage collection to free memory
        gc.collect()

    def predict(self, data):
        """
        Make predictions with the trained model.
        """
        X = data[0] if isinstance(data, tuple) else data
        return self.model.predict(X)

    def get_metrics(self, data) -> dict:
        """
        Calculate and return comprehensive metrics.
        """
        X, y_true = data
        y_pred = self.predict((X,))

        if self.model_type == 'regressor':
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            return {
                'mse': mse,
                'mae': mae, 
                'rmse': rmse,
                'r2': r2
            }
        elif self.model_type == 'classifier':
            # comprehensive classification metrics
            try:
                y_pred_proba = self.predict_proba((X,))
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # positive class probabilities
                auc = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                auc = None
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            if auc is not None:  
                metrics['auc'] = auc
                
            return metrics
        
        return {}

    def save(self, path):
        """
        Save the model to a file using joblib.
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a model from a file using joblib.
        """
        self.model = joblib.load(path)
        return self.model 
    

class XGBoostModel(GBDTModel):
    """
    Optimized XGBoost model implementation with complete metrics and GPU support.
    """
    def __init__(self, model_type='regressor'):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'regressor':
            self.model = xgb.XGBRegressor(verbosity=0)
        elif self.model_type == 'classifier':
            self.model = xgb.XGBClassifier(verbosity=0)
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'")
        self.hps = {}
        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if XGBoost GPU is available."""
        try:
            # test gpu availability
            test_model = xgb.XGBClassifier(tree_method='gpu_hist', verbosity=0)
            return True
        except Exception:
            return False

    def set_hps(self, hps: dict):
        """
        Set hyperparameters for the XGBoost model with GPU optimization.
        """
        self.hps = hps.copy()
        
        # optimize GPU settings
        if hps.get('tree_method') != 'hist' and self._gpu_available:
            self.hps.update({
                'tree_method': 'gpu_hist',
                'gpu_id': hps.get('gpu_id', 0),
                'verbosity': 0
            })
        else:
            self.hps['tree_method'] = 'hist'  # CPU fallback
            self.hps['verbosity'] = 0
        
        self.model.set_params(**self.hps)

    def train(self, train_data, valid_data=None):
        """
        Train the XGBoost model with optimized settings.
        """
        X_train, y_train = train_data
        
        eval_set = []
        if valid_data:
            X_valid, y_valid = valid_data
            eval_set = [(X_valid, y_valid)]

        # early_stopping_rounds is already set via set_params() in set_hps()
        # No need to pass it to fit() method
        
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        except Exception as e:
            # fallback to CPU if GPU fails
            if 'gpu' in str(e).lower() and self.hps.get('tree_method') == 'gpu_hist':
                warnings.warn("GPU training failed, falling back to CPU")
                self.hps['tree_method'] = 'hist'
                self.model.set_params(**self.hps)
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                raise e
        
        # force garbage collection to free memory
        gc.collect()

    def predict(self, data):
        """
        Make predictions with the trained model.
        """
        X = data[0] if isinstance(data, tuple) else data
        return self.model.predict(X)

    def get_metrics(self, data) -> dict:
        """
        Calculate and return comprehensive metrics.
        """
        X, y_true = data
        y_pred = self.predict((X,))

        if self.model_type == 'regressor':
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        elif self.model_type == 'classifier':
            # comprehensive classification metrics
            try:
                y_pred_proba = self.predict_proba((X,))
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # positive class probabilities
                auc = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                auc = None
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            if auc is not None:
                metrics['auc'] = auc
                
            return metrics
        
        return {}

    def save(self, path):
        """
        Save the model to a file using joblib.
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a model from a file using joblib.
        """
        self.model = joblib.load(path)
        return self.model
