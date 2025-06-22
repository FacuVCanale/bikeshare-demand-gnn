from .base_model import BaseModel
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoostModel(BaseModel):
    """
    CatBoost model implementation for both classification and regression.
    """
    
    def __init__(self, model_type='classifier', cat_features=None):
        super().__init__()
        self.model_type = model_type
        self.cat_features = cat_features or []
        
        if self.model_type == 'classifier':
            self.model = CatBoostClassifier(verbose=False)
        elif self.model_type == 'regressor':
            self.model = CatBoostRegressor(verbose=False)
        else:
            raise ValueError("model_type must be 'classifier' or 'regressor'")
        
        self.hps = {}

    def prepare_dataset(self, dataset):
        """
        Prepares the dataset for CatBoost model using double dispatch.
        """
        return dataset.prepare_for_catboost(self)

    def set_hps(self, hps: dict):
        """
        Set hyperparameters for the CatBoost model.
        """
        self.hps = hps
        self.model.set_params(**hps)

    def train(self, train_data, valid_data=None):
        """
        Train the CatBoost model.
        """
        X_train, y_train = train_data
        
        eval_set = None
        if valid_data:
            X_valid, y_valid = valid_data
            eval_set = (X_valid, y_valid)

        # catboost handles early stopping internally
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            cat_features=self.cat_features,
            verbose=self.hps.get('verbose', False)
        )

    def predict(self, data):
        """
        Make predictions with the trained model.
        """
        X = data[0] if isinstance(data, tuple) else data
        return self.model.predict(X)
    
    def predict_proba(self, data):
        """
        Make probability predictions (for classifiers).
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")
        
        X = data[0] if isinstance(data, tuple) else data
        return self.model.predict_proba(X)

    def get_metrics(self, data) -> dict:
        """
        Calculate and return metrics on the given data.
        """
        X, y_true = data
        y_pred = self.predict((X,))

        if self.model_type == 'classifier':
            # binary classification metrics
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
                'f1_score': f1,
            }
            
            if auc is not None:
                metrics['auc'] = auc
                
            return metrics
            
        elif self.model_type == 'regressor':
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return {
                'mse': mse, 
                'mae': mae,
                'rmse': np.sqrt(mse), 
                'r2': r2
            }
        
        return {}

    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save(self, path):
        """
        Save the model to a file.
        """
        self.model.save_model(path)
        print(f"CatBoost model saved to {path}")

    def load(self, path):
        """
        Load a model from a file.
        """
        if self.model_type == 'classifier':
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()
        
        self.model.load_model(path)
        print(f"CatBoost model loaded from {path}")
        return self.model 