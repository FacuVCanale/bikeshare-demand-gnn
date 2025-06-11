from src.models.base_model import BaseModel
from abc import abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb


class GBDTModel(BaseModel):
    """
    Abstract base class for GBDT models.
    """
    def prepare_dataset(self, dataset):
        """
        Prepares the dataset for a GBDT model.
        """
        dataset.prepare_for_gbdt(self)

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



class LightGBMModel(GBDTModel):
    """
    A LightGBM model implementation.
    """
    def __init__(self, model_type='regressor'):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'regressor':
            self.model = lgb.LGBMRegressor()
        elif self.model_type == 'classifier':
            self.model = lgb.LGBMClassifier()
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'")
        self.hps = {}

    def set_hps(self, hps: dict):
        """
        Set hyperparameters for the LightGBM model.
        """
        self.hps = hps
        self.model.set_params(**hps)

    def train(self, train_data, valid_data=None):
        """
        Train the LightGBM model.
        """
        X_train, y_train = train_data
        
        eval_set = []
        if valid_data:
            X_valid, y_valid = valid_data
            eval_set.append((X_valid, y_valid))

        early_stopping_rounds = self.hps.get('early_stopping_rounds', 10)
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(early_stopping_rounds)])

    def predict(self, data):
        """
        Make predictions with the trained model.
        """
        X = data[0] # Assuming data is a tuple (X, y) or just X
        return self.model.predict(X)

    def get_metrics(self, data) -> dict:
        """
        Calculate and return metrics on the given data.
        For regression, this returns MSE and R2.
        """
        X, y_true = data
        y_pred = self.predict((X,))

        if self.model_type == 'regressor':
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return {'mse': mse, 'r2': r2, 'rmse': np.sqrt(mse)}
        elif self.model_type == 'classifier':
            # Example for classifier, you can add more metrics
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_true, y_pred)
            return {'accuracy': acc}
        return {}

    def save(self, path):
        """
        Save the model to a file using joblib.
        """
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load a model from a file using joblib.
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model 
    

class XGBoostModel(GBDTModel):
    """
    A XGBoost model implementation.
    """
    def __init__(self, model_type='regressor'):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'regressor':
            self.model = xgb.XGBRegressor()
        elif self.model_type == 'classifier':
            self.model = xgb.XGBClassifier()
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'")
        self.hps = {}

    def set_hps(self, hps: dict):
        """
        Set hyperparameters for the XGBoost model.
        """
        self.hps = hps
        self.model.set_params(**hps)

    def train(self, train_data, valid_data=None):
        """
        Train the XGBoost model.
        """
        X_train, y_train = train_data
        
        eval_set = []
        if valid_data:
            X_valid, y_valid = valid_data
            eval_set.append((X_valid, y_valid))

        early_stopping_rounds = self.hps.get('early_stopping_rounds', 10)
        self.model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)

    def predict(self, data):
        """
        Make predictions with the trained model.
        """
        X = data[0]
        return self.model.predict(X)

    def get_metrics(self, data) -> dict:
        """
        Calculate and return metrics on the given data.
        """
        X, y_true = data
        y_pred = self.predict((X,))

        if self.model_type == 'regressor':
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return {'mse': mse, 'r2': r2, 'rmse': np.sqrt(mse)}
        elif self.model_type == 'classifier':
            acc = accuracy_score(y_true, y_pred)
            return {'accuracy': acc}
        return {}

    def save(self, path):
        """
        Save the model to a file using joblib.
        """
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load a model from a file using joblib.
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model
