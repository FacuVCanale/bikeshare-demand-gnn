from src.models.base_model import BaseModel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

class NNModel(BaseModel):
    """
    Base class for Neural Network models, using scikit-learn's MLPRegressor as an example.
    """
    def __init__(self):
        super().__init__()
        self.model = MLPRegressor()

    def prepare_dataset(self, dataset):
        """Prepares the dataset for a Neural Network model."""
        dataset.prepare_for_nn(self)

    def set_hps(self, hps: dict):
        self.hps = hps
        self.model.set_params(**hps)

    def train(self, train_data, valid_data=None):
        X_train, y_train = train_data
        self.model.fit(X_train, y_train)

    def predict(self, data):
        X = data[0]
        return self.model.predict(X)

    def get_metrics(self, data) -> dict:
        X, y_true = data
        y_pred = self.predict((X,))
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'mse': mse, 'r2': r2, 'rmse': np.sqrt(mse)}

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model 