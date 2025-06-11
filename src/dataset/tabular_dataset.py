import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.dataset.base_dataset import BaseDataset

class TabularDataset(BaseDataset):
    """
    A dataset class for tabular data.
    """
    def __init__(self, data: pd.DataFrame, target_columns: list[str]):
        super().__init__()
        self.data = data
        self.target_columns = target_columns
        self.scaler = StandardScaler()
        self._split_data()

    def _split_data(self):
        """
        Splits data into train, validation, and test sets.
        """
        X = self.data.drop(columns=self.target_columns)
        y = self.data[self.target_columns]

        # Split into train+valid and test
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Split train+valid into train and valid
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, y_train_valid, test_size=0.25, random_state=42 # 0.25 * 0.8 = 0.2
        )

        self.train = (X_train, y_train)
        self.valid = (X_valid, y_valid)
        self.test = (X_test, y_test)

    def prepare_for(self, model):
        """
        Kicks off the double dispatch process for data preparation.
        """
        model.prepare_dataset(self)

    def prepare_for_gbdt(self, model):
        """Prepares data for a GBDT model."""
        print(f"Preparing data for {model.__class__.__name__}")
        if len(self.target_columns) > 1:
            raise ValueError("GBDTModel only supports a single target column.")
        # No normalization needed for tree-based models
        print("No normalization needed.")

    def prepare_for_multi_model(self, model):
        """Prepares data for a MultiModel."""
        print(f"Preparing data for {model.__class__.__name__}")
        if len(self.target_columns) <= 1:
            raise ValueError("MultiModel requires multiple target columns.")
        self._normalize_data()
        print("Data normalized.")

    def prepare_for_nn(self, model):
        """Prepares data for an NNModel."""
        print(f"Preparing data for {model.__class__.__name__}")
        self._normalize_data()
        print("Data normalized.")

    def _normalize_data(self):
        """Helper method to normalize features using StandardScaler"""
        X_train, y_train = self.train
        X_valid, y_valid = self.valid
        X_test, y_test = self.test

        # Fit scaler on training data and transform all sets
        X_train_norm = self.scaler.fit_transform(X_train)
        X_valid_norm = self.scaler.transform(X_valid)
        X_test_norm = self.scaler.transform(X_test)

        # Update the data tuples with normalized features
        self.train = (pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index), y_train)
        self.valid = (pd.DataFrame(X_valid_norm, columns=X_valid.columns, index=X_valid.index), y_valid)
        self.test = (pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index), y_test)

