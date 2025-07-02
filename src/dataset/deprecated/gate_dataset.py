"""
Dataset class for Gate model training with double dispatch pattern.
"""
from typing import Tuple, Optional, List
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.features.feature_selector import GateFeatureSelector, create_gate_target


class GateDataset:
    """
    Dataset class for Gate model with feature selection and preprocessing.
    Implements double dispatch pattern for different model types.
    """
    
    def __init__(
        self, 
        train_path: str,
        val_path: str, 
        test_path: str,
        target_col: str = 'has_activity',
        feature_groups: Optional[List[str]] = None
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.feature_groups = feature_groups
        
        self.feature_selector = GateFeatureSelector()
        
        # data containers
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # processed data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.feature_names = None
        self.categorical_features = None
        
    def load_data(self):
        """Load the raw data from parquet files."""
        print("Loading datasets...")
        
        self.train_df = pl.read_parquet(self.train_path)
        self.val_df = pl.read_parquet(self.val_path)
        self.test_df = pl.read_parquet(self.test_path)
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Validation shape: {self.val_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        return self
    
    def create_targets(self):
        """Create binary target variables for Gate model."""
        print("Creating binary targets...")
        
        self.train_df = create_gate_target(self.train_df, self.target_col)
        self.val_df = create_gate_target(self.val_df, self.target_col)
        self.test_df = create_gate_target(self.test_df, self.target_col)
        
        # print target distribution
        train_pos = self.train_df.filter(pl.col(self.target_col) == 1).shape[0]
        train_total = self.train_df.shape[0]
        print(f"Train positive class ratio: {train_pos/train_total:.3f} ({train_pos}/{train_total})")
        
        val_pos = self.val_df.filter(pl.col(self.target_col) == 1).shape[0]
        val_total = self.val_df.shape[0]
        print(f"Validation positive class ratio: {val_pos/val_total:.3f} ({val_pos}/{val_total})")
        
        return self
    
    def select_features(self):
        """Select features for Gate model."""
        print("Selecting features for Gate model...")
        
        # get original feature count
        original_features = len(self.train_df.columns) - 2  # exclude ts_start and target
        
        # filter datasets
        self.train_df = self.feature_selector.filter_dataframe(
            self.train_df, self.target_col, self.feature_groups
        )
        self.val_df = self.feature_selector.filter_dataframe(
            self.val_df, self.target_col, self.feature_groups
        )
        self.test_df = self.feature_selector.filter_dataframe(
            self.test_df, self.target_col, self.feature_groups
        )
        
        # store feature names (exclude ts_start and target)
        self.feature_names = [col for col in self.train_df.columns 
                             if col not in ['ts_start', self.target_col]]
        
        selected_features = len(self.feature_names)
        print(f"Features: {original_features} -> {selected_features} (reduced by {original_features - selected_features})")
        
        # identify categorical features
        self.categorical_features = ['cluster_id']  # cluster_id is categorical
        
        return self
    
    def prepare_for_catboost(self, model):
        """
        Prepare dataset for CatBoost model (double dispatch).
        """
        print("Preparing data for CatBoost...")
        
        # convert to pandas for CatBoost
        train_pd = self.train_df.to_pandas()
        val_pd = self.val_df.to_pandas()
        test_pd = self.test_df.to_pandas()
        
        # separate features and targets
        self.X_train = train_pd[self.feature_names]
        self.y_train = train_pd[self.target_col]
        
        self.X_val = val_pd[self.feature_names]
        self.y_val = val_pd[self.target_col]
        
        self.X_test = test_pd[self.feature_names]
        self.y_test = test_pd[self.target_col]
        
        # set categorical features in model
        if hasattr(model, 'cat_features'):
            cat_indices = [i for i, col in enumerate(self.feature_names) 
                          if col in self.categorical_features]
            model.cat_features = cat_indices
        
        print(f"Data prepared - Features: {len(self.feature_names)}, Categorical: {len(self.categorical_features)}")
        
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)
    
    def prepare_for_gbdt(self, model):
        """
        Prepare dataset for GBDT models (LightGBM, XGBoost) - double dispatch.
        """
        print("Preparing data for GBDT...")
        
        # convert to pandas
        train_pd = self.train_df.to_pandas()
        val_pd = self.val_df.to_pandas()
        test_pd = self.test_df.to_pandas()
        
        # separate features and targets
        self.X_train = train_pd[self.feature_names]
        self.y_train = train_pd[self.target_col]
        
        self.X_val = val_pd[self.feature_names]
        self.y_val = val_pd[self.target_col]
        
        self.X_test = test_pd[self.feature_names]
        self.y_test = test_pd[self.target_col]
        
        # encode categorical features for GBDT
        if self.categorical_features:
            from sklearn.preprocessing import LabelEncoder
            
            for cat_col in self.categorical_features:
                if cat_col in self.feature_names:
                    le = LabelEncoder()
                    # fit on combined data to ensure consistent encoding
                    combined_values = pd.concat([
                        self.X_train[cat_col], 
                        self.X_val[cat_col], 
                        self.X_test[cat_col]
                    ])
                    le.fit(combined_values)
                    
                    self.X_train[cat_col] = le.transform(self.X_train[cat_col])
                    self.X_val[cat_col] = le.transform(self.X_val[cat_col])
                    self.X_test[cat_col] = le.transform(self.X_test[cat_col])
        
        print(f"Data prepared - Features: {len(self.feature_names)}")
        
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)
    
    def get_feature_info(self) -> dict:
        """Get information about features."""
        return {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'categorical_features': self.categorical_features,
            'n_categorical': len(self.categorical_features) if self.categorical_features else 0,
            'target_col': self.target_col
        }
    
    def print_data_summary(self):
        """Print summary of the dataset."""
        if self.train_df is None:
            print("Data not loaded yet. Call load_data() first.")
            return
        
        print("\n=== Dataset Summary ===")
        print(f"Train samples: {self.train_df.shape[0]:,}")
        print(f"Validation samples: {self.val_df.shape[0]:,}")
        print(f"Test samples: {self.test_df.shape[0]:,}")
        
        if self.feature_names:
            print(f"Features: {len(self.feature_names)}")
            print(f"Categorical features: {len(self.categorical_features)}")
        
        if self.target_col in self.train_df.columns:
            train_pos_rate = self.train_df.filter(pl.col(self.target_col) == 1).shape[0] / self.train_df.shape[0]
            val_pos_rate = self.val_df.filter(pl.col(self.target_col) == 1).shape[0] / self.val_df.shape[0]
            
            print(f"\nTarget distribution:")
            print(f"  Train positive rate: {train_pos_rate:.3f}")
            print(f"  Validation positive rate: {val_pos_rate:.3f}")
        
        # feature selection summary
        if hasattr(self, 'feature_selector') and self.feature_selector.selected_features:
            print(f"\nFeature selection summary:")
            all_original_features = [col for col in pl.read_parquet(self.train_path, n_rows=1).columns 
                                   if col not in ['ts_start']]
            self.feature_selector.print_feature_summary(all_original_features)
    
    def get_train_data(self) -> Tuple:
        """Get training data as (X, y) tuple."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_for_* method first.")
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Tuple:
        """Get validation data as (X, y) tuple."""
        if self.X_val is None or self.y_val is None:
            raise ValueError("Data not prepared. Call prepare_for_* method first.")
        return self.X_val, self.y_val
    
    def get_test_data(self) -> Tuple:
        """Get test data as (X, y) tuple."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not prepared. Call prepare_for_* method first.")
        return self.X_test, self.y_test 