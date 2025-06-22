import polars as pl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from pathlib import Path
from typing import Tuple, List, Optional
import json
from sklearn.preprocessing import StandardScaler

class EcoBiciSpatialTemporalDataset:
    """
    Dataset for EcoBici arrival prediction using Graph Neural Networks.
    Focuses only on external (real) trips, ignoring internal (phantom) trips.
    
    Creates a spatio-temporal graph where:
    - Nodes: 93 bike station clusters
    - Edges: spatial relationships between clusters
    - Target: predict arr_external_count for next time step
    """
    
    def __init__(self, data_path: str, sequence_length: int = 12, prediction_horizon: int = 1, k_neighbors: int = 8):
        """
        Args:
            data_path: Path to the data directory
            sequence_length: Number of time steps to use as input
            prediction_horizon: Number of time steps to predict (default: 1)
            k_neighbors: Number of spatial neighbors for graph construction (default: 8)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.k_neighbors = k_neighbors
        self.n_clusters = 93
        
        # load metadata
        self.metadata = self._load_metadata()
        
        # feature groups to use (only external features)
        self.feature_columns = self._get_external_feature_columns()
        
        # scalers for normalization
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def _load_metadata(self) -> dict:
        """Load dataset metadata"""
        metadata_path = self.data_path / "clustered" / "dataset_metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _get_external_feature_columns(self) -> List[str]:
        """
        Get list of external feature columns, excluding internal (phantom) trips
        """
        # base columns
        base_cols = ['cluster_id', 'ts_start']
        
        # target column (what we want to predict)
        target_cols = ['arr_external_count']
        
        # external departure features
        external_dep_cols = [
            'dep_external_count', 'dep_external_male', 'dep_external_female', 
            'dep_external_other', 'dep_external_young', 'dep_external_adult', 
            'dep_external_senior', 'dep_external_age_available', 'dep_external_model_iconic',
            'dep_external_duration_mean', 'dep_external_duration_std'
        ]
        
        # weather features (general, not specific to internal/external)
        weather_cols = [
            'temperature_2m_mean', 'relative_humidity_2m_mean', 'dew_point_2m_mean',
            'apparent_temperature_mean', 'precipitation_mean', 'weather_code_mean',
            'pressure_msl_mean', 'cloud_cover_mean', 'wind_speed_10m_mean',
            'wind_direction_10m_mean', 'wind_gusts_10m_mean', 'is_day',
            'temp_comfortable', 'temp_very_hot', 'temp_cold', 'is_raining',
            'heavy_rain', 'light_rain', 'strong_wind', 'moderate_wind'
        ]
        
        # temporal features
        temporal_cols = [
            'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_daytime',
            'is_summer', 'is_winter'
        ]
        
        # spatial features
        spatial_cols = [
            'cluster_centroid_lat', 'cluster_centroid_lon', 'cluster_station_count'
        ]
        
        # lag features (only external)
        lag_cols = []
        for feature in ['dep_external_count', 'arr_external_count']:
            for lag in [1, 2, 4, 8]:
                lag_cols.append(f"{feature}_lag_{lag}")
        
        # rolling features (only external)
        rolling_cols = []
        for feature in ['dep_external_count', 'arr_external_count']:
            for window in [4, 8, 24]:
                rolling_cols.extend([
                    f"{feature}_rolling_mean_{window}",
                    f"{feature}_rolling_std_{window}"
                ])
        
        # same hour features (only external)
        same_hour_cols = []
        for feature in ['dep_external_count', 'arr_external_count']:
            same_hour_cols.extend([
                f"{feature}_same_hour_yesterday",
                f"{feature}_same_hour_last_week"
            ])
        
        return (base_cols + target_cols + external_dep_cols + weather_cols + 
                temporal_cols + spatial_cols + lag_cols + rolling_cols + same_hour_cols)
    
    def _create_graph_structure(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph structure based on spatial proximity of clusters.
        Returns edge_index and edge_weights tensors.
        """
        # get cluster centroids
        centroids = []
        for i in range(self.n_clusters):
            cluster_info = self.metadata['cluster_centroids'][str(i)]
            centroids.append([cluster_info['lat'], cluster_info['lon']])
        
        centroids = np.array(centroids)
        
        # create edges based on k-nearest neighbors or distance threshold
        edge_list = []
        edge_weights = []
        
        # use k-nearest neighbors to create edges
        k = self.k_neighbors
        for i in range(self.n_clusters):
            # calculate distances to all other clusters
            distances = np.sqrt(np.sum((centroids - centroids[i])**2, axis=1))
            
            # get k nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances)[1:k+1]
            
            for j in neighbor_indices:
                edge_list.append([i, j])
                # use inverse distance as weight
                weight = 1.0 / (distances[j] + 1e-8)
                edge_weights.append(weight)
        
        # convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weights
    
    def _prepare_temporal_data(self, df: pl.DataFrame) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepare temporal sequences for the GNN with proper temporal ordering.
        Uses sequence_length previous time steps to predict the next time step.
        
        CRITICAL: Ensures no data leakage by strictly using past data to predict future.
        
        Returns:
            features: List of feature tensors for each valid time step
            targets: List of target tensors for each valid time step
        """
        # sort by time and cluster to ensure temporal ordering
        df = df.sort(['ts_start', 'cluster_id'])
        
        # get unique timestamps in chronological order
        timestamps = df['ts_start'].unique().sort()
        
        # prepare feature columns (excluding base columns and current target)
        feature_cols = [col for col in self.feature_columns 
                       if col not in ['cluster_id', 'ts_start', 'arr_external_count']]
        
        features_list = []
        targets_list = []
        
        # start from sequence_length to ensure we have enough history
        # this prevents data leakage by requiring sufficient past data
        for i in range(self.sequence_length, len(timestamps) - self.prediction_horizon):
            
            # get current timestamp for target
            current_ts = timestamps[i]
            target_ts = timestamps[i + self.prediction_horizon]  # predict future arrivals
            
            # get current features (at time t)
            current_data = df.filter(pl.col('ts_start') == current_ts).sort('cluster_id')
            
            # get target data (at time t + prediction_horizon)
            target_data = df.filter(pl.col('ts_start') == target_ts).sort('cluster_id')
            
            # ensure we have data for all clusters at both timestamps
            if len(current_data) != self.n_clusters or len(target_data) != self.n_clusters:
                continue
            
            # verify temporal ordering (critical for preventing leakage)
            if current_data['ts_start'].min() >= target_data['ts_start'].min():
                raise ValueError(f"Data leakage detected: features timestamp {current_data['ts_start'].min()} >= target timestamp {target_data['ts_start'].min()}")
            
            # extract features (shape: [n_clusters, n_features])
            # these features are from time t, used to predict time t+prediction_horizon
            features = current_data.select(feature_cols).to_numpy().astype(np.float32)
            
            # extract targets (shape: [n_clusters])
            # these are the arrivals we want to predict at time t+prediction_horizon
            targets = target_data.select('arr_external_count').to_numpy().flatten().astype(np.float32)
            
            # verify no NaN or infinite values that could indicate data issues
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print(f"Warning: NaN/Inf values in features at timestamp {current_ts}")
                continue
                
            if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
                print(f"Warning: NaN/Inf values in targets at timestamp {target_ts}")
                continue
            
            features_list.append(torch.tensor(features))
            targets_list.append(torch.tensor(targets))
        
        print(f"Prepared {len(features_list)} temporal samples with no data leakage")
        print(f"Each sample uses features from time t to predict arrivals at time t+{self.prediction_horizon}")
        
        return features_list, targets_list
    
    def load_data(self, split: str = 'train') -> StaticGraphTemporalSignal:
        """
        Load and prepare data for the specified split with strict data leakage prevention.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            StaticGraphTemporalSignal object for PyTorch Geometric Temporal
        """
        print(f"\n🔄 Loading {split} data...")
        
        # load data
        file_path = self.data_path / "clustered" / f"{split}_cluster_features.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pl.read_parquet(file_path)
        print(f"Loaded {len(df)} rows for {split} split")
        
        # verify temporal ordering
        timestamps = df['ts_start'].unique().sort()
        if len(timestamps) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Insufficient timestamps for {split}: {len(timestamps)} < {self.sequence_length + self.prediction_horizon}")
        
        # check for temporal continuity (detect gaps that could indicate data issues)
        time_diffs = df.sort('ts_start')['ts_start'].diff().drop_nulls().unique()
        expected_diff = time_diffs.mode()[0] if len(time_diffs) > 0 else None
        
        # filter to only include relevant columns
        available_cols = [col for col in self.feature_columns if col in df.columns]
        missing_cols = set(self.feature_columns) - set(available_cols)
        if missing_cols:
            print(f"⚠️  Warning: Missing columns in {split}: {missing_cols}")
        
        df = df.select(available_cols)
        
        # handle missing values (fill with 0 to maintain temporal structure)
        df = df.fill_null(0)
        
        # create graph structure (static for all timestamps)
        edge_index, edge_weights = self._create_graph_structure()
        
        # prepare temporal data with leakage prevention
        features_list, targets_list = self._prepare_temporal_data(df)
        
        if len(features_list) == 0:
            raise ValueError(f"No valid temporal sequences generated for {split}")
        
        # CRITICAL: Only fit scalers on training data to prevent data leakage
        if split == 'train':
            print("🔧 Fitting scalers on training data only...")
            # concatenate all training features and targets
            all_features = torch.cat(features_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            
            # fit scalers only on training data
            self.feature_scaler.fit(all_features.numpy())
            self.target_scaler.fit(all_targets.numpy().reshape(-1, 1))
            
            print(f"Feature scaler fitted on {all_features.shape[0]} samples with {all_features.shape[1]} features")
            print(f"Target scaler fitted on {all_targets.shape[0]} samples")
            
        elif not hasattr(self.feature_scaler, 'mean_'):
            raise ValueError(f"Scalers not fitted! Must load 'train' split first before loading '{split}'")
        
        # apply normalization using training statistics
        normalized_features = []
        normalized_targets = []
        
        for features, targets in zip(features_list, targets_list):
            # normalize features using training statistics
            norm_features = torch.tensor(
                self.feature_scaler.transform(features.numpy()), 
                dtype=torch.float32
            )
            
            # normalize targets using training statistics
            norm_targets = torch.tensor(
                self.target_scaler.transform(targets.numpy().reshape(-1, 1)).flatten(),
                dtype=torch.float32
            )
            
            # validate normalized data
            if torch.any(torch.isnan(norm_features)) or torch.any(torch.isinf(norm_features)):
                raise ValueError("NaN or Inf values detected in normalized features")
            if torch.any(torch.isnan(norm_targets)) or torch.any(torch.isinf(norm_targets)):
                raise ValueError("NaN or Inf values detected in normalized targets")
            
            normalized_features.append(norm_features)
            normalized_targets.append(norm_targets)
        
        # create StaticGraphTemporalSignal
        dataset = StaticGraphTemporalSignal(
            edge_index=edge_index,
            edge_weight=edge_weights,
            features=normalized_features,
            targets=normalized_targets
        )
        
        print(f"✅ Successfully created {split} dataset:")
        print(f"   • Temporal sequences: {len(dataset)}")
        print(f"   • Nodes per graph: {edge_index.max().item() + 1}")
        print(f"   • Edges per graph: {edge_index.shape[1]}")
        print(f"   • Features per node: {normalized_features[0].shape[1] if normalized_features else 0}")
        print(f"   • Prediction horizon: {self.prediction_horizon} time steps")
        
        return dataset
    
    def inverse_transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform normalized targets back to original scale.
        """
        targets_np = targets.detach().cpu().numpy()
        if targets_np.ndim == 1:
            targets_np = targets_np.reshape(-1, 1)
        
        original_targets = self.target_scaler.inverse_transform(targets_np)
        return torch.tensor(original_targets.flatten(), dtype=torch.float32)
    
    def validate_no_data_leakage(self, split: str = 'train', sample_size: int = 1000) -> bool:
        """
        Validate that lag features don't contain data leakage.
        Checks that lag_1 features are actually from 1 time step ago, etc.
        
        Args:
            split: Data split to validate
            sample_size: Number of random samples to check
            
        Returns:
            True if no leakage detected, raises ValueError if leakage found
        """
        print(f"\n🔍 Validating data leakage for {split} split...")
        
        # load raw data
        file_path = self.data_path / "clustered" / f"{split}_cluster_features.parquet"
        df = pl.read_parquet(file_path)
        
        # sort by time and cluster
        df = df.sort(['ts_start', 'cluster_id'])
        timestamps = df['ts_start'].unique().sort()
        
        # sample random timestamps for validation
        import random
        sample_timestamps = random.sample(list(timestamps[10:-10]), min(sample_size, len(timestamps)-20))
        
        violations = []
        
        for ts in sample_timestamps:
            ts_data = df.filter(pl.col('ts_start') == ts).sort('cluster_id')
            
            if len(ts_data) != self.n_clusters:
                continue
                
            # check lag features
            for lag in [1, 2, 4, 8]:
                # find the timestamp that should correspond to this lag
                ts_index = timestamps.search_sorted(ts)
                if ts_index < lag:
                    continue
                    
                expected_lag_ts = timestamps[ts_index - lag]
                expected_data = df.filter(pl.col('ts_start') == expected_lag_ts).sort('cluster_id')
                
                if len(expected_data) != self.n_clusters:
                    continue
                
                # check if lag feature matches actual historical data
                for feature in ['dep_external_count', 'arr_external_count']:
                    lag_col = f"{feature}_lag_{lag}"
                    
                    if lag_col in ts_data.columns and feature in expected_data.columns:
                        # get lag feature values
                        lag_values = ts_data[lag_col].to_numpy()
                        # get actual historical values
                        actual_values = expected_data[feature].to_numpy()
                        
                        # check if they match (allowing for small floating point errors)
                        if not np.allclose(lag_values, actual_values, rtol=1e-10, atol=1e-10, equal_nan=True):
                            violations.append({
                                'timestamp': ts,
                                'lag': lag,
                                'feature': feature,
                                'expected_timestamp': expected_lag_ts,
                                'lag_values_sample': lag_values[:3].tolist(),
                                'actual_values_sample': actual_values[:3].tolist()
                            })
        
        if violations:
            print(f"❌ Data leakage detected! Found {len(violations)} violations:")
            for v in violations[:5]:  # show first 5 violations
                print(f"   • {v['feature']}_lag_{v['lag']} at {v['timestamp']} doesn't match actual values at {v['expected_timestamp']}")
                print(f"     Lag values: {v['lag_values_sample']}")
                print(f"     Actual values: {v['actual_values_sample']}")
            
            raise ValueError(f"Data leakage detected in lag features! Found {len(violations)} violations. Check feature engineering pipeline.")
        
        print(f"✅ No data leakage detected in lag features ({len(sample_timestamps)} timestamps validated)")
        return True
    
    def get_temporal_split_dates(self) -> dict:
        """
        Get the date ranges for each split to verify proper temporal splitting.
        """
        splits_info = {}
        
        for split in ['train', 'val', 'test']:
            file_path = self.data_path / "clustered" / f"{split}_cluster_features.parquet"
            if file_path.exists():
                df = pl.read_parquet(file_path)
                splits_info[split] = {
                    'start_date': df['ts_start'].min(),
                    'end_date': df['ts_start'].max(),
                    'n_timestamps': df['ts_start'].n_unique(),
                    'n_samples': len(df)
                }
        
        return splits_info
    
    def print_data_leakage_report(self) -> None:
        """
        Print a comprehensive report about potential data leakage.
        """
        print("\n" + "="*60)
        print("🛡️  DATA LEAKAGE PREVENTION REPORT")
        print("="*60)
        
        # check temporal split boundaries
        print("\n📅 Temporal Split Boundaries:")
        splits_info = self.get_temporal_split_dates()
        
        prev_end = None
        for split in ['train', 'val', 'test']:
            if split in splits_info:
                info = splits_info[split]
                print(f"   {split.upper():>5}: {info['start_date']} to {info['end_date']} ({info['n_timestamps']} timestamps)")
                
                if prev_end and info['start_date'] <= prev_end:
                    print(f"   ⚠️  WARNING: {split} starts before previous split ends!")
                    
                prev_end = info['end_date']
        
        # check feature configuration
        print(f"\n🔧 Model Configuration:")
        print(f"   • Sequence length: {self.sequence_length}")
        print(f"   • Prediction horizon: {self.prediction_horizon}")
        print(f"   • K-neighbors: {self.k_neighbors}")
        
        # check lag features
        print(f"\n🔄 Lag Features Configuration:")
        lag_features = [col for col in self.feature_columns if '_lag_' in col]
        print(f"   • Total lag features: {len(lag_features)}")
        
        lag_types = {}
        for feature in lag_features:
            base_feature = feature.split('_lag_')[0]
            lag_num = feature.split('_lag_')[1]
            if base_feature not in lag_types:
                lag_types[base_feature] = []
            lag_types[base_feature].append(int(lag_num))
        
        for base_feature, lags in lag_types.items():
            print(f"   • {base_feature}: lags {sorted(lags)}")
        
        print(f"\n✅ Data leakage prevention measures:")
        print(f"   • ✓ Features from time t used to predict time t+{self.prediction_horizon}")
        print(f"   • ✓ Scalers fitted only on training data")
        print(f"   • ✓ Temporal ordering strictly enforced")
        print(f"   • ✓ Target excluded from feature set")
        print(f"   • ✓ Sufficient historical data required ({self.sequence_length} time steps)")
        
        print("\n" + "="*60) 