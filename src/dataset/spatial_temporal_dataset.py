import polars as pl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from pathlib import Path
from typing import Tuple, List, Optional
import json
import logging
from sklearn.preprocessing import StandardScaler

# setup module logger
logger = logging.getLogger(__name__)

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
        logger.info("🚀 Initializing EcoBiciSpatialTemporalDataset")
        logger.debug(f"Parameters: data_path={data_path}, sequence_length={sequence_length}, "
                    f"prediction_horizon={prediction_horizon}, k_neighbors={k_neighbors}")
        
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.k_neighbors = k_neighbors
        self.n_clusters = 93
        
        logger.info(f"🏗️  Configuration: {self.n_clusters} clusters, {self.k_neighbors} neighbors per cluster")
        
        # load metadata
        logger.debug("📄 Loading dataset metadata...")
        self.metadata = self._load_metadata()
        logger.info("✅ Metadata loaded successfully")
        
        # feature groups to use (only external features)
        logger.debug("🔧 Setting up feature columns...")
        self.feature_columns = self._get_external_feature_columns()
        logger.info(f"✅ Feature configuration: {len(self.feature_columns)} total columns")
        
        # scalers for normalization
        logger.debug("⚖️  Initializing data scalers...")
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        logger.info("✅ Scalers initialized")
        
        logger.info("🎉 EcoBiciSpatialTemporalDataset initialization complete")
        
    def _load_metadata(self) -> dict:
        """Load dataset metadata"""
        metadata_path = self.data_path / "clustered" / "dataset_metadata.json"
        logger.debug(f"📂 Loading metadata from: {metadata_path}")
        
        if not metadata_path.exists():
            logger.error(f"❌ Metadata file not found: {metadata_path}")
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✅ Successfully loaded metadata ({len(metadata)} keys)")
            return metadata
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            raise
    
    def _get_external_feature_columns(self) -> List[str]:
        """
        Get list of external feature columns, excluding internal (phantom) trips
        """
        logger.debug("🏗️  Building external feature column list...")
        
        # base columns
        base_cols = ['cluster_id', 'ts_start']
        logger.debug(f"   📋 Base columns: {len(base_cols)}")
        
        # target column (what we want to predict)
        target_cols = ['arr_external_count']
        logger.debug(f"   🎯 Target columns: {len(target_cols)}")
        
        # external departure features
        external_dep_cols = [
            'dep_external_count', 'dep_external_male', 'dep_external_female', 
            'dep_external_other', 'dep_external_young', 'dep_external_adult', 
            'dep_external_senior', 'dep_external_age_available', 'dep_external_model_iconic',
            'dep_external_duration_mean', 'dep_external_duration_std'
        ]
        logger.debug(f"   🚗 External departure features: {len(external_dep_cols)}")
        
        # weather features (general, not specific to internal/external)
        weather_cols = [
            'temperature_2m_mean', 'relative_humidity_2m_mean', 'dew_point_2m_mean',
            'apparent_temperature_mean', 'precipitation_mean', 'weather_code_mean',
            'pressure_msl_mean', 'cloud_cover_mean', 'wind_speed_10m_mean',
            'wind_direction_10m_mean', 'wind_gusts_10m_mean', 'is_day',
            'temp_comfortable', 'temp_very_hot', 'temp_cold', 'is_raining',
            'heavy_rain', 'light_rain', 'strong_wind', 'moderate_wind'
        ]
        logger.debug(f"   🌤️  Weather features: {len(weather_cols)}")
        
        # temporal features
        temporal_cols = [
            'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_daytime',
            'is_summer', 'is_winter'
        ]
        logger.debug(f"   ⏰ Temporal features: {len(temporal_cols)}")
        
        # spatial features
        spatial_cols = [
            'cluster_centroid_lat', 'cluster_centroid_lon', 'cluster_station_count'
        ]
        logger.debug(f"   🗺️  Spatial features: {len(spatial_cols)}")
        
        # lag features (only external)
        lag_cols = []
        for feature in ['dep_external_count', 'arr_external_count']:
            for lag in [1, 2, 4, 8]:
                lag_cols.append(f"{feature}_lag_{lag}")
        logger.debug(f"   ⏮️  Lag features: {len(lag_cols)}")
        
        # rolling features (only external)
        rolling_cols = []
        for feature in ['dep_external_count', 'arr_external_count']:
            for window in [4, 8, 24]:
                rolling_cols.extend([
                    f"{feature}_rolling_mean_{window}",
                    f"{feature}_rolling_std_{window}"
                ])
        logger.debug(f"   📊 Rolling features: {len(rolling_cols)}")
        
        # same hour features (only external)
        same_hour_cols = []
        for feature in ['dep_external_count', 'arr_external_count']:
            same_hour_cols.extend([
                f"{feature}_same_hour_yesterday",
                f"{feature}_same_hour_last_week"
            ])
        logger.debug(f"   🕐 Same-hour features: {len(same_hour_cols)}")
        
        total_cols = (base_cols + target_cols + external_dep_cols + weather_cols + 
                     temporal_cols + spatial_cols + lag_cols + rolling_cols + same_hour_cols)
        
        logger.info(f"🏗️  Feature column construction complete: {len(total_cols)} total columns")
        logger.debug("   📋 Feature breakdown: base(2) + target(1) + external_dep(11) + weather(18) + "
                    f"temporal(15) + spatial(3) + lag(8) + rolling(12) + same_hour(4)")
        
        return total_cols
    
    def _create_graph_structure(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph structure based on spatial proximity of clusters.
        Returns edge_index and edge_weights tensors.
        """
        logger.info("🕸️  Creating graph structure based on spatial proximity...")
        logger.debug(f"Building k-NN graph with k={self.k_neighbors} for {self.n_clusters} clusters")
        
        # get cluster centroids
        logger.debug("📍 Extracting cluster centroids from metadata...")
        centroids = []
        for i in range(self.n_clusters):
            if str(i) not in self.metadata['cluster_centroids']:
                logger.warning(f"⚠️  Missing centroid for cluster {i}")
                continue
            cluster_info = self.metadata['cluster_centroids'][str(i)]
            centroids.append([cluster_info['lat'], cluster_info['lon']])
        
        centroids = np.array(centroids)
        logger.info(f"✅ Extracted {len(centroids)} cluster centroids")
        logger.debug(f"   📍 Coordinate bounds: lat=[{centroids[:, 0].min():.4f}, {centroids[:, 0].max():.4f}], "
                    f"lon=[{centroids[:, 1].min():.4f}, {centroids[:, 1].max():.4f}]")
        
        # create edges based on k-nearest neighbors
        logger.debug(f"🔗 Building edges using {self.k_neighbors}-nearest neighbors...")
        edge_list = []
        edge_weights = []
        
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
        
        logger.debug(f"   🔗 Created {len(edge_list)} directed edges")
        
        # convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        logger.info(f"✅ Graph structure created: {edge_index.shape[1]} edges, "
                   f"avg degree: {edge_index.shape[1]/self.n_clusters:.1f}")
        logger.debug(f"   📊 Edge weights range: [{edge_weights.min():.6f}, {edge_weights.max():.6f}]")
        
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
        logger.info("🔄 Preparing temporal sequences with leakage prevention...")
        logger.debug(f"Input dataframe shape: {df.shape}")
        
        # sort by time and cluster to ensure temporal ordering
        logger.debug("📅 Sorting data by timestamp and cluster...")
        df = df.sort(['ts_start', 'cluster_id'])
        
        # get unique timestamps in chronological order
        timestamps = df['ts_start'].unique().sort()
        logger.info(f"⏰ Processing {len(timestamps)} unique timestamps")
        logger.debug(f"   📅 Date range: {timestamps[0]} to {timestamps[-1]}")
        
        # prepare feature columns (excluding base columns and current target)
        feature_cols = [col for col in self.feature_columns 
                       if col not in ['cluster_id', 'ts_start', 'arr_external_count']]
        logger.info(f"🔧 Using {len(feature_cols)} feature columns (excluding base and target)")
        
        features_list = []
        targets_list = []
        skipped_count = 0
        nan_inf_count = 0
        
        # start from sequence_length to ensure we have enough history
        # this prevents data leakage by requiring sufficient past data
        logger.info(f"🔒 Starting from timestamp {self.sequence_length} to ensure sufficient history")
        valid_range = range(self.sequence_length, len(timestamps) - self.prediction_horizon)
        logger.debug(f"   📊 Valid timestamp range: {len(valid_range)} out of {len(timestamps)} timestamps")
        
        for i in valid_range:
            # get current timestamp for target
            current_ts = timestamps[i]
            target_ts = timestamps[i + self.prediction_horizon]  # predict future arrivals
            
            logger.debug(f"   ⏳ Processing timestamp {i}: {current_ts} → {target_ts}")
            
            # get current features (at time t)
            current_data = df.filter(pl.col('ts_start') == current_ts).sort('cluster_id')
            
            # get target data (at time t + prediction_horizon)
            target_data = df.filter(pl.col('ts_start') == target_ts).sort('cluster_id')
            
            # ensure we have data for all clusters at both timestamps
            if len(current_data) != self.n_clusters or len(target_data) != self.n_clusters:
                logger.debug(f"   ⏭️  Skipping timestamp {i}: incomplete data "
                           f"(current: {len(current_data)}, target: {len(target_data)}, expected: {self.n_clusters})")
                skipped_count += 1
                continue
            
            # verify temporal ordering (critical for preventing leakage)
            if current_data['ts_start'].min() >= target_data['ts_start'].min():
                logger.error(f"❌ Data leakage detected: features timestamp {current_data['ts_start'].min()} "
                           f">= target timestamp {target_data['ts_start'].min()}")
                raise ValueError(f"Data leakage detected: features timestamp {current_data['ts_start'].min()} >= target timestamp {target_data['ts_start'].min()}")
            
            # extract features (shape: [n_clusters, n_features])
            # these features are from time t, used to predict time t+prediction_horizon
            features = current_data.select(feature_cols).to_numpy().astype(np.float32)
            
            # extract targets (shape: [n_clusters])
            # these are the arrivals we want to predict at time t+prediction_horizon
            targets = target_data.select('arr_external_count').to_numpy().flatten().astype(np.float32)
            
            # verify no NaN or infinite values that could indicate data issues
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning(f"⚠️  NaN/Inf values in features at timestamp {current_ts} - skipping")
                nan_inf_count += 1
                continue
                
            if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
                logger.warning(f"⚠️  NaN/Inf values in targets at timestamp {target_ts} - skipping")
                nan_inf_count += 1
                continue
            
            features_list.append(torch.tensor(features))
            targets_list.append(torch.tensor(targets))
        
        logger.info(f"✅ Temporal sequence preparation complete:")
        logger.info(f"   📊 Valid samples: {len(features_list)}")
        logger.info(f"   ⏭️  Skipped (incomplete): {skipped_count}")
        logger.info(f"   🚫 Skipped (NaN/Inf): {nan_inf_count}")
        logger.info(f"   🔒 Each sample uses features from time t to predict arrivals at time t+{self.prediction_horizon}")
        
        if len(features_list) == 0:
            logger.error("❌ No valid temporal sequences generated!")
            raise ValueError("No valid temporal sequences could be generated from the data")
        
        return features_list, targets_list
    
    def load_data(self, split: str = 'train') -> StaticGraphTemporalSignal:
        """
        Load and prepare data for the specified split with strict data leakage prevention.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            StaticGraphTemporalSignal object for PyTorch Geometric Temporal
        """
        logger.info(f"🔄 Loading and preparing {split} data...")
        
        # load data
        file_path = self.data_path / "clustered" / f"{split}_cluster_features.parquet"
        logger.debug(f"📂 Loading from: {file_path}")
        
        if not file_path.exists():
            logger.error(f"❌ Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.debug("📖 Reading parquet file...")
        df = pl.read_parquet(file_path)
        logger.info(f"✅ Loaded {len(df):,} rows for {split} split")
        logger.debug(f"   📊 Data shape: {df.shape}")
        logger.debug(f"   🏷️  Columns: {len(df.columns)}")
        
        # verify temporal ordering
        logger.debug("⏰ Verifying temporal structure...")
        timestamps = df['ts_start'].unique().sort()
        logger.info(f"   📅 Found {len(timestamps)} unique timestamps")
        
        if len(timestamps) < self.sequence_length + self.prediction_horizon:
            error_msg = f"Insufficient timestamps for {split}: {len(timestamps)} < {self.sequence_length + self.prediction_horizon}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # check for temporal continuity (detect gaps that could indicate data issues)
        logger.debug("🔍 Checking temporal continuity...")
        time_diffs = df.sort('ts_start')['ts_start'].diff().drop_nulls().unique()
        expected_diff = time_diffs.mode()[0] if len(time_diffs) > 0 else None
        logger.debug(f"   ⏱️  Time differences found: {len(time_diffs)} unique intervals")
        if expected_diff:
            logger.debug(f"   📊 Most common interval: {expected_diff}")
        
        # filter to only include relevant columns
        logger.debug("🔧 Filtering to relevant columns...")
        available_cols = [col for col in self.feature_columns if col in df.columns]
        missing_cols = set(self.feature_columns) - set(available_cols)
        
        if missing_cols:
            logger.warning(f"⚠️  Missing columns in {split}: {sorted(missing_cols)}")
            logger.debug(f"   📊 Available: {len(available_cols)}/{len(self.feature_columns)} columns")
        else:
            logger.info(f"✅ All {len(self.feature_columns)} expected columns found")
        
        df = df.select(available_cols)
        
        # handle missing values (fill with 0 to maintain temporal structure)
        logger.debug("🔧 Handling missing values...")
        null_counts = df.null_count()
        total_nulls = sum([count for count in null_counts.row(0)])
        if total_nulls > 0:
            logger.warning(f"⚠️  Found {total_nulls:,} null values - filling with 0")
            df = df.fill_null(0)
        else:
            logger.debug("   ✅ No null values found")
        
        # create graph structure (static for all timestamps)
        logger.info("🕸️  Creating graph structure...")
        edge_index, edge_weights = self._create_graph_structure()
        
        # prepare temporal data with leakage prevention
        logger.info("🔄 Preparing temporal sequences...")
        features_list, targets_list = self._prepare_temporal_data(df)
        
        if len(features_list) == 0:
            error_msg = f"No valid temporal sequences generated for {split}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # CRITICAL: Only fit scalers on training data to prevent data leakage
        if split == 'train':
            logger.info("🔧 Fitting scalers on training data only...")
            # concatenate all training features and targets
            all_features = torch.cat(features_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            
            logger.debug(f"   📊 Training data for scaling: {all_features.shape[0]:,} samples")
            logger.debug(f"   🎯 Feature shape: {all_features.shape}")
            logger.debug(f"   🎯 Target shape: {all_targets.shape}")
            
            # fit scalers only on training data
            logger.debug("   ⚖️  Fitting feature scaler...")
            self.feature_scaler.fit(all_features.numpy())
            logger.debug("   ⚖️  Fitting target scaler...")
            self.target_scaler.fit(all_targets.numpy().reshape(-1, 1))
            
            logger.info(f"✅ Scalers fitted successfully:")
            logger.info(f"   📊 Feature scaler: {all_features.shape[0]:,} samples × {all_features.shape[1]} features")
            logger.info(f"   🎯 Target scaler: {all_targets.shape[0]:,} samples")
            
            # log scaling statistics
            logger.debug(f"   📈 Feature stats: mean={self.feature_scaler.mean_[:5]}, std={self.feature_scaler.scale_[:5]} (first 5)")
            logger.debug(f"   📈 Target stats: mean={self.target_scaler.mean_[0]:.4f}, std={self.target_scaler.scale_[0]:.4f}")
            
        elif not hasattr(self.feature_scaler, 'mean_'):
            error_msg = f"Scalers not fitted! Must load 'train' split first before loading '{split}'"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        else:
            logger.info(f"✅ Using pre-fitted scalers for {split} split")
        
        # apply normalization using training statistics
        logger.info("⚖️  Applying normalization using training statistics...")
        normalized_features = []
        normalized_targets = []
        
        for i, (features, targets) in enumerate(zip(features_list, targets_list)):
            logger.debug(f"   🔧 Normalizing sample {i+1}/{len(features_list)}")
            
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
                logger.error(f"❌ NaN or Inf values detected in normalized features at sample {i}")
                raise ValueError("NaN or Inf values detected in normalized features")
            if torch.any(torch.isnan(norm_targets)) or torch.any(torch.isinf(norm_targets)):
                logger.error(f"❌ NaN or Inf values detected in normalized targets at sample {i}")
                raise ValueError("NaN or Inf values detected in normalized targets")
            
            normalized_features.append(norm_features)
            normalized_targets.append(norm_targets)
        
        logger.info("✅ Normalization complete - all samples validated")
        
        # create StaticGraphTemporalSignal
        logger.info("🏗️  Creating StaticGraphTemporalSignal...")
        dataset = StaticGraphTemporalSignal(
            edge_index=edge_index,
            edge_weight=edge_weights,
            features=normalized_features,
            targets=normalized_targets
        )
        
        # final validation and summary
        logger.info(f"✅ Successfully created {split} dataset:")
        logger.info(f"   🔢 Temporal sequences: {len(dataset)}")
        logger.info(f"   🏠 Nodes per graph: {edge_index.max().item() + 1}")
        logger.info(f"   🔗 Edges per graph: {edge_index.shape[1]}")
        logger.info(f"   📊 Features per node: {normalized_features[0].shape[1] if normalized_features else 0}")
        logger.info(f"   🔮 Prediction horizon: {self.prediction_horizon} time steps")
        
        if normalized_features:
            sample_feature_stats = normalized_features[0]
            logger.debug(f"   📈 Sample feature range: [{sample_feature_stats.min():.4f}, {sample_feature_stats.max():.4f}]")
        if normalized_targets:
            sample_target_stats = normalized_targets[0]
            logger.debug(f"   🎯 Sample target range: [{sample_target_stats.min():.4f}, {sample_target_stats.max():.4f}]")
        
        return dataset
    
    def inverse_transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform normalized targets back to original scale.
        """
        logger.debug(f"🔄 Inverse transforming targets: {targets.shape}")
        
        targets_np = targets.detach().cpu().numpy()
        if targets_np.ndim == 1:
            targets_np = targets_np.reshape(-1, 1)
        
        original_targets = self.target_scaler.inverse_transform(targets_np)
        result = torch.tensor(original_targets.flatten(), dtype=torch.float32)
        
        logger.debug(f"   ✅ Inverse transform complete: {result.shape}")
        return result
    
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
        logger.info(f"🔍 Validating data leakage for {split} split...")
        logger.debug(f"   📊 Sample size: {sample_size}")
        
        # load raw data
        file_path = self.data_path / "clustered" / f"{split}_cluster_features.parquet"
        logger.debug(f"📂 Loading validation data from: {file_path}")
        df = pl.read_parquet(file_path)
        
        # sort by time and cluster
        logger.debug("📅 Sorting data for temporal validation...")
        df = df.sort(['ts_start', 'cluster_id'])
        timestamps = df['ts_start'].unique().sort()
        logger.debug(f"   ⏰ Found {len(timestamps)} timestamps for validation")
        
        # sample random timestamps for validation
        import random
        valid_timestamps = list(timestamps[10:-10])  # avoid edge cases
        if len(valid_timestamps) < sample_size:
            sample_size = len(valid_timestamps)
            logger.warning(f"⚠️  Reduced sample size to {sample_size} (limited timestamps)")
        
        sample_timestamps = random.sample(valid_timestamps, sample_size)
        logger.info(f"🎲 Testing {len(sample_timestamps)} random timestamps")
        
        violations = []
        
        for idx, ts in enumerate(sample_timestamps):
            if idx % 100 == 0:
                logger.debug(f"   🔍 Progress: {idx}/{len(sample_timestamps)} timestamps checked")
                
            ts_data = df.filter(pl.col('ts_start') == ts).sort('cluster_id')
            
            if len(ts_data) != self.n_clusters:
                logger.debug(f"   ⏭️  Skipping timestamp {ts}: incomplete data ({len(ts_data)}/{self.n_clusters})")
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
            logger.error(f"❌ Data leakage detected! Found {len(violations)} violations:")
            for v in violations[:5]:  # show first 5 violations
                logger.error(f"   • {v['feature']}_lag_{v['lag']} at {v['timestamp']} doesn't match actual values at {v['expected_timestamp']}")
                logger.error(f"     Lag values: {v['lag_values_sample']}")
                logger.error(f"     Actual values: {v['actual_values_sample']}")
            
            raise ValueError(f"Data leakage detected in lag features! Found {len(violations)} violations. Check feature engineering pipeline.")
        
        logger.info(f"✅ No data leakage detected in lag features")
        logger.info(f"   📊 Validated {len(sample_timestamps)} timestamps successfully")
        return True
    
    def get_temporal_split_dates(self) -> dict:
        """
        Get the date ranges for each split to verify proper temporal splitting.
        """
        logger.debug("📅 Gathering temporal split information...")
        splits_info = {}
        
        for split in ['train', 'val', 'test']:
            file_path = self.data_path / "clustered" / f"{split}_cluster_features.parquet"
            if file_path.exists():
                logger.debug(f"   📂 Analyzing {split} split...")
                df = pl.read_parquet(file_path)
                splits_info[split] = {
                    'start_date': df['ts_start'].min(),
                    'end_date': df['ts_start'].max(),
                    'n_timestamps': df['ts_start'].n_unique(),
                    'n_samples': len(df)
                }
            else:
                logger.debug(f"   ⏭️  {split} split file not found")
        
        logger.debug(f"✅ Gathered info for {len(splits_info)} splits")
        return splits_info
    
    def print_data_leakage_report(self) -> None:
        """
        Print a comprehensive report about potential data leakage.
        """
        logger.info("=" * 60)
        logger.info("🛡️  DATA LEAKAGE PREVENTION REPORT")
        logger.info("=" * 60)
        
        # check temporal split boundaries
        logger.info("📅 Temporal Split Boundaries:")
        splits_info = self.get_temporal_split_dates()
        
        prev_end = None
        for split in ['train', 'val', 'test']:
            if split in splits_info:
                info = splits_info[split]
                logger.info(f"   {split.upper():>5}: {info['start_date']} to {info['end_date']} ({info['n_timestamps']} timestamps)")
                
                if prev_end and info['start_date'] <= prev_end:
                    logger.warning(f"   ⚠️  WARNING: {split} starts before previous split ends!")
                    
                prev_end = info['end_date']
        
        # check feature configuration
        logger.info("🔧 Model Configuration:")
        logger.info(f"   • Sequence length: {self.sequence_length}")
        logger.info(f"   • Prediction horizon: {self.prediction_horizon}")
        logger.info(f"   • K-neighbors: {self.k_neighbors}")
        
        # check lag features
        logger.info("🔄 Lag Features Configuration:")
        lag_features = [col for col in self.feature_columns if '_lag_' in col]
        logger.info(f"   • Total lag features: {len(lag_features)}")
        
        lag_types = {}
        for feature in lag_features:
            base_feature = feature.split('_lag_')[0]
            lag_num = feature.split('_lag_')[1]
            if base_feature not in lag_types:
                lag_types[base_feature] = []
            lag_types[base_feature].append(int(lag_num))
        
        for base_feature, lags in lag_types.items():
            logger.info(f"   • {base_feature}: lags {sorted(lags)}")
        
        logger.info("✅ Data leakage prevention measures:")
        logger.info(f"   • ✓ Features from time t used to predict time t+{self.prediction_horizon}")
        logger.info(f"   • ✓ Scalers fitted only on training data")
        logger.info(f"   • ✓ Temporal ordering strictly enforced")
        logger.info(f"   • ✓ Target excluded from feature set")
        logger.info(f"   • ✓ Sufficient historical data required ({self.sequence_length} time steps)")
        
        logger.info("=" * 60) 