"""
Temporal Graph Dataset for dynamic GNN training.

This module provides dataset classes for handling temporal sequences of graph data
with dynamic adjacency matrices, specifically designed for the Gated GCN model.

Key features:
- Temporal sequence handling without data leakage
- Dynamic graph loading for each time step
- Efficient memory management for large temporal datasets
- Proper train/val/test isolation with temporal integrity
"""

import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import warnings
from sklearn.preprocessing import StandardScaler
import pickle


class TemporalGraphData:
    """
    Container for temporal graph data.
    
    Holds a sequence of node features and corresponding edge indices
    for dynamic graph neural network training.
    """
    
    def __init__(
        self,
        x_seq: torch.Tensor,
        edge_seq: List[torch.Tensor],
        y: torch.Tensor,
        num_nodes: int,
        timestamps: Optional[List] = None
    ):
        """
        Initialize temporal graph data.
        
        Args:
            x_seq: Node features sequence [seq_len, num_nodes, num_features]
            edge_seq: List of edge indices for each time step
            y: Target values [num_nodes, num_targets]
            num_nodes: Number of nodes in the graph
            timestamps: Optional timestamps for each step
        """
        self.x = x_seq
        self.edge_index = edge_seq
        self.y = y
        self.num_nodes = num_nodes
        self.timestamps = timestamps
        
        # validate consistency
        assert len(self.x.shape) == 3, "x_seq must be 3D: [seq_len, num_nodes, num_features]"
        assert len(edge_seq) == x_seq.shape[0], "edge_seq length must match sequence length"
        assert x_seq.shape[1] == num_nodes, "x_seq num_nodes dimension mismatch"
        
    @property
    def seq_len(self) -> int:
        return self.x.shape[0]
    
    @property
    def num_features(self) -> int:
        return self.x.shape[2]
        
    def to(self, device: torch.device):
        """Move data to specified device."""
        self.x = self.x.to(device)
        self.edge_index = [edge.to(device) for edge in self.edge_index]
        self.y = self.y.to(device)
        return self


class TemporalGraphDataset(Dataset):
    """
    Dataset for temporal graph sequences with data leakage prevention.
    
    This dataset ensures that:
    1. Each sequence uses only historical data for prediction
    2. Target data is never used in feature construction
    3. Dynamic graphs are computed using only past correlations
    4. Temporal ordering is preserved across train/val/test splits
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 24,
        prediction_horizon: int = 1,
        correlation_window: int = 168,  # 1 week for correlation calculation
        min_correlation: float = 0.1,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None
    ):
        """
        Initialize temporal graph dataset.
        
        Args:
            data_path: Path to parquet file with temporal data
            sequence_length: Length of input sequences (historical steps)
            prediction_horizon: Steps ahead to predict (typically 1)
            correlation_window: Window size for computing dynamic correlations
            min_correlation: Minimum correlation threshold for edges
            use_cache: Whether to cache processed sequences
            cache_dir: Directory for caching (default: same as data_path)
            exclude_patterns: List of substring patterns; any feature containing
                              one of them will be excluded from the final
                              feature set (e.g. ['_rolling_std_', 'has_']).
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.correlation_window = correlation_window
        self.min_correlation = min_correlation
        self.use_cache = use_cache
        
        # setup cache directory
        if cache_dir is None:
            cache_dir = self.data_path.parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # patterns for optional feature exclusion (redundant or undesired)
        default_exclude = [
            '_rolling_std_',    # high colinearity with rolling mean
            'has_',             # binary activity flags and their lags/rolling
            '_flag_',           # generic flag patterns if present
            '_weather_flag_',   # explicit weather flags if named so
            'cluster_station_count',
            '_rolling_mean_',  # static scalar per node
        ]
        self.exclude_patterns = exclude_patterns if exclude_patterns is not None else default_exclude
        
        # load and process data
        self._load_data()
        self._create_sequences()
        
    def _load_data(self):
        """Load and validate temporal data."""
        print(f"Loading temporal data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # load parquet data
        self.raw_data = pl.read_parquet(self.data_path)
        
        # validate required columns
        required_cols = ['cluster_id', 'ts_start']
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # identify feature and target columns
        self._identify_columns()
        
        # sort by cluster and time to ensure temporal ordering
        self.raw_data = self.raw_data.sort(['cluster_id', 'ts_start'])
        
        print(f"Loaded {self.raw_data.shape[0]} temporal records")
        print(f"Features: {len(self.feature_cols)} columns")
        print(f"Targets: {len(self.target_cols)} columns")
        
    def _identify_columns(self):
        """Identify feature and target columns while preventing data leakage."""
        all_cols = self.raw_data.columns
        
        # exclude metadata columns
        exclude_cols = ['cluster_id', 'ts_start']
        
        # identify target columns (current arrivals - what we want to predict)
        self.target_cols = [
            col for col in all_cols 
            if col.startswith('arr_external_count') or col.startswith('dep_external_count')
            and not any(pattern in col for pattern in ['lag_', 'rolling_', 'same_'])
        ]
        
        # if no external targets found, look for any arrival targets
        if not self.target_cols:
            self.target_cols = [
                col for col in all_cols 
                if (col.startswith('arr_') and 'count' in col and 
                    not any(pattern in col for pattern in ['lag_', 'rolling_', 'same_']))
            ]
        
        # ensure we have at least one target
        if not self.target_cols:
            # fallback to first numeric column that could be a target
            numeric_cols = [col for col in all_cols if col not in exclude_cols]
            if numeric_cols:
                self.target_cols = [numeric_cols[0]]
                warnings.warn(f"No clear target columns found, using {self.target_cols[0]}")
            else:
                raise ValueError("No suitable target columns found")
        
        # identify safe feature columns (no data leakage)
        self.feature_cols = []
        
        # lag features (historical data - safe)
        lag_features = [
            col for col in all_cols 
            if any(pattern in col for pattern in ['_lag_', '_rolling_', '_same_hour_', '_yesterday', '_last_week'])
        ]
        self.feature_cols.extend(lag_features)
        
        # temporal features (always safe as they're about timing, not events)
        temporal_features = [
            col for col in all_cols 
            if any(pattern in col for pattern in [
                'hour_', 'dow_', 'month_', 'is_weekend', 'is_morning_rush', 
                'is_evening_rush', 'is_daytime', 'is_summer', 'is_winter'
            ])
        ]
        self.feature_cols.extend(temporal_features)
        
        # weather features (current conditions known at prediction time - safe)
        weather_features = [
            col for col in all_cols 
            if any(pattern in col for pattern in [
                'temp', 'humidity', 'wind_speed', 'precipitation', 'pressure', 
                'visibility', 'weather_condition', 'feels_like', 'uv_index', 'cloud_cover'
            ])
        ]
        self.feature_cols.extend(weather_features)
        
        # cluster metadata (static information - safe)
        metadata_features = [
            col for col in all_cols 
            if any(pattern in col for pattern in [
                'cluster_centroid_', 'cluster_station_count'
            ])
        ]
        self.feature_cols.extend(metadata_features)
        
        # remove duplicates and ensure all columns exist
        self.feature_cols = list(set(self.feature_cols))
        self.feature_cols = [col for col in self.feature_cols if col in all_cols]
        
        # remove any targets from features (extra safety)
        self.feature_cols = [col for col in self.feature_cols if col not in self.target_cols]
        
        # drop user-defined redundant patterns
        def _matches_exclude(col: str) -> bool:
            return any(pat in col for pat in self.exclude_patterns)

        before_len = len(self.feature_cols)
        self.feature_cols = [col for col in self.feature_cols if not _matches_exclude(col)]
        removed_cnt = before_len - len(self.feature_cols)
        if removed_cnt > 0:
            print(f"🧹 Removed {removed_cnt} redundant features based on exclude_patterns")
        
        print(f"Safe feature columns identified: {len(self.feature_cols)}")
        print(f"Target columns: {self.target_cols}")
        
        if len(self.feature_cols) == 0:
            raise ValueError("No safe feature columns found - check feature engineering")
        
        # validate that no target variables are in safe features (data leakage check)
        unsafe_features = [f for f in self.feature_cols if f in self.target_cols]
        if unsafe_features:
            raise ValueError(f"Data leakage detected: target variables in features: {unsafe_features}")
    
    def _find_safe_correlation_feature(self) -> Optional[str]:
        """
        Find a safe historical feature to use for computing correlations.
        
        This prevents data leakage by using lag features instead of target variables.
        """
        if not self.target_cols:
            return None
        
        target_col = self.target_cols[0]
        correlation_candidates = []
        
        # 1. look for lag versions of the target variable
        if 'arr_external_count' in target_col:
            lag_patterns = ['arr_external_count_lag_', 'arr_total_count_lag_', 'arr_count_lag_']
        elif 'dep_external_count' in target_col:
            lag_patterns = ['dep_external_count_lag_', 'dep_total_count_lag_', 'dep_count_lag_']
        else:
            lag_patterns = ['_lag_1', '_lag_2', '_lag_3']
        
        for pattern in lag_patterns:
            candidates = [col for col in self.feature_cols if pattern in col]
            correlation_candidates.extend(candidates)
        
        # 2. look for rolling statistics of similar variables
        rolling_patterns = [
            'arr_external_count_rolling_', 'dep_external_count_rolling_',
            'arr_total_count_rolling_', 'dep_total_count_rolling_'
        ]
        
        for pattern in rolling_patterns:
            candidates = [col for col in self.feature_cols if pattern in col]
            correlation_candidates.extend(candidates)
        
        # 3. look for any lag feature as fallback
        if not correlation_candidates:
            correlation_candidates = [col for col in self.feature_cols if '_lag_' in col]
        
        # 4. look for any rolling feature as last resort
        if not correlation_candidates:
            correlation_candidates = [col for col in self.feature_cols if '_rolling_' in col]
        
        # return the first available candidate or None
        if correlation_candidates:
            safe_feature = correlation_candidates[0]
            # additional safety check
            if safe_feature in self.target_cols:
                warnings.warn(f"Selected correlation feature {safe_feature} is a target variable!")
                return None
            return safe_feature
        else:
            warnings.warn("No safe correlation feature found - graphs will use no correlations")
            return None
            
    def _create_sequences(self):
        """Create temporal sequences without data leakage."""
        cache_file = self.cache_dir / f"sequences_{self.sequence_length}_{self.prediction_horizon}.pkl"
        
        if self.use_cache and cache_file.exists():
            print(f"Loading cached sequences from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.sequences = pickle.load(f)
            return
        
        print("Creating temporal sequences...")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Prediction horizon: {self.prediction_horizon}")
        
        # get unique clusters and timestamps
        clusters = sorted(self.raw_data['cluster_id'].unique())
        timestamps = sorted(self.raw_data['ts_start'].unique())
        
        self.num_clusters = len(clusters)
        
        # create cluster to index mapping
        self.cluster_to_idx = {cluster: idx for idx, cluster in enumerate(clusters)}
        
        self.sequences = []
        
        # minimum time steps needed for a valid sequence
        min_steps_needed = self.sequence_length + self.prediction_horizon + self.correlation_window
        
        if len(timestamps) < min_steps_needed:
            raise ValueError(
                f"Insufficient temporal data: need {min_steps_needed} steps, "
                f"have {len(timestamps)}"
            )
        
        # create sequences starting from sufficient historical data
        valid_start_idx = self.correlation_window + self.sequence_length
        
        for t_idx in range(valid_start_idx, len(timestamps) - self.prediction_horizon + 1):
            current_ts = timestamps[t_idx]
            target_ts = timestamps[t_idx + self.prediction_horizon - 1]
            
            # get historical sequence (no data leakage)
            seq_timestamps = timestamps[t_idx - self.sequence_length:t_idx]
            
            # extract sequence features (only historical data)
            x_seq = self._extract_sequence_features(seq_timestamps, clusters)
            
            # extract targets (future data - what we want to predict)
            y_target = self._extract_targets(target_ts, clusters)
            
            # create dynamic graphs for historical sequence
            edge_seq = self._create_dynamic_graphs(seq_timestamps, clusters)
            
            # create temporal graph data object
            temporal_data = TemporalGraphData(
                x_seq=torch.tensor(x_seq, dtype=torch.float32),
                edge_seq=edge_seq,
                y=torch.tensor(y_target, dtype=torch.float32),
                num_nodes=self.num_clusters,
                timestamps=seq_timestamps + [target_ts]
            )
            
            self.sequences.append(temporal_data)
        
        print(f"Created {len(self.sequences)} temporal sequences")
        
        # cache sequences
        if self.use_cache:
            print(f"Caching sequences to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.sequences, f)
    
    def _extract_sequence_features(self, timestamps: List, clusters: List) -> np.ndarray:
        """Extract feature matrix for a temporal sequence."""
        seq_len = len(timestamps)
        num_features = len(self.feature_cols)
        
        # initialize feature matrix
        x_seq = np.zeros((seq_len, self.num_clusters, num_features))
        
        for t_idx, ts in enumerate(timestamps):
            # get data for this timestamp
            ts_data = self.raw_data.filter(pl.col('ts_start') == ts)
            
            if ts_data.height == 0:
                # no data for this timestamp, keep zeros
                continue
            
            # extract features
            for cluster_id in clusters:
                cluster_idx = self.cluster_to_idx[cluster_id]
                cluster_data = ts_data.filter(pl.col('cluster_id') == cluster_id)
                
                if cluster_data.height > 0:
                    # extract feature values
                    feature_values = cluster_data.select(self.feature_cols).to_numpy()[0]
                    x_seq[t_idx, cluster_idx, :] = feature_values
        
        # handle NaN values
        x_seq = np.nan_to_num(x_seq, nan=0.0)
        
        return x_seq
    
    def _extract_targets(self, target_ts, clusters: List) -> np.ndarray:
        """Extract target values for prediction."""
        # get data for target timestamp
        target_data = self.raw_data.filter(pl.col('ts_start') == target_ts)
        
        # initialize target array
        y_target = np.zeros((self.num_clusters, len(self.target_cols)))
        
        if target_data.height > 0:
            for cluster_id in clusters:
                cluster_idx = self.cluster_to_idx[cluster_id]
                cluster_data = target_data.filter(pl.col('cluster_id') == cluster_id)
                
                if cluster_data.height > 0:
                    # extract target values
                    target_values = cluster_data.select(self.target_cols).to_numpy()[0]
                    y_target[cluster_idx, :] = target_values
        
        # handle NaN values
        y_target = np.nan_to_num(y_target, nan=0.0)
        
        # if single target, squeeze last dimension
        if len(self.target_cols) == 1:
            y_target = y_target.squeeze(-1)
        
        return y_target
    
    def _create_dynamic_graphs(self, timestamps: List, clusters: List) -> List[torch.Tensor]:
        """Create dynamic adjacency matrices based on historical correlations."""
        edge_seq = []
        
        for ts in timestamps:
            # for each timestamp, compute correlations using historical data
            # (correlation_window steps before current timestamp)
            correlation_data = self._get_correlation_data(ts, clusters)
            
            if correlation_data is not None:
                edge_index = self._compute_correlation_graph(correlation_data, clusters)
            else:
                # fallback to empty graph if no correlation data
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            edge_seq.append(edge_index)
        
        return edge_seq
    
    def _get_correlation_data(self, current_ts, clusters: List) -> Optional[np.ndarray]:
        """Get historical data for correlation computation using safe lag features."""
        # get all timestamps before current_ts
        all_timestamps = sorted(self.raw_data['ts_start'].unique())
        try:
            current_idx = all_timestamps.index(current_ts)
        except ValueError:
            return None  # timestamp not found
        
        if current_idx < self.correlation_window:
            return None  # insufficient data for correlation
        
        # get historical window
        hist_timestamps = all_timestamps[current_idx - self.correlation_window:current_idx]
        
        # find safe historical feature for correlation (no data leakage)
        safe_correlation_feature = self._find_safe_correlation_feature()
        
        if safe_correlation_feature is None:
            return None  # no safe feature available
        
        # extract historical feature data for correlation
        correlation_data = np.zeros((len(hist_timestamps), len(clusters)))
        
        for t_idx, ts in enumerate(hist_timestamps):
            ts_data = self.raw_data.filter(pl.col('ts_start') == ts)
            
            for cluster_id in clusters:
                cluster_idx = self.cluster_to_idx[cluster_id]
                cluster_data = ts_data.filter(pl.col('cluster_id') == cluster_id)
                
                if cluster_data.height > 0:
                    # use safe historical feature for correlation (no data leakage)
                    if safe_correlation_feature in cluster_data.columns:
                        feature_value = cluster_data.select(safe_correlation_feature).to_numpy()[0, 0]
                        correlation_data[t_idx, cluster_idx] = feature_value
        
        return correlation_data
    
    def _compute_correlation_graph(self, correlation_data: np.ndarray, clusters: List) -> torch.Tensor:
        """Compute graph edges based on demand correlations."""
        num_clusters = len(clusters)
        
        # compute correlation matrix
        corr_matrix = np.corrcoef(correlation_data.T)
        
        # handle NaN correlations (replace with 0)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # create edges based on correlation threshold
        edges = []
        
        for i in range(num_clusters):
            for j in range(num_clusters):
                if i != j and abs(corr_matrix[i, j]) >= self.min_correlation:
                    edges.append([i, j])
        
        # convert to edge_index format
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> TemporalGraphData:
        return self.sequences[idx]
    
    def get_data_info(self) -> Dict:
        """Get dataset information."""
        if len(self.sequences) == 0:
            return {}
        
        sample = self.sequences[0]
        
        return {
            'num_sequences': len(self.sequences),
            'sequence_length': sample.seq_len,
            'num_nodes': sample.num_nodes,
            'num_features': sample.num_features,
            'num_targets': len(self.target_cols),
            'feature_columns': self.feature_cols,
            'target_columns': self.target_cols
        }


def create_temporal_splits(
    train_path: str,
    val_path: str,
    test_path: str,
    **dataset_kwargs
) -> Tuple[TemporalGraphDataset, TemporalGraphDataset, TemporalGraphDataset]:
    """
    Create temporal graph datasets for train/val/test splits.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data  
        test_path: Path to test data
        **dataset_kwargs: Additional arguments for TemporalGraphDataset
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("Creating temporal graph datasets...")
    
    train_dataset = TemporalGraphDataset(train_path, **dataset_kwargs)
    val_dataset = TemporalGraphDataset(val_path, **dataset_kwargs)
    test_dataset = TemporalGraphDataset(test_path, **dataset_kwargs)
    
    # print dataset information
    train_info = train_dataset.get_data_info()
    val_info = val_dataset.get_data_info()
    test_info = test_dataset.get_data_info()
    
    print(f"Train dataset: {train_info.get('num_sequences', 0)} sequences")
    print(f"Val dataset: {val_info.get('num_sequences', 0)} sequences")
    print(f"Test dataset: {test_info.get('num_sequences', 0)} sequences")
    
    return train_dataset, val_dataset, test_dataset 