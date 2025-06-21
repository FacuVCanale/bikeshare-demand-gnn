"""
Feature selection and filtering utilities for the Gate model.
"""
from typing import List, Dict, Set
import polars as pl
import pandas as pd


class GateFeatureSelector:
    """
    Feature selector specifically designed for the Gate model.
    Filters features based on the Gate model requirements.
    """
    
    def __init__(self):
        self.selected_features = None
        self.feature_groups = self._define_feature_groups()
    
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """
        Define feature groups for the Gate model based on requirements.
        """
        return {
            # core identification and geometry
            'cluster_info': [
                'cluster_id',
                'cluster_centroid_lat', 
                'cluster_centroid_lon', 
                'cluster_station_count'
            ],
            
            # temporal features
            'temporal_base': [
                'hour', 'day_of_week', 'month', 'year'
            ],
            
            'temporal_cyclic': [
                'hour_sin', 'hour_cos',
                'dow_sin', 'dow_cos', 
                'month_sin', 'month_cos'
            ],
            
            'temporal_flags': [
                'is_weekend', 'is_morning_rush', 'is_evening_rush',
                'is_daytime', 'is_summer', 'is_winter'
            ],
            
            # activity aggregated features
            'activity_counts': [
                'dep_internal_count', 'dep_external_count',
                'arr_internal_count', 'arr_external_count'
            ],
            
            'activity_flags': [
                'has_dep_internal', 'has_dep_external',
                'has_arr_internal', 'has_arr_external'
            ],
            
            # weather features (global, not trip-specific)
            'weather_base': [
                'temperature_2m_mean', 'relative_humidity_2m_mean',
                'dew_point_2m_mean', 'apparent_temperature_mean',
                'precipitation_mean', 'weather_code_mean',
                'pressure_msl_mean'
            ],
            
            'weather_cloud': [
                'cloud_cover_mean', 'cloud_cover_low_mean',
                'cloud_cover_mid_mean', 'cloud_cover_high_mean'
            ],
            
            'weather_wind': [
                'vapour_pressure_deficit_mean', 'wind_speed_10m_mean',
                'wind_gusts_10m_mean'
            ],
            
            'weather_solar': [
                'sunshine_duration_mean', 'direct_radiation_mean'
            ],
            
            # lag features (only for core activity counts)
            'activity_lags': [
                'dep_internal_count_lag_1', 'dep_internal_count_lag_2',
                'dep_internal_count_lag_4', 'dep_internal_count_lag_8',
                'dep_external_count_lag_1', 'dep_external_count_lag_2', 
                'dep_external_count_lag_4', 'dep_external_count_lag_8',
                'arr_internal_count_lag_1', 'arr_internal_count_lag_2',
                'arr_internal_count_lag_4', 'arr_internal_count_lag_8',
                'arr_external_count_lag_1', 'arr_external_count_lag_2',
                'arr_external_count_lag_4', 'arr_external_count_lag_8'
            ],
            
            'flag_lags': [
                'has_dep_internal_lag_1', 'has_dep_internal_lag_2',
                'has_dep_internal_lag_4', 'has_dep_internal_lag_8',
                'has_dep_external_lag_1', 'has_dep_external_lag_2',
                'has_dep_external_lag_4', 'has_dep_external_lag_8',
                'has_arr_internal_lag_1', 'has_arr_internal_lag_2',
                'has_arr_internal_lag_4', 'has_arr_internal_lag_8',
                'has_arr_external_lag_1', 'has_arr_external_lag_2',
                'has_arr_external_lag_4', 'has_arr_external_lag_8'
            ],
            
            # rolling features (only for core activity counts)  
            'activity_rolling': [
                'dep_internal_count_rolling_mean_4', 'dep_internal_count_rolling_std_4',
                'dep_internal_count_rolling_mean_8', 'dep_internal_count_rolling_std_8',
                'dep_internal_count_rolling_mean_24', 'dep_internal_count_rolling_std_24',
                'dep_external_count_rolling_mean_4', 'dep_external_count_rolling_std_4',
                'dep_external_count_rolling_mean_8', 'dep_external_count_rolling_std_8',
                'dep_external_count_rolling_mean_24', 'dep_external_count_rolling_std_24',
                'arr_internal_count_rolling_mean_4', 'arr_internal_count_rolling_std_4',
                'arr_internal_count_rolling_mean_8', 'arr_internal_count_rolling_std_8',
                'arr_internal_count_rolling_mean_24', 'arr_internal_count_rolling_std_24',
                'arr_external_count_rolling_mean_4', 'arr_external_count_rolling_std_4',
                'arr_external_count_rolling_mean_8', 'arr_external_count_rolling_std_8',
                'arr_external_count_rolling_mean_24', 'arr_external_count_rolling_std_24'
            ],
            
            'flag_rolling': [
                'has_dep_internal_rolling_mean_4', 'has_dep_internal_rolling_std_4',
                'has_dep_internal_rolling_mean_8', 'has_dep_internal_rolling_std_8',
                'has_dep_internal_rolling_mean_24', 'has_dep_internal_rolling_std_24',
                'has_dep_external_rolling_mean_4', 'has_dep_external_rolling_std_4',
                'has_dep_external_rolling_mean_8', 'has_dep_external_rolling_std_8',
                'has_dep_external_rolling_mean_24', 'has_dep_external_rolling_std_24',
                'has_arr_internal_rolling_mean_4', 'has_arr_internal_rolling_std_4',
                'has_arr_internal_rolling_mean_8', 'has_arr_internal_rolling_std_8',
                'has_arr_internal_rolling_mean_24', 'has_arr_internal_rolling_std_24',
                'has_arr_external_rolling_mean_4', 'has_arr_external_rolling_std_4',
                'has_arr_external_rolling_mean_8', 'has_arr_external_rolling_std_8',
                'has_arr_external_rolling_mean_24', 'has_arr_external_rolling_std_24'
            ],
            
            # same hour features (only for core activity counts)
            'activity_same_hour': [
                'dep_internal_count_same_hour_yesterday', 'dep_internal_count_same_hour_last_week',
                'dep_external_count_same_hour_yesterday', 'dep_external_count_same_hour_last_week',
                'arr_internal_count_same_hour_yesterday', 'arr_internal_count_same_hour_last_week',
                'arr_external_count_same_hour_yesterday', 'arr_external_count_same_hour_last_week'
            ],
            
            'flag_same_hour': [
                'has_dep_internal_same_hour_yesterday', 'has_dep_internal_same_hour_last_week',
                'has_dep_external_same_hour_yesterday', 'has_dep_external_same_hour_last_week',
                'has_arr_internal_same_hour_yesterday', 'has_arr_internal_same_hour_last_week',
                'has_arr_external_same_hour_yesterday', 'has_arr_external_same_hour_last_week'
            ],
            
            # lag availability flags
            'lag_flags': [
                'has_short_term_lags', 'has_daily_lags', 'has_weekly_lags'
            ]
        }
    
    def get_gate_features(self, include_groups: List[str] = None) -> List[str]:
        """
        Get the list of features to use for the Gate model.
        
        Args:
            include_groups: List of feature groups to include. If None, includes all recommended groups.
            
        Returns:
            List of feature names
        """
        if include_groups is None:
            # default recommended groups for Gate model
            include_groups = [
                'cluster_info', 'temporal_base', 'temporal_cyclic', 'temporal_flags',
                'activity_counts', 'activity_flags', 'weather_base', 'weather_cloud',
                'weather_wind', 'weather_solar', 'activity_lags', 'flag_lags',
                'activity_rolling', 'flag_rolling', 'activity_same_hour', 
                'flag_same_hour', 'lag_flags'
            ]
        
        features = []
        for group in include_groups:
            if group in self.feature_groups:
                features.extend(self.feature_groups[group])
            else:
                print(f"Warning: Feature group '{group}' not found. Available groups: {list(self.feature_groups.keys())}")
        
        # remove ts_start as it's not a feature for training
        features = [f for f in features if f != 'ts_start']
        
        self.selected_features = features
        return features
    
    def filter_dataframe(self, df, target_col: str = None, include_groups: List[str] = None):
        """
        Filter dataframe to keep only selected features.
        
        Args:
            df: Input dataframe (polars or pandas)
            target_col: Name of target column to keep (if any)
            include_groups: List of feature groups to include
            
        Returns:
            Filtered dataframe
        """
        features = self.get_gate_features(include_groups)
        
        # always keep ts_start for temporal reference and target if specified
        keep_cols = ['ts_start'] + features
        if target_col and target_col not in keep_cols:
            keep_cols.append(target_col)
        
        # filter based on available columns
        if isinstance(df, pl.DataFrame):
            available_cols = df.columns
            keep_cols = [col for col in keep_cols if col in available_cols]
            return df.select(keep_cols)
        elif isinstance(df, pd.DataFrame):
            available_cols = df.columns.tolist()
            keep_cols = [col for col in keep_cols if col in available_cols]
            return df[keep_cols]
        else:
            raise ValueError("DataFrame must be polars or pandas DataFrame")
    
    def get_excluded_features(self, all_features: List[str]) -> List[str]:
        """
        Get list of features that are excluded from the Gate model.
        
        Args:
            all_features: List of all available features
            
        Returns:
            List of excluded feature names
        """
        if self.selected_features is None:
            self.get_gate_features()
        
        return [f for f in all_features if f not in self.selected_features and f != 'ts_start']
    
    def print_feature_summary(self, all_features: List[str] = None):
        """
        Print a summary of selected and excluded features.
        """
        if self.selected_features is None:
            self.get_gate_features()
        
        print(f"Selected features for Gate model: {len(self.selected_features)}")
        print("Feature groups included:")
        for group, features in self.feature_groups.items():
            included_count = len([f for f in features if f in self.selected_features])
            if included_count > 0:
                print(f"  - {group}: {included_count}/{len(features)} features")
        
        if all_features:
            excluded = self.get_excluded_features(all_features)
            print(f"\nExcluded features: {len(excluded)}")
            
            # group excluded features by pattern
            excluded_patterns = {}
            for feature in excluded:
                if any(pattern in feature for pattern in ['_male', '_female', '_other']):
                    excluded_patterns.setdefault('demographic', []).append(feature)
                elif any(pattern in feature for pattern in ['_young', '_adult', '_senior']):
                    excluded_patterns.setdefault('age_groups', []).append(feature)
                elif 'duration' in feature:
                    excluded_patterns.setdefault('duration', []).append(feature)
                elif any(pattern in feature for pattern in ['_comfort_', '_feel_diff_']):
                    excluded_patterns.setdefault('comfort_derived', []).append(feature)
                elif any(pattern in feature for pattern in ['dep_internal_temperature', 'dep_external_temperature', 'arr_internal_temperature', 'arr_external_temperature']):
                    excluded_patterns.setdefault('trip_specific_weather', []).append(feature)
                elif 'model_iconic' in feature:
                    excluded_patterns.setdefault('bike_model', []).append(feature)
                else:
                    excluded_patterns.setdefault('other', []).append(feature)
            
            for pattern, features in excluded_patterns.items():
                print(f"  - {pattern}: {len(features)} features")


def create_gate_target(df, target_col: str = 'has_activity') -> pl.DataFrame:
    """
    Create binary target variable for Gate model.
    Target = 1 if there's any departure or arrival activity, 0 otherwise.
    
    Args:
        df: Input dataframe
        target_col: Name for the target column
        
    Returns:
        Dataframe with added target column
    """
    if isinstance(df, pl.DataFrame):
        return df.with_columns([
            ((pl.col('dep_internal_count') > 0) | 
             (pl.col('dep_external_count') > 0) | 
             (pl.col('arr_internal_count') > 0) | 
             (pl.col('arr_external_count') > 0)).cast(pl.Int32).alias(target_col)
        ])
    elif isinstance(df, pd.DataFrame):
        df = df.copy()
        df[target_col] = ((df['dep_internal_count'] > 0) | 
                         (df['dep_external_count'] > 0) | 
                         (df['arr_internal_count'] > 0) | 
                         (df['arr_external_count'] > 0)).astype(int)
        return df
    else:
        raise ValueError("DataFrame must be polars or pandas DataFrame") 