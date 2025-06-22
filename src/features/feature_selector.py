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
        # features excluded to prevent data leakage with Gate target
        # these features describe activity IN THE CURRENT PERIOD that we're trying to predict
        self.excluded_for_data_leakage = [
            # direct activity counts for current period (target leakage)
            'dep_internal_count', 'dep_external_count',
            'arr_internal_count', 'arr_external_count',
            
            # activity flags for current period (target leakage)
            'has_dep_internal', 'has_dep_external',
            'has_arr_internal', 'has_arr_external',
            
            # demographic features of current period activity (derived from target)
            'dep_internal_male', 'dep_internal_female', 'dep_internal_other',
            'dep_external_male', 'dep_external_female', 'dep_external_other',
            'arr_internal_male', 'arr_internal_female', 'arr_internal_other',
            'arr_external_male', 'arr_external_female', 'arr_external_other',
            
            # age group features of current period activity (derived from target)
            'dep_internal_young', 'dep_internal_adult', 'dep_internal_senior',
            'dep_external_young', 'dep_external_adult', 'dep_external_senior',
            'arr_internal_young', 'arr_internal_adult', 'arr_internal_senior',
            'arr_external_young', 'arr_external_adult', 'arr_external_senior',
            'dep_internal_age_available', 'dep_external_age_available',
            'arr_internal_age_available', 'arr_external_age_available',
            
            # bike model features of current period activity (derived from target)
            'dep_internal_model_iconic', 'dep_external_model_iconic',
            'arr_internal_model_iconic', 'arr_external_model_iconic',
            
            # trip duration features of current period activity (derived from target)
            'dep_internal_duration_mean', 'dep_internal_duration_std',
            'dep_external_duration_mean', 'dep_external_duration_std',
            
            # weather features aggregated BY current period activity (indirect leakage)
            # these are problematic because they're computed ONLY for active trips in current period
            'dep_internal_temperature_2m_mean', 'dep_internal_relative_humidity_2m_mean',
            'dep_internal_dew_point_2m_mean', 'dep_internal_apparent_temperature_mean',
            'dep_internal_precipitation_mean', 'dep_internal_weather_code_mean',
            'dep_internal_pressure_msl_mean', 'dep_internal_cloud_cover_mean',
            'dep_internal_cloud_cover_low_mean', 'dep_internal_cloud_cover_mid_mean',
            'dep_internal_cloud_cover_high_mean', 'dep_internal_vapour_pressure_deficit_mean',
            'dep_internal_wind_speed_10m_mean', 'dep_internal_wind_direction_10m_mean',
            'dep_internal_wind_gusts_10m_mean', 'dep_internal_soil_temperature_0_to_7cm_mean',
            'dep_internal_soil_moisture_0_to_7cm_mean', 'dep_internal_sunshine_duration_mean',
            'dep_internal_direct_radiation_mean', 'dep_internal_temp_feel_diff_mean',
            'dep_internal_weather_comfort_mean',
            
            'dep_external_temperature_2m_mean', 'dep_external_relative_humidity_2m_mean',
            'dep_external_dew_point_2m_mean', 'dep_external_apparent_temperature_mean',
            'dep_external_precipitation_mean', 'dep_external_weather_code_mean',
            'dep_external_pressure_msl_mean', 'dep_external_cloud_cover_mean',
            'dep_external_cloud_cover_low_mean', 'dep_external_cloud_cover_mid_mean',
            'dep_external_cloud_cover_high_mean', 'dep_external_vapour_pressure_deficit_mean',
            'dep_external_wind_speed_10m_mean', 'dep_external_wind_direction_10m_mean',
            'dep_external_wind_gusts_10m_mean', 'dep_external_soil_temperature_0_to_7cm_mean',
            'dep_external_soil_moisture_0_to_7cm_mean', 'dep_external_sunshine_duration_mean',
            'dep_external_direct_radiation_mean', 'dep_external_temp_feel_diff_mean',
            'dep_external_weather_comfort_mean',
            
            'arr_internal_temperature_2m_mean', 'arr_internal_relative_humidity_2m_mean',
            'arr_internal_dew_point_2m_mean', 'arr_internal_apparent_temperature_mean',
            'arr_internal_precipitation_mean', 'arr_internal_weather_code_mean',
            'arr_internal_pressure_msl_mean', 'arr_internal_cloud_cover_mean',
            'arr_internal_cloud_cover_low_mean', 'arr_internal_cloud_cover_mid_mean',
            'arr_internal_cloud_cover_high_mean', 'arr_internal_vapour_pressure_deficit_mean',
            'arr_internal_wind_speed_10m_mean', 'arr_internal_wind_direction_10m_mean',
            'arr_internal_wind_gusts_10m_mean', 'arr_internal_soil_temperature_0_to_7cm_mean',
            'arr_internal_soil_moisture_0_to_7cm_mean', 'arr_internal_sunshine_duration_mean',
            'arr_internal_direct_radiation_mean', 'arr_internal_temp_feel_diff_mean',
            'arr_internal_weather_comfort_mean',
            
            'arr_external_temperature_2m_mean', 'arr_external_relative_humidity_2m_mean',
            'arr_external_dew_point_2m_mean', 'arr_external_apparent_temperature_mean',
            'arr_external_precipitation_mean', 'arr_external_weather_code_mean',
            'arr_external_pressure_msl_mean', 'arr_external_cloud_cover_mean',
            'arr_external_cloud_cover_low_mean', 'arr_external_cloud_cover_mid_mean',
            'arr_external_cloud_cover_high_mean', 'arr_external_vapour_pressure_deficit_mean',
            'arr_external_wind_speed_10m_mean', 'arr_external_wind_direction_10m_mean',
            'arr_external_wind_gusts_10m_mean', 'arr_external_soil_temperature_0_to_7cm_mean',
            'arr_external_soil_moisture_0_to_7cm_mean', 'arr_external_sunshine_duration_mean',
            'arr_external_direct_radiation_mean', 'arr_external_temp_feel_diff_mean',
            'arr_external_weather_comfort_mean',
            
            # weather boolean flag counts aggregated BY current period activity (indirect leakage)
            'dep_internal_is_day_count', 'dep_internal_temp_comfortable_count',
            'dep_internal_temp_very_hot_count', 'dep_internal_temp_cold_count',
            'dep_internal_is_raining_count', 'dep_internal_heavy_rain_count',
            'dep_internal_light_rain_count', 'dep_internal_strong_wind_count',
            'dep_internal_moderate_wind_count', 'dep_internal_weather_matched_count',
            
            'dep_external_is_day_count', 'dep_external_temp_comfortable_count',
            'dep_external_temp_very_hot_count', 'dep_external_temp_cold_count',
            'dep_external_is_raining_count', 'dep_external_heavy_rain_count',
            'dep_external_light_rain_count', 'dep_external_strong_wind_count',
            'dep_external_moderate_wind_count', 'dep_external_weather_matched_count',
            
            'arr_internal_is_day_count', 'arr_internal_temp_comfortable_count',
            'arr_internal_temp_very_hot_count', 'arr_internal_temp_cold_count',
            'arr_internal_is_raining_count', 'arr_internal_heavy_rain_count',
            'arr_internal_light_rain_count', 'arr_internal_strong_wind_count',
            'arr_internal_moderate_wind_count', 'arr_internal_weather_matched_count',
            
            'arr_external_is_day_count', 'arr_external_temp_comfortable_count',
            'arr_external_temp_very_hot_count', 'arr_external_temp_cold_count',
            'arr_external_is_raining_count', 'arr_external_heavy_rain_count',
            'arr_external_light_rain_count', 'arr_external_strong_wind_count',
            'arr_external_moderate_wind_count', 'arr_external_weather_matched_count'
        ]
    
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
                'wind_direction_10m_mean', 'wind_gusts_10m_mean'
            ],
            
            'weather_soil': [
                'soil_temperature_0_to_7cm_mean', 'soil_moisture_0_to_7cm_mean'
            ],
            
            'weather_solar': [
                'sunshine_duration_mean', 'direct_radiation_mean'
            ],
            
            'weather_derived': [
                'temp_feel_diff_mean', 'weather_comfort_mean'
            ],
            
            'weather_flags': [
                'is_day', 'temp_comfortable', 'temp_very_hot', 'temp_cold',
                'is_raining', 'heavy_rain', 'light_rain', 'strong_wind',
                'moderate_wind', 'weather_matched', 'weather_data_available'
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
            # default recommended groups for Gate model (excluding activity_counts and activity_flags to prevent data leakage)
            include_groups = [
                'cluster_info', 'temporal_base', 'temporal_cyclic', 'temporal_flags',
                'weather_base', 'weather_cloud', 'weather_wind', 'weather_soil', 'weather_solar', 
                'weather_derived', 'weather_flags', 'activity_lags', 'flag_lags', 'activity_rolling', 'flag_rolling', 
                'activity_same_hour', 'flag_same_hour', 'lag_flags'
            ]
        
        # check for excluded groups that would cause data leakage
        forbidden_groups = ['activity_counts', 'activity_flags']
        for forbidden in forbidden_groups:
            if forbidden in include_groups:
                print(f"⚠️  WARNING: Feature group '{forbidden}' is excluded to prevent data leakage.")
                print(f"   These features are used to construct the Gate target and would cause perfect but unrealistic performance.")
                print(f"   Removing '{forbidden}' from feature groups.")
                include_groups = [g for g in include_groups if g != forbidden]
        
        features = []
        for group in include_groups:
            if group in self.feature_groups:
                features.extend(self.feature_groups[group])
            else:
                print(f"Warning: Feature group '{group}' not found. Available groups: {list(self.feature_groups.keys())}")
        
        # remove ts_start as it's not a feature for training
        features = [f for f in features if f != 'ts_start']
        
        # double-check: remove any data leakage features that might have been included
        original_count = len(features)
        features = [f for f in features if f not in self.excluded_for_data_leakage]
        if len(features) < original_count:
            removed_count = original_count - len(features)
            print(f"🔒 Removed {removed_count} features to prevent data leakage")
        
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
        
        print(f"\n🎯 GATE MODEL FEATURE SELECTION SUMMARY")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"\nFeature groups included:")
        for group, features in self.feature_groups.items():
            included_count = len([f for f in features if f in self.selected_features])
            if included_count > 0:
                print(f"  ✅ {group}: {included_count}/{len(features)} features")
        
        # show data leakage exclusions with clear explanation
        if self.excluded_for_data_leakage:
            print(f"\n🚫 DATA LEAKAGE PREVENTION")
            print(f"Excluded {len(self.excluded_for_data_leakage)} features that would cause data leakage:")
            print(f"   (These describe activity in the CURRENT period we're trying to predict)")
            
            # group leakage features by type for clarity
            leakage_groups = {
                'Activity Counts': [f for f in self.excluded_for_data_leakage if any(pattern in f for pattern in ['_count']) and not any(pattern in f for pattern in ['_lag_', '_rolling_', '_same_hour_'])],
                'Activity Flags': [f for f in self.excluded_for_data_leakage if f.startswith('has_') and not any(pattern in f for pattern in ['_lag_', '_rolling_', '_same_hour_'])],
                'Demographics': [f for f in self.excluded_for_data_leakage if any(pattern in f for pattern in ['_male', '_female', '_other', '_young', '_adult', '_senior', '_age_available'])],
                'Trip Features': [f for f in self.excluded_for_data_leakage if any(pattern in f for pattern in ['_duration_', '_model_'])],
                'Trip-Weather Aggregations': [f for f in self.excluded_for_data_leakage if any(pattern in f for pattern in ['dep_internal_', 'dep_external_', 'arr_internal_', 'arr_external_']) and any(pattern in f for pattern in ['_temperature_', '_humidity_', '_precipitation_', '_weather_', '_wind_', '_cloud_', '_radiation_', '_is_day_', '_temp_', '_rain_'])]
            }
            
            for group_name, group_features in leakage_groups.items():
                if group_features:
                    print(f"   • {group_name}: {len(group_features)} features")
        
        if all_features:
            excluded = self.get_excluded_features(all_features)
            # exclude the data leakage features from the count since they're shown separately
            excluded_non_leakage = [f for f in excluded if f not in self.excluded_for_data_leakage]
            
            if excluded_non_leakage:
                print(f"\n📋 OTHER EXCLUDED FEATURES: {len(excluded_non_leakage)}")
                print(f"   (Not included in current feature groups but available)")
                
                # group excluded features by pattern
                excluded_patterns = {}
                for feature in excluded_non_leakage:
                    if any(pattern in feature for pattern in ['_male', '_female', '_other']):
                        excluded_patterns.setdefault('demographic_features', []).append(feature)
                    elif any(pattern in feature for pattern in ['_young', '_adult', '_senior']):
                        excluded_patterns.setdefault('age_group_features', []).append(feature)
                    elif 'duration' in feature and any(pattern in feature for pattern in ['_lag_', '_rolling_', '_same_hour_']):
                        excluded_patterns.setdefault('duration_lag_features', []).append(feature)
                    elif any(pattern in feature for pattern in ['_comfort_', '_feel_diff_']):
                        excluded_patterns.setdefault('comfort_derived_features', []).append(feature)
                    elif 'model_iconic' in feature and any(pattern in feature for pattern in ['_lag_', '_rolling_', '_same_hour_']):
                        excluded_patterns.setdefault('bike_model_lag_features', []).append(feature)
                    else:
                        excluded_patterns.setdefault('other_features', []).append(feature)
                
                for pattern, features in excluded_patterns.items():
                    if features:
                        print(f"   • {pattern.replace('_', ' ').title()}: {len(features)} features")
            
        print(f"\n💡 SUMMARY:")
        print(f"   ✅ Safe features selected: {len(self.selected_features)}")
        print(f"   🚫 Data leakage features excluded: {len(self.excluded_for_data_leakage)}")
        if all_features:
            total_available = len([f for f in all_features if f not in ['ts_start']])
            print(f"   📊 Total available features: {total_available}")
            print(f"   📈 Usage rate: {len(self.selected_features)/total_available*100:.1f}%")
    
    def get_data_leakage_features(self) -> List[str]:
        """
        Get list of features excluded due to data leakage concerns.
        
        Returns:
            List of feature names that would cause data leakage
        """
        return self.excluded_for_data_leakage.copy()


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