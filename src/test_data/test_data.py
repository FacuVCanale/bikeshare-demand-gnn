"""
Test suite for EcoBici data processing pipeline.
Tests data quality, missing values, types, and logical consistency for each step.
"""

import os
import sys
import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pytest
import warnings

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_analysis import (
    filter_data_until_date,
    temporal_split_data,
    engineer_ecobici_features,
    pivot_features_for_global_model,
    step_01_convert_to_polars,
    step_02_create_departures_table,
    step_03_create_arrivals_table,
    step_04_create_shifted_arrivals,
    step_05_create_weather_table,
    step_06_create_station_metadata,
    step_07_build_skeleton,
    step_08_join_base_features,
    step_09_temporal_features,
    step_10_rolling_features,
    step_11_neighbor_features,
)


class DataQualityTester:
    """Comprehensive data quality testing for EcoBici pipeline."""
    
    def __init__(self, checkpoint_dir: str = "feature_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.test_results = {}
        
    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test results."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.test_results[test_name] = {"passed": passed, "message": message}
        print(f"{status}: {test_name}")
        if message:
            print(f"   Details: {message}")
    
    def print_summary(self):
        """Print test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["passed"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFailed tests:")
            for test_name, result in self.test_results.items():
                if not result["passed"]:
                    print(f"  - {test_name}: {result['message']}")
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def check_no_nulls(self, df: Union[pd.DataFrame, pl.DataFrame], 
                      columns: List[str], test_name: str) -> bool:
        """Check that specified columns have no null values."""
        try:
            if isinstance(df, pl.DataFrame):
                null_counts = df.select([pl.col(c).null_count().alias(c) for c in columns]).to_pandas().iloc[0]
            else:
                null_counts = df[columns].isnull().sum()
            
            has_nulls = null_counts.sum() > 0
            if has_nulls:
                null_cols = [col for col, count in null_counts.items() if count > 0]
                self.log_test_result(test_name, False, f"Null values found in: {null_cols}")
                return False
            else:
                self.log_test_result(test_name, True)
                return True
        except Exception as e:
            self.log_test_result(test_name, False, f"Error checking nulls: {str(e)}")
            return False
    
    def check_data_types(self, df: Union[pd.DataFrame, pl.DataFrame], 
                        expected_types: Dict[str, str], test_name: str) -> bool:
        """Check that columns have expected data types."""
        try:
            if isinstance(df, pl.DataFrame):
                actual_types = {col: str(df[col].dtype) for col in expected_types.keys() if col in df.columns}
            else:
                actual_types = {col: str(df[col].dtype) for col in expected_types.keys() if col in df.columns}
            
            type_mismatches = []
            for col, expected_type in expected_types.items():
                if col in actual_types:
                    actual_type = actual_types[col]
                    if expected_type not in actual_type:
                        type_mismatches.append(f"{col}: expected {expected_type}, got {actual_type}")
            
            if type_mismatches:
                self.log_test_result(test_name, False, f"Type mismatches: {type_mismatches}")
                return False
            else:
                self.log_test_result(test_name, True)
                return True
        except Exception as e:
            self.log_test_result(test_name, False, f"Error checking types: {str(e)}")
            return False
    
    def check_value_ranges(self, df: Union[pd.DataFrame, pl.DataFrame], 
                          range_checks: Dict[str, Tuple[float, float]], test_name: str) -> bool:
        """Check that numeric columns fall within expected ranges."""
        try:
            violations = []
            for col, (min_val, max_val) in range_checks.items():
                if col in df.columns:
                    if isinstance(df, pl.DataFrame):
                        col_min = df[col].min()
                        col_max = df[col].max()
                    else:
                        col_min = df[col].min()
                        col_max = df[col].max()
                    
                    if col_min < min_val or col_max > max_val:
                        violations.append(f"{col}: range [{col_min}, {col_max}] outside expected [{min_val}, {max_val}]")
            
            if violations:
                self.log_test_result(test_name, False, f"Range violations: {violations}")
                return False
            else:
                self.log_test_result(test_name, True)
                return True
        except Exception as e:
            self.log_test_result(test_name, False, f"Error checking ranges: {str(e)}")
            return False
    
    # ==================== STEP-SPECIFIC TESTS ====================
    
    def test_raw_data_quality(self, df_raw: pd.DataFrame) -> bool:
        """Test the quality of raw input data."""
        print("\n🔍 Testing Raw Data Quality...")
        
        # Check basic structure
        required_cols = ['fecha_origen_recorrido', 'id_estacion_origen', 'id_estacion_destino']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            self.log_test_result("raw_data_required_columns", False, f"Missing columns: {missing_cols}")
            return False
        else:
            self.log_test_result("raw_data_required_columns", True)
        
        # Check for completely empty dataframe
        if len(df_raw) == 0:
            self.log_test_result("raw_data_not_empty", False, "DataFrame is empty")
            return False
        else:
            self.log_test_result("raw_data_not_empty", True)
        
        # Check date column
        if df_raw['fecha_origen_recorrido'].isnull().all():
            self.log_test_result("raw_data_dates_not_null", False, "All dates are null")
            return False
        else:
            self.log_test_result("raw_data_dates_not_null", True)
        
        return True
    
    def test_step_01_polars_conversion(self, df_raw: pd.DataFrame, dt_minutes: int = 30) -> bool:
        """Test Step 1: Polars conversion and date normalization."""
        print("\n🔍 Testing Step 1: Polars Conversion...")
        
        try:
            df, bike_models = step_01_convert_to_polars(df_raw, dt_minutes, self.checkpoint_dir)
            
            # Check output types
            if not isinstance(df, pl.DataFrame):
                self.log_test_result("step01_output_type", False, f"Expected polars DataFrame, got {type(df)}")
                return False
            else:
                self.log_test_result("step01_output_type", True)
            
            # Check bike models extraction
            if not isinstance(bike_models, list) or len(bike_models) == 0:
                self.log_test_result("step01_bike_models", False, f"Invalid bike_models: {bike_models}")
                return False
            else:
                self.log_test_result("step01_bike_models", True)
            
            # Check key columns exist
            required_cols = ['ts_start', 'station_id', 'fecha_origen_recorrido']
            self.check_no_nulls(df, [col for col in required_cols if col in df.columns], "step01_no_nulls_key_cols")
            
            # Check datetime normalization
            if 'ts_start' in df.columns:
                ts_dtype = str(df['ts_start'].dtype)
                if 'datetime' not in ts_dtype.lower():
                    self.log_test_result("step01_datetime_type", False, f"ts_start type: {ts_dtype}")
                    return False
                else:
                    self.log_test_result("step01_datetime_type", True)
            
            return True
            
        except Exception as e:
            self.log_test_result("step01_execution", False, f"Step failed: {str(e)}")
            return False
    
    def test_step_02_departures_table(self) -> bool:
        """Test Step 2: Departures table creation."""
        print("\n🔍 Testing Step 2: Departures Table...")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, "step02_departures.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step02_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            df_dep = pl.read_parquet(checkpoint_path)
            
            # Check required columns
            required_cols = ['ts_start', 'station_id', 'dep_last_DT']
            missing_cols = [col for col in required_cols if col not in df_dep.columns]
            if missing_cols:
                self.log_test_result("step02_required_columns", False, f"Missing: {missing_cols}")
                return False
            else:
                self.log_test_result("step02_required_columns", True)
            
            # Check departure lags
            lag_cols = [col for col in df_dep.columns if col.startswith('dep_lag_')]
            if len(lag_cols) == 0:
                self.log_test_result("step02_lag_features", False, "No departure lag features found")
                return False
            else:
                self.log_test_result("step02_lag_features", True)
            
            # Check non-negative departures
            numeric_cols = ['dep_last_DT'] + lag_cols
            range_checks = {col: (0, float('inf')) for col in numeric_cols if col in df_dep.columns}
            self.check_value_ranges(df_dep, range_checks, "step02_non_negative_departures")
            
            # Check user profile columns
            profile_cols = [col for col in df_dep.columns if col.startswith('share_')]
            if len(profile_cols) > 0:
                # Shares should be between 0 and 1
                share_checks = {col: (0, 1) for col in profile_cols}
                self.check_value_ranges(df_dep, share_checks, "step02_share_ranges")
            
            return True
            
        except Exception as e:
            self.log_test_result("step02_execution", False, f"Step failed: {str(e)}")
            return False
    
    def test_step_03_arrivals_table(self) -> bool:
        """Test Step 3: Arrivals table creation."""
        print("\n🔍 Testing Step 3: Arrivals Table...")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, "step03_arrivals.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step03_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            df_arr = pl.read_parquet(checkpoint_path)
            
            # Check required columns
            required_cols = ['ts_start', 'station_id', 'arr_last_DT']
            self.check_no_nulls(df_arr, [col for col in required_cols if col in df_arr.columns], "step03_no_nulls_key_cols")
            
            # Check arrival lags
            lag_cols = [col for col in df_arr.columns if col.startswith('arr_lag_')]
            if len(lag_cols) == 0:
                self.log_test_result("step03_lag_features", False, "No arrival lag features found")
                return False
            else:
                self.log_test_result("step03_lag_features", True)
            
            # Check non-negative arrivals
            numeric_cols = ['arr_last_DT'] + lag_cols
            range_checks = {col: (0, float('inf')) for col in numeric_cols if col in df_arr.columns}
            self.check_value_ranges(df_arr, range_checks, "step03_non_negative_arrivals")
            
            return True
            
        except Exception as e:
            self.log_test_result("step03_execution", False, f"Step failed: {str(e)}")
            return False
    
    def test_step_04_shifted_arrivals(self) -> bool:
        """Test Step 4: Shifted arrivals (target variable)."""
        print("\n🔍 Testing Step 4: Shifted Arrivals...")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, "step04_shifted_arrivals.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step04_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            df_target = pl.read_parquet(checkpoint_path)
            
            # Check target column
            target_col = 'y_arrivals_next_DT'
            if target_col not in df_target.columns:
                self.log_test_result("step04_target_column", False, f"Target column {target_col} not found")
                return False
            else:
                self.log_test_result("step04_target_column", True)
            
            # Check non-negative target values
            range_checks = {target_col: (0, float('inf'))}
            self.check_value_ranges(df_target, range_checks, "step04_non_negative_target")
           
            # Check required ID columns
            required_cols = ['ts_start', 'station_id']
            self.check_no_nulls(df_target, required_cols, "step04_no_nulls_ids")
           
            return True
           
        except Exception as e:
            self.log_test_result("step04_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_05_weather_table(self) -> bool:
        """Test Step 5: Weather table creation."""
        print("\n🔍 Testing Step 5: Weather Table...")
       
        checkpoint_path = os.path.join(self.checkpoint_dir, "step05_weather.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step05_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
       
        try:
            df_weather = pl.read_parquet(checkpoint_path)
           
            # Check weather columns
            weather_cols = [col for col in df_weather.columns if col.startswith('weather_')]
            if len(weather_cols) == 0:
                self.log_test_result("step05_weather_columns", False, "No weather columns found")
                return False
            else:
                self.log_test_result("step05_weather_columns", True)
           
            # Check temperature ranges (should be reasonable)
            temp_cols = [col for col in weather_cols if 'temperature' in col]
            for col in temp_cols:
                if col in df_weather.columns:
                    range_checks = {col: (-50, 60)}  # Reasonable temperature range in Celsius
                    self.check_value_ranges(df_weather, range_checks, f"step05_temperature_range_{col}")
           
            # Check humidity ranges (0-100%)
            humidity_cols = [col for col in weather_cols if 'humidity' in col]
            for col in humidity_cols:
                if col in df_weather.columns:
                    range_checks = {col: (0, 100)}
                    self.check_value_ranges(df_weather, range_checks, f"step05_humidity_range_{col}")
           
            return True
           
        except Exception as e:
            self.log_test_result("step05_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_06_station_metadata(self) -> bool:
        """Test Step 6: Station metadata creation."""
        print("\n🔍 Testing Step 6: Station Metadata...")
       
        checkpoint_path = os.path.join(self.checkpoint_dir, "step06_stations.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step06_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
       
        try:
            df_stations = pl.read_parquet(checkpoint_path)
           
            # Check required columns
            required_cols = ['station_id']
            self.check_no_nulls(df_stations, required_cols, "step06_no_nulls_station_id")
           
            # Check for coordinates if they exist
            coord_cols = [col for col in df_stations.columns if any(word in col.lower() for word in ['lat', 'lon', 'x', 'y'])]
            if coord_cols:
                # Basic coordinate range checks
                for col in coord_cols:
                    if 'lat' in col.lower():
                        range_checks = {col: (-90, 90)}
                    elif 'lon' in col.lower():
                        range_checks = {col: (-180, 180)}
                    else:
                        continue
                    self.check_value_ranges(df_stations, range_checks, f"step06_coordinate_range_{col}")
           
            return True
           
        except Exception as e:
            self.log_test_result("step06_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_07_skeleton(self) -> bool:
        """Test Step 7: Skeleton creation."""
        print("\n🔍 Testing Step 7: Skeleton...")
       
        checkpoint_path = os.path.join(self.checkpoint_dir, "step07_skeleton.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step07_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
       
        try:
            df_skeleton = pl.read_parquet(checkpoint_path)
           
            # Check required columns
            required_cols = ['ts_start', 'station_id']
            self.check_no_nulls(df_skeleton, required_cols, "step07_no_nulls_required")
           
            # Check time series continuity
            if 'ts_start' in df_skeleton.columns:
                timestamps = df_skeleton['ts_start'].unique().sort()
                if len(timestamps) > 1:
                    # Check if timestamps are regularly spaced
                    time_diffs = timestamps[1:] - timestamps[:-1]
                    if isinstance(df_skeleton, pl.DataFrame):
                        unique_diffs = len(time_diffs.unique())
                    else:
                        unique_diffs = len(pd.Series(time_diffs).unique())
                   
                    if unique_diffs > 2:  # Allow some flexibility
                        self.log_test_result("step07_regular_timestamps", False, "Irregular timestamp spacing")
                    else:
                        self.log_test_result("step07_regular_timestamps", True)
           
            return True
           
        except Exception as e:
            self.log_test_result("step07_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_08_joined_features(self) -> bool:
        """Test Step 8: Joined base features."""
        print("\n🔍 Testing Step 8: Joined Features...")
       
        checkpoint_path = os.path.join(self.checkpoint_dir, "step08_joined_with_weather.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step08_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
       
        try:
            df_joined = pl.read_parquet(checkpoint_path)
           
            # Check that we have features from previous steps
            feature_groups = {
                'departures': [col for col in df_joined.columns if col.startswith('dep_')],
                'arrivals': [col for col in df_joined.columns if col.startswith('arr_')],
                'weather': [col for col in df_joined.columns if col.startswith('weather_')],
                'target': [col for col in df_joined.columns if 'y_arrivals' in col]
            }
           
            for group, cols in feature_groups.items():
                if len(cols) == 0:
                    self.log_test_result(f"step08_{group}_features", False, f"No {group} features found")
                else:
                    self.log_test_result(f"step08_{group}_features", True)
           
            # Check required ID columns
            required_cols = ['ts_start', 'station_id']
            self.check_no_nulls(df_joined, required_cols, "step08_no_nulls_ids")
           
            return True
           
        except Exception as e:
            self.log_test_result("step08_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_09_temporal_features(self) -> bool:
        """Test Step 9: Temporal features."""
        print("\n🔍 Testing Step 9: Temporal Features...")
       
        checkpoint_path = os.path.join(self.checkpoint_dir, "step09_temporal.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step09_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
       
        try:
            df_temporal = pl.read_parquet(checkpoint_path)
           
            # Check for temporal feature groups
            temporal_feature_groups = {
                'hour': [col for col in df_temporal.columns if 'hour' in col],
                'day_of_week': [col for col in df_temporal.columns if 'dow' in col],
                'month': [col for col in df_temporal.columns if 'month' in col],
                'weekend': [col for col in df_temporal.columns if 'weekend' in col]
            }
           
            for group, cols in temporal_feature_groups.items():
                if len(cols) == 0:
                    self.log_test_result(f"step09_{group}_features", False, f"No {group} features found")
                else:
                    self.log_test_result(f"step09_{group}_features", True)
           
            # Check sine/cosine encoding ranges
            trig_cols = [col for col in df_temporal.columns if any(x in col for x in ['sin_', 'cos_'])]
            for col in trig_cols:
                range_checks = {col: (-1, 1)}
                self.check_value_ranges(df_temporal, range_checks, f"step09_trig_range_{col}")
           
            return True
           
        except Exception as e:
            self.log_test_result("step09_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_10_rolling_features(self) -> bool:
        """Test Step 10: Rolling features."""
        print("\n🔍 Testing Step 10: Rolling Features...")
       
        checkpoint_path = os.path.join(self.checkpoint_dir, "step10_rolling.parquet")
        if not os.path.exists(checkpoint_path):
            self.log_test_result("step10_checkpoint_exists", False, f"Checkpoint not found: {checkpoint_path}")
            return False
       
        try:
            df_rolling = pl.read_parquet(checkpoint_path)
           
            # Check for rolling feature patterns
            rolling_patterns = ['_ma_', '_std_', '_ratio_']
            rolling_cols = [col for col in df_rolling.columns
                           if any(pattern in col for pattern in rolling_patterns)]
           
            if len(rolling_cols) == 0:
                self.log_test_result("step10_rolling_features", False, "No rolling features found")
                return False
            else:
                self.log_test_result("step10_rolling_features", True)
           
            # Check non-negative rolling features (where applicable)
            non_negative_patterns = ['_ma_', '_std_']
            non_negative_cols = [col for col in rolling_cols
                               if any(pattern in col for pattern in non_negative_patterns)]
           
            for col in non_negative_cols:
                range_checks = {col: (0, float('inf'))}
                self.check_value_ranges(df_rolling, range_checks, f"step10_non_negative_{col}")
           
            return True
           
        except Exception as e:
            self.log_test_result("step10_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_step_11_neighbor_features(self) -> bool:
        """Test Step 11: Neighbor features."""
        print("\n🔍 Testing Step 11: Neighbor Features...")
       
        final_result_path = os.path.join(self.checkpoint_dir, "df_feat_final_result.parquet")
        if not os.path.exists(final_result_path):
            self.log_test_result("step11_final_result_exists", False, f"Final result not found: {final_result_path}")
            return False
       
        try:
            df_final = pl.read_parquet(final_result_path)
           
            # Check for neighbor features
            neighbor_cols = [col for col in df_final.columns if 'near_' in col]
            if len(neighbor_cols) == 0:
                self.log_test_result("step11_neighbor_features", False, "No neighbor features found")
                return False
            else:
                self.log_test_result("step11_neighbor_features", True)
           
            # Check final feature count
            feature_cols = [col for col in df_final.columns
                           if col not in ['ts_start', 'station_id']]
           
            expected_min_features = 20  # Should have at least 20 features
            if len(feature_cols) < expected_min_features:
                self.log_test_result("step11_feature_count", False,
                                   f"Only {len(feature_cols)} features, expected at least {expected_min_features}")
                return False
            else:
                self.log_test_result("step11_feature_count", True)
           
            return True
           
        except Exception as e:
            self.log_test_result("step11_execution", False, f"Step failed: {str(e)}")
            return False
   
    def test_pivot_features(self) -> bool:
        """Test pivoting to wide format."""
        print("\n🔍 Testing Pivot Features...")
       
        x_wide_path = os.path.join(self.checkpoint_dir, "X_wide.parquet")
        y_wide_path = os.path.join(self.checkpoint_dir, "y_wide.parquet")
       
        # Check if pivot files exist
        if not os.path.exists(x_wide_path):
            self.log_test_result("pivot_x_wide_exists", False, f"X_wide not found: {x_wide_path}")
            return False
        else:
            self.log_test_result("pivot_x_wide_exists", True)
       
        if not os.path.exists(y_wide_path):
            self.log_test_result("pivot_y_wide_exists", False, f"y_wide not found: {y_wide_path}")
            return False
        else:
            self.log_test_result("pivot_y_wide_exists", True)
       
        try:
            X_wide = pl.read_parquet(x_wide_path)
            y_wide = pl.read_parquet(y_wide_path)
           
            # Check dimensions match
            if X_wide.shape[0] != y_wide.shape[0]:
                self.log_test_result("pivot_dimensions_match", False,
                                   f"X_wide rows: {X_wide.shape[0]}, y_wide rows: {y_wide.shape[0]}")
                return False
            else:
                self.log_test_result("pivot_dimensions_match", True)
           
            # Check time alignment
            if 'ts_start' in X_wide.columns and 'ts_start' in y_wide.columns:
                x_times = set(X_wide['ts_start'].to_list())
                y_times = set(y_wide['ts_start'].to_list())
               
                if x_times != y_times:
                    self.log_test_result("pivot_time_alignment", False, "Timestamps don't match between X and y")
                    return False
                else:
                    self.log_test_result("pivot_time_alignment", True)
           
            return True
           
        except Exception as e:
            self.log_test_result("pivot_execution", False, f"Pivot test failed: {str(e)}")
            return False
   
    # ==================== MAIN TEST RUNNER ====================
   
    def run_full_pipeline_tests(self, df_raw: pd.DataFrame,
                               dt_minutes: int = 30, n_neighbors: int = 5) -> bool:
        """Run all tests for the complete pipeline."""
        print("🧪 RUNNING FULL PIPELINE TESTS")
        print("="*60)
       
        # Test raw data
        self.test_raw_data_quality(df_raw)
       
        # Test individual steps
        self.test_step_01_polars_conversion(df_raw, dt_minutes)
        self.test_step_02_departures_table()
        self.test_step_03_arrivals_table()
        self.test_step_04_shifted_arrivals()
        self.test_step_05_weather_table()
        self.test_step_06_station_metadata()
        self.test_step_07_skeleton()
        self.test_step_08_joined_features()
        self.test_step_09_temporal_features()
        self.test_step_10_rolling_features()
        self.test_step_11_neighbor_features()
       
        # Test pivot functionality
        self.test_pivot_features()
       
        # Print summary
        self.print_summary()
       
        # Return overall success
        return all(result["passed"] for result in self.test_results.values())


# ==================== CONVENIENCE FUNCTIONS ====================

def test_pipeline_from_checkpoint(checkpoint_dir: str = "feature_checkpoints") -> bool:
    """Test pipeline using existing checkpoints."""
    tester = DataQualityTester(checkpoint_dir)
   
    print("🧪 TESTING PIPELINE FROM CHECKPOINTS")
    print("="*60)
   
    # Test each step's checkpoint
    tester.test_step_02_departures_table()
    tester.test_step_03_arrivals_table()
    tester.test_step_04_shifted_arrivals()
    tester.test_step_05_weather_table()
    tester.test_step_06_station_metadata()
    tester.test_step_07_skeleton()
    tester.test_step_08_joined_features()
    tester.test_step_09_temporal_features()
    tester.test_step_10_rolling_features()
    tester.test_step_11_neighbor_features()
    tester.test_pivot_features()
   
    tester.print_summary()
    return all(result["passed"] for result in tester.test_results.values())


def test_pipeline_with_data(df_raw: pd.DataFrame,
                           checkpoint_dir: str = "feature_checkpoints",
                           dt_minutes: int = 30, n_neighbors: int = 5) -> bool:
    """Test pipeline with raw data."""
    tester = DataQualityTester(checkpoint_dir)
    return tester.run_full_pipeline_tests(df_raw, dt_minutes, n_neighbors)


def quick_data_check(df: Union[pd.DataFrame, pl.DataFrame], name: str = "DataFrame") -> None:
    """Quick data quality check for any DataFrame."""
    print(f"\n🔍 Quick Check: {name}")
    print("-" * 40)
   
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
   
    # Missing values
    if isinstance(df, pl.DataFrame):
        null_counts = df.select([pl.col(c).null_count().alias(c) for c in df.columns]).to_pandas().iloc[0]
    else:
        null_counts = df.isnull().sum()
   
    missing_cols = [col for col, count in null_counts.items() if count > 0]
    if missing_cols:
        print(f"⚠️  Columns with missing values: {len(missing_cols)}")
        for col in missing_cols[:5]:  # Show first 5
            print(f"   - {col}: {null_counts[col]} missing")
        if len(missing_cols) > 5:
            print(f"   ... and {len(missing_cols) - 5} more")
    else:
        print("✅ No missing values")
   
    # Data types
    if isinstance(df, pl.DataFrame):
        dtypes = {col: str(df[col].dtype) for col in df.columns}
    else:
        dtypes = df.dtypes.to_dict()
   
    type_summary = {}
    for dtype in dtypes.values():
        dtype_str = str(dtype)
        type_summary[dtype_str] = type_summary.get(dtype_str, 0) + 1
   
    print(f"Data types: {type_summary}")


if __name__ == "__main__":
    # Example usage
    print("Data Quality Tester for EcoBici Pipeline")
    print("Use test_pipeline_from_checkpoint() to test existing results")
    print("Use test_pipeline_with_data(df_raw) to test with new data")