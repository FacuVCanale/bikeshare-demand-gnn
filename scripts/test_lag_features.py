#!/usr/bin/env python3
"""
Test lag features to ensure no data leakage.

This script tests the lag feature generation on a small sample
to verify that future information is not leaked into past predictions.
"""

import polars as pl
import sys
import os
from datetime import datetime, timedelta

# add src to python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.cluster_features import ClusterFeatureGenerator


def create_test_data():
    """Create simple test data to verify lag features."""
    
    # create a simple time series for 2 clusters over 10 time periods
    test_data = []
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    
    for cluster_id in [0, 1]:
        for i in range(10):
            timestamp = base_time + timedelta(minutes=30 * i)
            
            # create predictable pattern: cluster 0 has increasing values, cluster 1 has decreasing
            if cluster_id == 0:
                dep_count = i + 1
                arr_count = i + 2
            else:
                dep_count = 10 - i
                arr_count = 11 - i
                
            test_data.append([
                cluster_id,
                timestamp,
                dep_count,
                0,  # dep_external_count
                arr_count,
                0,  # arr_external_count
                1 if dep_count > 0 else 0,  # has_dep_internal
                0,  # has_dep_external
                1 if arr_count > 0 else 0,  # has_arr_internal
                0,  # has_arr_external
                dep_count,  # dep_internal_male (same as count for simplicity)
                0,  # dep_internal_female
                dep_count // 2,  # dep_internal_young
                dep_count // 2,  # dep_internal_adult
                0,  # dep_internal_senior
                1,  # dep_internal_model_iconic
                30.0,  # dep_internal_duration_mean
                arr_count,  # arr_internal_male
                0,  # arr_internal_female
                arr_count // 2,  # arr_internal_young
                arr_count // 2,  # arr_internal_adult
                0,  # arr_internal_senior
                1,  # arr_internal_model_iconic
            ])
    
    schema = {
        'cluster_id': pl.Int32,
        'ts_start': pl.Datetime("us"),
        'dep_internal_count': pl.Int32,
        'dep_external_count': pl.Int32,
        'arr_internal_count': pl.Int32,
        'arr_external_count': pl.Int32,
        'has_dep_internal': pl.Int32,
        'has_dep_external': pl.Int32,
        'has_arr_internal': pl.Int32,
        'has_arr_external': pl.Int32,
        'dep_internal_male': pl.Int32,
        'dep_internal_female': pl.Int32,
        'dep_internal_young': pl.Int32,
        'dep_internal_adult': pl.Int32,
        'dep_internal_senior': pl.Int32,
        'dep_internal_model_iconic': pl.Int32,
        'dep_internal_duration_mean': pl.Float64,
        'arr_internal_male': pl.Int32,
        'arr_internal_female': pl.Int32,
        'arr_internal_young': pl.Int32,
        'arr_internal_adult': pl.Int32,
        'arr_internal_senior': pl.Int32,
        'arr_internal_model_iconic': pl.Int32,
    }
    
    return pl.DataFrame(test_data, schema=schema, orient="row")


def test_lag_features():
    """Test lag features for correctness and data leakage."""
    
    print("Creating test data...")
    test_df = create_test_data()
    
    print("Original data:")
    print(test_df.select(['cluster_id', 'ts_start', 'dep_internal_count', 'arr_internal_count']))
    
    print("\nGenerating lag features...")
    feature_gen = ClusterFeatureGenerator(dt_minutes=30, log_level="INFO")
    df_with_lags = feature_gen.add_lag_features(test_df)
    
    print("\nData with lag features:")
    lag_cols = ['cluster_id', 'ts_start', 'dep_internal_count', 'dep_internal_count_lag_1', 
                'dep_internal_count_lag_2', 'arr_internal_count', 'arr_internal_count_lag_1']
    
    if all(col in df_with_lags.columns for col in lag_cols):
        print(df_with_lags.select(lag_cols))
    else:
        print("Some lag columns not found. Available columns:")
        print([col for col in df_with_lags.columns if 'lag' in col][:10])
    
    # test for data leakage: lag_1 should never equal current value except at boundaries
    print("\nTesting for data leakage...")
    
    # for cluster 0, dep_internal_count should be increasing: 1, 2, 3, 4, ...
    # so lag_1 should be: null, 1, 2, 3, ...
    cluster_0_data = df_with_lags.filter(pl.col('cluster_id') == 0).sort('ts_start')
    
    if 'dep_internal_count_lag_1' in cluster_0_data.columns:
        print("Cluster 0 - Current vs Lag_1:")
        print(cluster_0_data.select(['ts_start', 'dep_internal_count', 'dep_internal_count_lag_1']))
        
        # check that lag_1 is always less than current (for increasing pattern)
        leakage_check = cluster_0_data.filter(
            (pl.col('dep_internal_count_lag_1') > 0) &  # ignore nulls/zeros
            (pl.col('dep_internal_count_lag_1') >= pl.col('dep_internal_count'))
        )
        
        if leakage_check.height > 0:
            print("❌ POTENTIAL DATA LEAKAGE DETECTED!")
            print(leakage_check)
        else:
            print("✅ No data leakage detected in lag_1")
    
    # test rolling features
    if 'dep_internal_count_rolling_mean_4' in df_with_lags.columns:
        print("\nRolling mean features (should exclude current period):")
        print(cluster_0_data.select(['ts_start', 'dep_internal_count', 'dep_internal_count_rolling_mean_4']))
    
    # test cyclical features
    if 'dep_internal_count_same_hour_yesterday' in df_with_lags.columns:
        print("\nSame hour yesterday features:")
        print(cluster_0_data.select(['ts_start', 'dep_internal_count', 'dep_internal_count_same_hour_yesterday']))
    
    print(f"\nTotal features: {df_with_lags.width}")
    lag_feature_count = len([col for col in df_with_lags.columns if 'lag' in col or 'rolling' in col or 'yesterday' in col or 'last_week' in col])
    print(f"Lag features: {lag_feature_count}")


if __name__ == "__main__":
    test_lag_features() 