#!/usr/bin/env python3
"""
Debug script to check feature removal logic
"""

import polars as pl
import json

def identify_features_to_remove(columns):
    features_to_remove = []
    
    # Remove ALL internal features
    internal_features = [c for c in columns if 'internal' in c]
    features_to_remove.extend(internal_features)
    
    # Current arrivals
    arr_current = [c for c in columns if c.startswith('arr_') and not any(
        x in c for x in ['lag_', 'rolling_', 'same_hour_']
    )]
    features_to_remove.extend(arr_current)
    
    # Current departures
    dep_current = [c for c in columns if c.startswith('dep_') and not any(
        x in c for x in ['lag_', 'rolling_', 'same_hour_']
    )]
    features_to_remove.extend(dep_current)
    
    # Current activity flags
    has_current = [c for c in columns if c.startswith('has_') and not any(
        x in c for x in ['lag_', 'rolling_', 'same_hour_']
    )]
    features_to_remove.extend(has_current)
    
    # Meta-features
    meta_flags = ['has_short_term_lags', 'has_daily_lags', 'has_weekly_lags', 'weather_data_available']
    meta_flags_present = [c for c in meta_flags if c in columns]
    features_to_remove.extend(meta_flags_present)
    
    return list(set(features_to_remove))

def identify_target_features(columns):
    targets = ['arr_external_count']
    demographic_targets = [
        'arr_external_male',
        'arr_external_female', 
        'arr_external_young',
        'arr_external_adult',
        'arr_external_senior'
    ]
    available_targets = [t for t in targets + demographic_targets if t in columns]
    return available_targets

# Load data
print("Loading data...")
df = pl.read_parquet('data/clustered/train_cluster_features.parquet')
print(f"Original shape: {df.shape}")

# Check targets
all_target_cols = identify_target_features(df.columns)
target_cols = ['arr_external_count']
print(f"Target columns we want: {target_cols}")
print(f"Available target columns: {all_target_cols}")

# Check feature removal
features_to_remove = identify_features_to_remove(df.columns)
features_to_remove_filtered = [f for f in features_to_remove if f not in target_cols]

print(f"\nOriginal features: {len(df.columns)}")
print(f"Features to remove: {len(features_to_remove)}")
print(f"Features to remove (filtered): {len(features_to_remove_filtered)}")

# Check if our target is being removed
print(f"\nIs 'arr_external_count' in original data? {'arr_external_count' in df.columns}")
print(f"Is 'arr_external_count' being removed? {'arr_external_count' in features_to_remove}")
print(f"Is 'arr_external_count' being removed (filtered)? {'arr_external_count' in features_to_remove_filtered}")

# Show what external arrival columns exist
external_arr_cols = [c for c in df.columns if c.startswith('arr_external')]
print(f"\nAll external arrival columns in data: {external_arr_cols}")

# Sample data check
print(f"\nSample arr_external_count values:")
print(df.select(['cluster_id', 'ts_start', 'arr_external_count']).head(10)) 