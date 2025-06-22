#!/usr/bin/env python3
"""
Debug script to check feature removal logic
"""

import polars as pl
import json
from pathlib import Path

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

def debug_sequence_creation():
    """Debug sequence creation issues"""
    print("=== Debugging Sequence Creation ===\n")
    
    # Load data
    parquet_path = Path('data/clustered/train_cluster_features.parquet')
    df = pl.read_parquet(str(parquet_path))
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.select('ts_start').min().item()} to {df.select('ts_start').max().item()}")
    
    # Check cluster distribution
    cluster_counts = df.group_by('cluster_id').len().sort('len', descending=True)
    print(f"\nCluster data distribution:")
    print(f"Number of clusters: {len(cluster_counts)}")
    print(f"Samples per cluster - Min: {cluster_counts.select('len').min().item()}, Max: {cluster_counts.select('len').max().item()}")
    print(f"Samples per cluster - Mean: {cluster_counts.select('len').mean().item():.1f}")
    
    # Show top clusters
    print("\nTop 10 clusters by sample count:")
    print(cluster_counts.head(10))
    
    # Check sequence requirements
    sequence_length = 24
    prediction_horizon = 1
    min_required = sequence_length + prediction_horizon
    
    print(f"\nSequence requirements:")
    print(f"Sequence length: {sequence_length}")
    print(f"Prediction horizon: {prediction_horizon}")
    print(f"Minimum samples required per cluster: {min_required}")
    
    # Count how many clusters meet the requirement
    valid_clusters = cluster_counts.filter(pl.col('len') >= min_required)
    print(f"\nClusters with enough data: {len(valid_clusters)} / {len(cluster_counts)}")
    
    if len(valid_clusters) == 0:
        print("❌ NO CLUSTERS have enough data for sequence creation!")
        print("This explains why the sequences list is empty.")
        
        # Suggest solutions
        print("\n🔧 Possible solutions:")
        print(f"1. Reduce sequence_length from {sequence_length} to a smaller value")
        print(f"2. Check if data is properly time-sorted")
        print(f"3. Verify date range and time intervals")
        
        # Check time intervals
        sample_cluster = df.filter(pl.col('cluster_id') == 0).sort('ts_start')
        if len(sample_cluster) > 1:
            time_diffs = sample_cluster.with_columns([
                (pl.col('ts_start').diff().dt.total_seconds() / 60).alias('minutes_diff')
            ]).select('minutes_diff').drop_nulls()
            
            print(f"\nTime intervals for cluster 0:")
            print(f"  Most common interval: {time_diffs.mode().item()} minutes")
            print(f"  Expected: 30 minutes")
    else:
        print(f"✅ {len(valid_clusters)} clusters have enough data")
        
        # Calculate potential sequences
        total_sequences = 0
        for row in valid_clusters.iter_rows():
            cluster_id, n_samples = row
            n_sequences = n_samples - min_required + 1
            total_sequences += n_sequences
            
        print(f"Total potential sequences: {total_sequences:,}")
    
    # Check target column presence
    print(f"\n=== Target Column Check ===")
    target_cols = ['arr_external_count']
    
    for target in target_cols:
        if target in df.columns:
            print(f"✅ {target} is present")
            # Check values
            stats = df.select(target).describe()
            print(f"   Stats: {stats}")
            
            # Check non-zero values
            non_zero = df.filter(pl.col(target) > 0).height
            print(f"   Non-zero samples: {non_zero:,} / {df.height:,} ({100*non_zero/df.height:.1f}%)")
        else:
            print(f"❌ {target} is missing")

if __name__ == "__main__":
    debug_sequence_creation() 