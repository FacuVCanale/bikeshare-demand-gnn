import polars as pl
import numpy as np
from pathlib import Path

# read a sample of the clustered data
print('hola')
data_path = Path("data/clustered/train_cluster_features.parquet")
df = pl.read_parquet(data_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns[:20])  # show first 20 columns
print(f"\n... and {len(df.columns) - 20} more columns")

print("\nFirst few rows of key columns:")
key_cols = ['cluster_id', 'ts_start', 'arr_internal_count', 'arr_external_count', 
            'dep_internal_count', 'dep_external_count', 'cluster_station_count']
print(df.select(key_cols).head(10))

print("\nUnique clusters:", df['cluster_id'].n_unique())
print("Date range:", df['ts_start'].min(), "to", df['ts_start'].max())

# check temporal sampling
print("\nTemporal sampling:")
sample_cluster = df.filter(pl.col('cluster_id') == 0).sort('ts_start')
time_diffs = sample_cluster['ts_start'].diff()
print("Time differences between consecutive records:")
print(time_diffs.value_counts().head())

# check if all clusters have data at each timestamp
print("\nRecords per timestamp (sample):")
time_counts = df.group_by('ts_start').agg(pl.col('cluster_id').count().alias('count'))
print(time_counts.head())

print("\nTotal arrivals distribution:")
df_with_total = df.with_columns(
    (pl.col('arr_internal_count') + pl.col('arr_external_count')).alias('total_arrivals')
)
print(df_with_total['total_arrivals'].describe()) 