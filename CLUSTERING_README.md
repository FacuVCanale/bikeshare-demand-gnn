# EcoBici Clustering Pipeline

This document describes the clustering pipeline for generating cluster-level features from EcoBici trip data.

## Overview

The clustering pipeline:
1. **Loads and merges** all datasets (trips_with_weather, trips, users)
2. **Creates temporal splits** without data leakage (train/val/test)
3. **Fits K-means clustering** on training station coordinates only
4. **Generates cluster-level features** with internal/external trip distinction
5. **Handles ghost trips** and comprehensive user demographics
6. **Saves datasets** with rich feature sets for machine learning

## Key Concepts

### Ghost Trips
- **Internal trips**: Trips within the same cluster (origin and destination stations belong to same cluster)
- **External trips**: Trips across different clusters or involving unmapped stations
- **Ghost trips**: Same as internal trips - these would disappear if we naively aggregated all stations in a cluster

### Data Leakage Prevention
- Clustering is fitted **only on training data** stations
- Temporal splits ensure no future information leaks into training
- All statistics computed from training set only

### Feature Types Generated

#### Cluster Metadata
- `cluster_id`: Cluster identifier (0 to k-1)
- `cluster_centroid_lat/lon`: Geographic center of cluster
- `cluster_station_count`: Number of stations in cluster

#### Temporal Features
- `hour`, `day_of_week`, `month`, `year`: Basic time components
- `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`: Cyclical encodings
- `is_weekend`, `is_morning_rush`, `is_evening_rush`: Business logic
- `is_summer`, `is_winter`: Seasonal indicators

#### Trip Volume Features
- `dep_internal_count`, `dep_external_count`: Departure counts by type
- `arr_internal_count`, `arr_external_count`: Arrival counts by type
- `has_dep_internal/external`, `has_arr_internal/external`: Occurrence flags

#### User Demographics (by trip type)
- `dep/arr_internal/external_male/female/other`: Gender counts
- `dep/arr_internal/external_age_mean/std/min/max`: Age statistics
- `dep/arr_internal/external_age_available`: Age data availability count

#### Bike Model Features
- `dep/arr_internal/external_model_types`: Number of different bike models

#### Duration Features
- `dep_internal/external_duration_mean/std`: Trip duration statistics

#### Weather Features (if available)
- Temperature, humidity, precipitation, wind speed, etc.
- `weather_data_available`: Weather data availability flag

#### Quality Flags
- `user_matched`: User information successfully joined
- `age_available`: Age information available
- `weather_available`: Weather information available

## Usage

### Basic Usage

```bash
# Run with default parameters
python scripts/generate_cluster_dataset.py

# Specify custom parameters
python scripts/generate_cluster_dataset.py \
    --data_dir data \
    --output_dir data/clustered \
    --n_clusters 93 \
    --dt_minutes 30 \
    --train_end_date "2023-01-01" \
    --val_end_date "2023-07-01"
```

### Advanced Usage with Checkpointing

```bash
# Use checkpoints for resumable processing
python scripts/generate_cluster_dataset.py \
    --use_checkpoints \
    --log_level DEBUG

# Clear checkpoints and start fresh
python scripts/generate_cluster_dataset.py \
    --clear_checkpoints \
    --use_checkpoints
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `"data"` | Root data directory |
| `--output_dir` | `"data/clustered"` | Output directory for results |
| `--n_clusters` | `93` | Number of K-means clusters |
| `--dt_minutes` | `30` | Time aggregation interval (minutes) |
| `--train_end_date` | `"2023-01-01"` | End date for training set |
| `--val_end_date` | `"2023-07-01"` | End date for validation set |
| `--random_state` | `42` | Random seed for reproducibility |
| `--log_level` | `"INFO"` | Logging verbosity |
| `--use_checkpoints` | `False` | Enable checkpoint system |
| `--clear_checkpoints` | `False` | Clear existing checkpoints |

## Expected Data Structure

The script expects the following data files:

```
data/
├── processed/
│   └── trips_with_weather.parquet    # Main dataset with weather
├── raw/combined/
│   ├── trips.parquet                 # Complete trips dataset
│   └── users.parquet                 # User information
```

### Required Columns

#### trips_with_weather.parquet
- `id_recorrido`: Trip ID
- `fecha_origen_recorrido`: Origin timestamp
- `fecha_destino_recorrido`: Destination timestamp (optional)
- `id_estacion_origen/destino`: Station IDs
- `lat_estacion_origen/destino`: Station coordinates
- `long_estacion_origen/destino`: Station coordinates
- `id_usuario`: User ID
- `modelo_bicicleta`: Bike model
- `genero`: Gender
- `duracion_recorrido`: Trip duration
- Weather columns: `temperature_2m`, `precipitation`, etc.

#### users.parquet
- `id_usuario`: User ID (matching trips)
- `genero_usuario`: User gender
- `edad_usuario`: User age

## Output Structure

```
data/clustered/
├── train_cluster_features.parquet    # Training dataset
├── val_cluster_features.parquet      # Validation dataset  
├── test_cluster_features.parquet     # Test dataset
├── models/
│   └── kmeans_k93_model.pkl          # Fitted clustering model
├── checkpoints/                      # Intermediate results (if enabled)
├── dataset_metadata.json             # Generation metadata
└── generation_summary.txt            # Human-readable summary
```

## Dataset Structure

Each output dataset has the structure:
- **Rows**: One row per cluster per time interval
- **Columns**: ~50-80 features depending on data availability
- **Time Range**: Complete time range of input data
- **Memory**: Optimized with Polars for large datasets

Example row represents:
- Cluster ID 15
- Time interval 2022-03-15 14:30:00 to 2022-03-15 15:00:00
- All aggregated trip statistics for that cluster and time period

## Performance Considerations

### Memory Usage
- Uses Polars for memory-efficient processing
- Checkpointing system prevents memory overflow
- Processes data in temporal chunks if needed

### Processing Time
- ~10-30 minutes for full dataset (depending on hardware)
- Checkpointing allows resuming interrupted runs
- Parallel processing where possible

### Disk Space
- Input: ~1-2GB total
- Output: ~300-500MB per split
- Checkpoints: ~2-3GB temporary (if enabled)

## Troubleshooting

### Common Issues

1. **Memory Error**
   ```bash
   # Use checkpointing to reduce memory usage
   python scripts/generate_cluster_dataset.py --use_checkpoints
   ```

2. **Missing Data Files**
   ```
   FileNotFoundError: data/processed/trips_with_weather.parquet
   ```
   - Ensure data files exist in expected locations
   - Check `--data_dir` parameter

3. **Date Format Issues**
   ```
   # Ensure dates are in YYYY-MM-DD format
   --train_end_date "2023-01-01"
   ```

4. **Clustering Convergence**
   - If K-means doesn't converge, try different `--random_state`
   - Reduce `--n_clusters` if stations are too sparse

### Debug Mode

```bash
# Run with verbose logging
python scripts/generate_cluster_dataset.py --log_level DEBUG

# Check logs
tail -f cluster_dataset_generation.log
```

## Architecture

### Module Structure

```
src/clustering/
├── __init__.py                   # Module exports
├── data_preparation.py           # Data loading and merging
├── station_clustering.py         # K-means clustering logic  
└── cluster_features.py          # Feature generation
```

### Key Classes

- **`DataPreparator`**: Handles data loading, merging, and temporal splitting
- **`StationClusterer`**: K-means clustering with coordinate standardization
- **`ClusterFeatureGenerator`**: Generates cluster-level aggregated features

## Validation

The pipeline includes several validation steps:

1. **Data Quality Checks**: Null counts, coordinate bounds, trip counts
2. **Temporal Validation**: Ensures no data leakage between splits
3. **Clustering Validation**: Checks cluster size distribution and coverage
4. **Feature Validation**: Validates feature completeness and statistics

## Extending the Pipeline

### Adding New Features

1. Extend `ClusterFeatureGenerator.aggregate_departures()` or `aggregate_arrivals()`
2. Add new aggregation expressions to the `.agg()` calls
3. Update documentation and tests

### Custom Clustering Algorithms

1. Inherit from `StationClusterer` base class
2. Implement `fit_clustering()` and `predict_cluster()` methods
3. Maintain same interface for downstream compatibility

### Different Time Aggregations

Simply change the `--dt_minutes` parameter:
```bash
# 15-minute intervals
python scripts/generate_cluster_dataset.py --dt_minutes 15

# 1-hour intervals  
python scripts/generate_cluster_dataset.py --dt_minutes 60
```

---

**Note**: This pipeline was designed for the EcoBici Buenos Aires dataset but can be adapted for other bike-sharing systems with similar data structures. 