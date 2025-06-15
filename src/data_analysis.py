from __future__ import annotations
import pandas as pd
from typing import Tuple, Dict, Union
import numpy as np
import polars as pl
from datetime import timedelta
import math
from sklearn.neighbors import NearestNeighbors
import os
import holidays
import gc
import psutil
from tqdm import tqdm

# Import testing functionality
try:
    from test_data.test_data import DataQualityTester, quick_data_check
except ImportError:
    # Fallback if test_data module is not available
    print("⚠️ Warning: Testing module not available. Pipeline will run without validation.")
    DataQualityTester = None
    quick_data_check = None

# =============================================================================
# FUNCIÓN DE SPLIT DE DATOS TEMPORAL
# =============================================================================

def filter_data_until_date(users_df: pd.DataFrame, trips_df: pd.DataFrame, 
                          max_date: str = "2024-08-31",
                          user_date_col: str = 'fecha_alta',
                          trip_date_col: str = 'fecha_origen_recorrido',
                          verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    OPTIMIZED version to filter data until a specific maximum date.

    Optimizations:
    - Uses vectorized operations for date comparisons
    - Avoids redundant datetime conversions
    - More efficient boolean indexing
    - Batch processing for memory efficiency
    - Robust date type handling for both string and datetime columns
    
    Args:
        users_df: DataFrame of users
        trips_df: DataFrame of trips  
        max_date: Maximum date to include (YYYY-MM-DD format)
        user_date_col: Date column in users_df
        trip_date_col: Date column in trips_df
        verbose: Whether to show information
       
    Returns:
        Tuple with (users_filtered, trips_filtered)
    """
    if verbose:
        print(f"FILTERING DATA UNTIL {max_date}")
        print(f"   Input: Users={len(users_df):,}, Trips={len(trips_df):,}")
   
    max_datetime = pd.to_datetime(max_date)
   
    # Optimize users filtering
    if verbose:
        print(f"   Filtering users...")
   
    # ensure date column is properly converted to datetime
    users_date_series = users_df[user_date_col]
    if not pd.api.types.is_datetime64_any_dtype(users_date_series):
        users_date_series = pd.to_datetime(users_date_series, errors='coerce')
   
    # Vectorized filtering - much faster than multiple conditions
    users_mask = (users_date_series.notna()) & (users_date_series <= max_datetime)
    users_filtered = users_df[users_mask].copy()
   
    # Optimize trips filtering
    if verbose:
        print(f"   Filtering trips...")
   
    # ensure date column is properly converted to datetime
    trips_date_series = trips_df[trip_date_col]
    if not pd.api.types.is_datetime64_any_dtype(trips_date_series):
        trips_date_series = pd.to_datetime(trips_date_series, errors='coerce')
   
    # Vectorized filtering for large dataset
    trips_mask = (trips_date_series.notna()) & (trips_date_series <= max_datetime)
    trips_filtered = trips_df[trips_mask].copy()
   
    if verbose:
        users_removed = len(users_df) - len(users_filtered)
        trips_removed = len(trips_df) - len(trips_filtered)
        print(f"   Users: {len(users_df):,} → {len(users_filtered):,} (-{users_removed:,})")
        print(f"   Trips: {len(trips_df):,} → {len(trips_filtered):,} (-{trips_removed:,})")
   
    return users_filtered, trips_filtered


def temporal_split_data(users_df: pd.DataFrame, trips_df: pd.DataFrame,
                       train_end_date: str, val_end_date: str, test_end_date: str,
                       user_date_col: str = 'fecha_alta',
                       trip_date_col: str = 'fecha_origen_recorrido',
                       verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    OPTIMIZED version to split users and trips data by temporal ranges.
   
    Major optimizations:
    - Uses vectorized operations for all date comparisons
    - Batch processing for memory efficiency  
    - Efficient boolean indexing
    - Reduced copying operations
    - Smart memory management for large datasets
    - Robust date type handling for both string and datetime columns
   
    Args:
        users_df: DataFrame of users (already preprocessed)
        trips_df: DataFrame of trips (already preprocessed)
        train_end_date: End date for training (YYYY-MM-DD format)
        val_end_date: End date for validation (YYYY-MM-DD format)
        test_end_date: End date for test (YYYY-MM-DD format)
        user_date_col: Name of date column in users_df
        trip_date_col: Name of date column in trips_df
        verbose: Whether to show split information
       
    Returns:
        Dict with the split DataFrames
    """
   
    if verbose:
        print("OPTIMIZED TEMPORAL SPLIT")
        print(f"   Train ≤ {train_end_date} | Val: {train_end_date} - {val_end_date} | Test: {val_end_date} - {test_end_date}")
        print(f"   Input: Users={len(users_df):,}, Trips={len(trips_df):,}")
   
    # Convert dates once - more efficient
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    test_end = pd.to_datetime(test_end_date)
   
    # Get date columns as series and ensure proper datetime conversion
    user_dates = users_df[user_date_col]
    if not pd.api.types.is_datetime64_any_dtype(user_dates):
        user_dates = pd.to_datetime(user_dates, errors='coerce')
   
    trip_dates = trips_df[trip_date_col]
    if not pd.api.types.is_datetime64_any_dtype(trip_dates):
        trip_dates = pd.to_datetime(trip_dates, errors='coerce')
   
    # Vectorized splitting for users - much faster than individual filters
    if verbose:
        print("   Splitting users...")
   
    users_train_mask = user_dates <= train_end
    users_val_mask = (user_dates > train_end) & (user_dates <= val_end)
    users_test_mask = (user_dates > val_end) & (user_dates <= test_end)
   
    users_train = users_df[users_train_mask].copy()
    users_val = users_df[users_val_mask].copy()
    users_test = users_df[users_test_mask].copy()
   
    # Vectorized splitting for trips - optimized for large datasets
    if verbose:
        print("   Splitting trips...")
   
    trips_train_mask = trip_dates <= train_end
    trips_val_mask = (trip_dates > train_end) & (trip_dates <= val_end)
    trips_test_mask = (trip_dates > val_end) & (trip_dates <= test_end)
   
    trips_train = trips_df[trips_train_mask].copy()
    trips_val = trips_df[trips_val_mask].copy()
    trips_test = trips_df[trips_test_mask].copy()
   
    if verbose:
        print(f"   Users: Train={len(users_train):,}, Val={len(users_val):,}, Test={len(users_test):,}")
        print(f"   Trips: Train={len(trips_train):,}, Val={len(trips_val):,}, Test={len(trips_test):,}")
   
    return {
        'users_train': users_train,
        'users_val': users_val,
        'users_test': users_test,
        'trips_train': trips_train,
        'trips_val': trips_val,
        'trips_test': trips_test
    }


def _haversine_np(lon1, lat1, lon2, lat2):
    """
    Haversine vectorizado (km) – usado como métrica en NearestNeighbors.
    """
    # convertir a radianes
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _build_skeleton(stations_df: pl.DataFrame, start_ts: pl.Series,
                    end_ts: pl.Series, freq: str) -> pl.DataFrame:
    """
    Devuelve un DataFrame con todas las combinaciones (station_id, ts_start)
    entre start y end inclusive al paso indicado.
    """
    ts_range = pl.datetime_range(
        start=start_ts,
        end=end_ts,
        interval=freq,
        eager=True,
        time_unit="us"
    ).alias("ts_start")
    skeleton = stations_df.select(["station_id"]).join(
        ts_range.to_frame(),
        how="cross"
    )
    return skeleton


def _log_memory_usage(step_name: str):
    """Log current memory usage for debugging kernel crashes."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  -> Memory usage after {step_name}: {memory_mb:.1f} MB")
        
        # Also check available system memory
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024 / 1024 / 1024
        print(f"  -> Available system memory: {available_gb:.1f} GB")
        
        # Warning if memory usage is high
        if memory_mb > 4000:  # More than 4GB
            print(f"  ⚠️  High memory usage detected! Consider reducing chunk size.")
        
        if available_gb < 2:  # Less than 2GB available
            print(f"  ⚠️  Low system memory! Consider freeing up memory.")
            
    except Exception:
        pass  # Silently fail if psutil is not available


def _validate_step_with_action(tester: object, step_test_func, step_name: str,
                              abort_on_failure: bool = True, retry_func=None) -> bool:
    """
    Validate a pipeline step and take action based on results.
   
    Args:
        tester: DataQualityTester instance
        step_test_func: Function to test the step (bound method)
        step_name: Name of the step for logging
        abort_on_failure: Whether to abort pipeline on failure
        retry_func: Optional function to retry the step
   
    Returns:
        bool: True if validation passed or was skipped, False if failed and should abort
    """
    if tester is None:
        print(f"  🔍 Skipping validation for {step_name} (tester not available)")
        return True
   
    print(f"  🔍 Validating {step_name}...")
   
    try:
        passed = step_test_func()
       
        if passed:
            print(f"  ✅ {step_name} validation PASSED")
            return True
        else:
            print(f"  ❌ {step_name} validation FAILED")
           
            # Try retry if available
            if retry_func is not None:
                print(f"  🔄 Attempting to retry {step_name}...")
                try:
                    retry_func()
                    # Test again after retry
                    passed_after_retry = step_test_func()
                    if passed_after_retry:
                        print(f"  ✅ {step_name} validation PASSED after retry")
                        return True
                    else:
                        print(f"  ❌ {step_name} validation still FAILED after retry")
                except Exception as e:
                    print(f"  ⚠️ Retry failed for {step_name}: {str(e)}")
           
            # Decide what to do on failure
            if abort_on_failure:
                print(f"  🛑 Aborting pipeline due to {step_name} validation failure")
                return False
            else:
                print(f"  ⚠️ Continuing pipeline despite {step_name} validation failure")
                return True
               
    except Exception as e:
        print(f"  ⚠️ Validation error for {step_name}: {str(e)}")
        if abort_on_failure:
            print(f"  🛑 Aborting pipeline due to validation error")
            return False
        else:
            print(f"  ⚠️ Continuing pipeline despite validation error")
            return True


def clear_feature_checkpoints(checkpoint_dir: str = "feature_checkpoints"):
    """Clear all feature engineering checkpoints to force recreation."""
    import glob
    import os
   
    if os.path.exists(checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.parquet"))
        if checkpoint_files:
            print(f"Clearing {len(checkpoint_files)} checkpoint files...")
            for file in checkpoint_files:
                try:
                    os.remove(file)
                    print(f"  -> Removed: {os.path.basename(file)}")
                except Exception as e:
                    print(f"  -> Error removing {file}: {e}")
            print("✅ Checkpoints cleared!")
        else:
            print("No checkpoint files found.")
    else:
        print("Checkpoint directory does not exist.")


def _create_temporal_features(df_feat: pl.DataFrame, checkpoint_dir: str, temporal_path: str, holiday_dates: list) -> pl.DataFrame:
    """Create temporal features with memory-efficient operations.
   
    Changes:
      • Processes data in time windows to reduce memory pressure
      • Uses smaller data types (Int8, Float32) to save memory
      • Uses streaming operations for all heavy computations
      • Saves intermediate results to disk
      • Uses aggressive compression
    """
    print("  -> Creating temporal features with memory optimization...")
    
    # Check if checkpoint already exists
    df_cached = _load_checkpoint_if_exists(temporal_path, "temporal features")
    if df_cached is not None:
        return df_cached
   
    # Get time range for windowing
    min_ts = df_feat["ts_start"].min()
    max_ts = df_feat["ts_start"].max()
    time_window = timedelta(days=30)  # Process one month at a time

    # Feriados de Argentina - handle safely
    try:
        if not holiday_dates:  # Only compute if not provided
            min_year = df_feat["ts_start"].dt.year().min()
            max_year = df_feat["ts_start"].dt.year().max()
            ar_holidays = holidays.AR(years=range(min_year, max_year + 1))
            holiday_dates = [pd.to_datetime(date).date() for date in ar_holidays.keys()]
    except Exception as e:
        print(f"  -> Warning: Could not load holidays: {e}. Using empty holidays list.")
        holiday_dates = []
   
    # Process in time windows
    temp_chunk_dir = os.path.join(checkpoint_dir, "temp_temporal_chunks")
    os.makedirs(temp_chunk_dir, exist_ok=True)
   
    current_ts = min_ts
    chunk_paths = []
   
    while current_ts < max_ts:
        window_end = min(current_ts + time_window, max_ts)
        chunk_path = os.path.join(temp_chunk_dir, f"temporal_{current_ts.strftime('%Y%m%d')}.parquet")
        chunk_paths.append(chunk_path)
       
        print(f"  -> Processing window: {current_ts} to {window_end}")
       
        # Get data for this time window
        window_df = df_feat.filter(
            (pl.col("ts_start") >= current_ts) &
            (pl.col("ts_start") < window_end)
        )
       
        if not window_df.is_empty():
            try:
                # Basic temporal features (all Int8 to save memory)
                window_df = window_df.with_columns([
                    pl.col("ts_start").dt.hour().alias("hour").cast(pl.Int8),
                    pl.col("ts_start").dt.weekday().alias("dow").cast(pl.Int8),
                    pl.col("ts_start").dt.month().alias("month").cast(pl.Int8),
                    pl.col("ts_start").dt.day().alias("day").cast(pl.Int8),
                    pl.col("ts_start").dt.date().is_in(holiday_dates).alias("is_holiday_ar").cast(pl.Int8),
                ])
               
                # Cyclic encodings (Float32 to save memory)
                window_df = window_df.with_columns([
                    (pl.col("hour") * 2 * math.pi / 24).sin().alias("sin_hour").cast(pl.Float32),
                    (pl.col("hour") * 2 * math.pi / 24).cos().alias("cos_hour").cast(pl.Float32),
                    (pl.col("dow") * 2 * math.pi / 7).sin().alias("sin_dow").cast(pl.Float32),
                    (pl.col("dow") * 2 * math.pi / 7).cos().alias("cos_dow").cast(pl.Float32),
                    (pl.col("month") * 2 * math.pi / 12).sin().alias("sin_month").cast(pl.Float32),
                    (pl.col("month") * 2 * math.pi / 12).cos().alias("cos_month").cast(pl.Float32),
                ])
               
                # First, create independent binary flags that don't depend on others
                window_df = window_df.with_columns([
                    # In Polars, weekday() returns 0-6 where 0 is Monday and 6 is Sunday
                    pl.col("dow").is_in([5, 6]).alias("is_weekend").cast(pl.Int8),  # 5=Saturday, 6=Sunday
                    pl.col("day").is_in([1, 15]).alias("payday_flag").cast(pl.Int8),
                    pl.col("month").is_in([1, 7]).alias("vacation_season").cast(pl.Int8),
                ])

                # Peak commute depends on the newly-created is_weekend flag, so add it in a second pass
                window_df = window_df.with_columns(
                    ((pl.col("hour").is_in([7, 8, 9, 17, 18, 19])) & (pl.col("is_weekend") == 0))
                        .cast(pl.Int8)
                        .alias("peak_commute")
                )
               
                # Save window to disk immediately
                window_df.write_parquet(
                    chunk_path,
                    compression="zstd",
                    compression_level=3,
                    row_group_size=5_000
                )
            except Exception as e:
                print(f"  -> Error processing window {current_ts}: {e}")
                # Create empty chunk to maintain order
                window_df.select(["ts_start", "station_id"]).write_parquet(chunk_path)
       
        current_ts = window_end
        gc.collect()
        _log_memory_usage(f"processed window until {window_end}")
   
    print("  -> Combining all temporal windows...")
    # Combine all windows using streaming with better error handling
    try:
        # Filter out non-existent files before processing
        existing_chunk_paths = [f for f in chunk_paths if os.path.exists(f)]
        
        if not existing_chunk_paths:
            print("  -> Warning: No temporal chunk files found, creating basic temporal features...")
            # Fallback: create basic temporal features directly
            df_feat_final = df_feat.with_columns([
                pl.col("ts_start").dt.hour().alias("hour").cast(pl.Int8),
                pl.col("ts_start").dt.weekday().alias("dow").cast(pl.Int8),
                pl.col("ts_start").dt.month().alias("month").cast(pl.Int8),
                pl.col("ts_start").dt.day().alias("day").cast(pl.Int8),
                pl.lit(0).alias("is_holiday_ar").cast(pl.Int8),  # No holidays for fallback
            ]).with_columns([
                (pl.col("hour") * 2 * 3.14159 / 24).sin().alias("sin_hour").cast(pl.Float32),
                (pl.col("hour") * 2 * 3.14159 / 24).cos().alias("cos_hour").cast(pl.Float32),
                (pl.col("dow") * 2 * 3.14159 / 7).sin().alias("sin_dow").cast(pl.Float32),
                (pl.col("dow") * 2 * 3.14159 / 7).cos().alias("cos_dow").cast(pl.Float32),
                (pl.col("month") * 2 * 3.14159 / 12).sin().alias("sin_month").cast(pl.Float32),
                (pl.col("month") * 2 * 3.14159 / 12).cos().alias("cos_month").cast(pl.Float32),
                pl.col("dow").is_in([5, 6]).alias("is_weekend").cast(pl.Int8),
                pl.col("day").is_in([1, 15]).alias("payday_flag").cast(pl.Int8),
                pl.col("month").is_in([1, 7]).alias("vacation_season").cast(pl.Int8),
                ((pl.col("hour").is_in([7, 8, 9, 17, 18, 19])) & (pl.col("dow").is_in([5, 6]).not_()))
                    .cast(pl.Int8)
                    .alias("peak_commute")
            ])
        else:
            print(f"  -> Found {len(existing_chunk_paths)} temporal chunk files to combine...")
            df_feat_final = pl.concat(
                [pl.scan_parquet(f) for f in existing_chunk_paths],
                how="vertical"
            ).collect()
    except Exception as e:
        print(f"  -> Error combining temporal chunks: {e}")
        print("  -> Creating fallback temporal features...")
        # Fallback: create basic temporal features directly
        df_feat_final = df_feat.with_columns([
            pl.col("ts_start").dt.hour().alias("hour").cast(pl.Int8),
            pl.col("ts_start").dt.weekday().alias("dow").cast(pl.Int8),
            pl.col("ts_start").dt.month().alias("month").cast(pl.Int8),
            pl.col("ts_start").dt.day().alias("day").cast(pl.Int8),
            pl.lit(0).alias("is_holiday_ar").cast(pl.Int8),
        ]).with_columns([
            (pl.col("hour") * 2 * 3.14159 / 24).sin().alias("sin_hour").cast(pl.Float32),
            (pl.col("hour") * 2 * 3.14159 / 24).cos().alias("cos_hour").cast(pl.Float32),
            (pl.col("dow") * 2 * 3.14159 / 7).sin().alias("sin_dow").cast(pl.Float32),
            (pl.col("dow") * 2 * 3.14159 / 7).cos().alias("cos_dow").cast(pl.Float32),
            (pl.col("month") * 2 * 3.14159 / 12).sin().alias("sin_month").cast(pl.Float32),
            (pl.col("month") * 2 * 3.14159 / 12).cos().alias("cos_month").cast(pl.Float32),
            pl.col("dow").is_in([5, 6]).alias("is_weekend").cast(pl.Int8),
            pl.col("day").is_in([1, 15]).alias("payday_flag").cast(pl.Int8),
            pl.col("month").is_in([1, 7]).alias("vacation_season").cast(pl.Int8),
            ((pl.col("hour").is_in([7, 8, 9, 17, 18, 19])) & (pl.col("dow").is_in([5, 6]).not_()))
                .cast(pl.Int8)
                .alias("peak_commute")
        ])
   
    # Clean up temp files
    print("  -> Cleaning up temporary files...")
    for f in chunk_paths:
        try:
            os.remove(f)
        except:
            pass
    try:
        os.rmdir(temp_chunk_dir)
    except:
        pass
   
    # Final sort
    print("  -> Final sort by timestamp...")
    df_feat_final = df_feat_final.sort(["ts_start", "station_id"])
   
    # Save final result
    print(f"  -> Saving checkpoint: {temporal_path}")
    try:
        df_feat_final.write_parquet(
            temporal_path,
            compression="zstd",
            compression_level=3,
            row_group_size=5_000
        )
    except Exception as e:
        print(f"  -> Error with write: {e}")
   
    return df_feat_final


# =============================================================================
# PARAMETRIZED FEATURE ENGINEERING PIPELINE
# =============================================================================

def _load_checkpoint_if_exists(path: str, description: str) -> pl.DataFrame:
    """Load checkpoint if exists, return None otherwise."""
    if os.path.exists(path):
        print(f"  -> Loading {description} from checkpoint: {os.path.basename(path)}")
        return pl.read_parquet(path)
    return None


def _save_checkpoint(df: pl.DataFrame, path: str, description: str, use_streaming: bool = True):
    """Save checkpoint with error handling."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        if use_streaming:
            df.lazy().sink_parquet(path, compression="snappy", row_group_size=10000)
        else:
            df.write_parquet(path)
        print(f"  -> Saved {description} checkpoint: {os.path.basename(path)}")
    except Exception as e:
        if use_streaming:
            try:
                df.lazy().sink_parquet(path, compression="snappy", row_group_size=5000)
                print(f"  -> Saved {description} checkpoint with smaller chunks: {os.path.basename(path)}")
            except Exception as e2:
                print(f"  -> Failed to save {description} checkpoint: {e2}")
        else:
            print(f"  -> Failed to save {description} checkpoint: {e}")


def step_01_convert_to_polars(df_raw: pd.DataFrame, dt_minutes: int, checkpoint_dir: str) -> tuple[pl.DataFrame, list]:
    """Step 1: Convert to Polars and normalize dates."""
    print("[Step 1/11] Converting to Polars and normalizing dates...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step01_polars_conversion.parquet")
    df_cached = _load_checkpoint_if_exists(checkpoint_path, "Polars conversion")
   
    if df_cached is not None:
        # Also need to get bike models
        bike_models = []
        if "modelo_bicicleta" in df_cached.columns:
            bike_models = df_cached["modelo_bicicleta"].unique().to_list()
        return df_cached, bike_models
   
    step_tasks = ["Converting to Polars", "Casting station IDs", "Processing date columns", "Creating time features"]
    with tqdm(total=len(step_tasks), desc="Step 1", leave=False) as pbar:
        pbar.set_description("Converting to Polars")
        df = pl.from_pandas(df_raw)
        pbar.update(1)

        pbar.set_description("Casting station IDs")
        df = df.with_columns([
            pl.col("id_estacion_origen").cast(pl.Int64),
            pl.col("id_estacion_destino").cast(pl.Int64),
        ])
        pbar.update(1)

        pbar.set_description("Processing date columns")
        for col in ("fecha_origen_recorrido", "fecha_destino_recorrido"):
            if col in df.columns:
                if df[col].dtype == pl.String:
                    df = df.with_columns(pl.col(col).str.to_datetime(time_unit="us"))
                else:
                    df = df.with_columns(pl.col(col).cast(pl.Datetime("us")))
        pbar.update(1)

        pbar.set_description("Creating time features")
        freq = f"{dt_minutes}m"
        df = df.with_columns(
            ts_dep=pl.col("fecha_origen_recorrido").dt.truncate(freq),
            ts_arr=pl.col("fecha_destino_recorrido").dt.truncate(freq)
        )

        bike_models = []
        if "modelo_bicicleta" in df.columns:
            bike_models = df["modelo_bicicleta"].unique().to_list()
        pbar.update(1)

    _save_checkpoint(df, checkpoint_path, "Polars conversion")
    return df, bike_models


def step_02_create_departures_table(df: pl.DataFrame, bike_models: list, checkpoint_dir: str) -> pl.DataFrame:
    """Step 2: Create departures table with lags & user profiles."""
    print("[Step 2/11] Creating departures table with lags & user profiles...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step02_departures.parquet")
    dep_cached = _load_checkpoint_if_exists(checkpoint_path, "departures table")
   
    if dep_cached is not None:
        return dep_cached

    step_tasks = ["Building aggregations", "Computing departures", "Calculating proportions", "Adding lag features"]
    with tqdm(total=len(step_tasks), desc="Step 2", leave=False) as pbar:
        pbar.set_description("Building aggregations")
        aggs = [
            pl.len().alias("dep_last_DT"),
            pl.mean("duracion_recorrido").alias("trip_dur_mean_last_DT"),
            (pl.col("genero") == "MALE").sum().alias("male_cnt"),
            (pl.col("genero") == "FEMALE").sum().alias("female_cnt"),
            (pl.col("genero") == "OTHER").sum().alias("other_cnt"),
        ]
       
        if "modelo_bicicleta" in df.columns:
            for model in bike_models:
                if model:
                    aggs.append((pl.col("modelo_bicicleta") == model).sum().alias(f"model_{model}_cnt"))
        pbar.update(1)

        pbar.set_description("Computing departures")
        dep = (
            df.group_by(["id_estacion_origen", "ts_dep"])
              .agg(aggs)
              .rename({"id_estacion_origen": "station_id", "ts_dep": "ts_start"})
              .sort(["station_id", "ts_start"])
        )
        pbar.update(1)

        pbar.set_description("Calculating proportions")
        dep = dep.with_columns([
            pl.when(pl.col("dep_last_DT") > 0)
              .then(pl.col("male_cnt") / pl.col("dep_last_DT"))
              .otherwise(0.0).alias("share_male"),
            pl.when(pl.col("dep_last_DT") > 0)
              .then(pl.col("female_cnt") / pl.col("dep_last_DT"))
              .otherwise(0.0).alias("share_female"),
            pl.when(pl.col("dep_last_DT") > 0)
              .then(pl.col("other_cnt") / pl.col("dep_last_DT"))
              .otherwise(0.0).alias("share_other"),
        ]).drop(["male_cnt", "female_cnt", "other_cnt"])
        pbar.update(1)

        pbar.set_description("Adding lag features")
        dep = dep.with_columns([
            pl.col("dep_last_DT").shift(k).over("station_id").alias(f"dep_lag_{k}")
            for k in range(1, 7)
        ])
        pbar.update(1)

    _save_checkpoint(dep, checkpoint_path, "departures table")
    return dep


def step_03_create_arrivals_table(df: pl.DataFrame, checkpoint_dir: str) -> pl.DataFrame:
    """Step 3: Create arrivals table with lags."""
    print("[Step 3/11] Creating arrivals table with lags...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step03_arrivals.parquet")
    arr_cached = _load_checkpoint_if_exists(checkpoint_path, "arrivals table")
   
    if arr_cached is not None:
        return arr_cached

    step_tasks = ["Computing arrivals", "Adding lag features"]
    with tqdm(total=len(step_tasks), desc="Step 3", leave=False) as pbar:
        pbar.set_description("Computing arrivals")
        arr = (
            df.group_by(["id_estacion_destino", "ts_arr"])
              .agg(pl.len().alias("arr_last_DT"))
              .rename({"id_estacion_destino": "station_id", "ts_arr": "ts_start"})
              .sort(["station_id", "ts_start"])
        )
        pbar.update(1)
       
        pbar.set_description("Adding lag features")
        arr = arr.with_columns([
            pl.col("arr_last_DT").shift(k).over("station_id").alias(f"arr_lag_{k}")
            for k in range(1, 7)
        ])
        pbar.update(1)

    _save_checkpoint(arr, checkpoint_path, "arrivals table")
    return arr


def step_04_create_shifted_arrivals(dt_minutes: int, checkpoint_dir: str) -> pl.DataFrame:
    """Step 4: Create shifted arrivals table for target."""
    print("[Step 4/11] Creating shifted arrivals table for target...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step04_shifted_arrivals.parquet")
    arr_next_cached = _load_checkpoint_if_exists(checkpoint_path, "shifted arrivals")
   
    if arr_next_cached is not None:
        return arr_next_cached

    step_tasks = ["Loading arrivals base", "Creating shifted target"]
    with tqdm(total=len(step_tasks), desc="Step 4", leave=False) as pbar:
        pbar.set_description("Loading arrivals base")
        arr_path = os.path.join(checkpoint_dir, "step03_arrivals.parquet")
        arr_base = pl.read_parquet(arr_path).select(["station_id", "ts_start", "arr_last_DT"])
        pbar.update(1)
       
        pbar.set_description("Creating shifted target")
        arr_next = (
            arr_base.with_columns(
                (pl.col("ts_start") - timedelta(minutes=dt_minutes)).alias("ts_start")
            ).rename({"arr_last_DT": "y_arrivals_next_DT"})
        )
        pbar.update(1)

    _save_checkpoint(arr_next, checkpoint_path, "shifted arrivals")
    return arr_next


def step_05_create_weather_table(df: pl.DataFrame, checkpoint_dir: str) -> pl.DataFrame:
    """Step 5: Create enhanced weather table."""
    print("[Step 5/11] Creating enhanced weather table...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step05_weather.parquet")
    weather_cached = _load_checkpoint_if_exists(checkpoint_path, "weather table")
   
    if weather_cached is not None:
        return weather_cached

    weather_cols = [c for c in df.columns if c.startswith("weather_")]
   
    if not weather_cols:
        return None

    step_tasks = ["Extracting weather data", "Mapping weather codes", "Adding derived features", "Adding lag features"]
    with tqdm(total=len(step_tasks), desc="Step 5", leave=False) as pbar:
        pbar.set_description("Extracting weather data")
        weather = (
            df.select(["ts_dep"] + weather_cols)
              .unique(subset=["ts_dep"], keep="first")
              .rename({"ts_dep": "ts_start"})
              .sort("ts_start")
        )
        pbar.update(1)
       
        pbar.set_description("Mapping weather codes")
        def map_weather_code(code):
            weather_mapping = {
                0: "Clear", 1: "Clouds", 2: "Clouds", 3: "Clouds",
                45: "Fog", 48: "Fog",
                51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
                56: "Drizzle", 57: "Drizzle",
                61: "Rain", 63: "Rain", 65: "Rain",
                66: "Rain", 67: "Rain",
                71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
                80: "Rain", 81: "Rain", 82: "Rain",
                85: "Snow", 86: "Snow",
                95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"
            }
            return weather_mapping.get(code, "Other")
        pbar.update(1)
       
        pbar.set_description("Adding derived features")
        weather = weather.with_columns([
            (pl.col("weather_precipitation") > 0).cast(pl.Int8).alias("precip_flag"),
            (pl.col("weather_wind_direction_10m") * (math.pi / 180)).sin().alias("wind_dir_sin"),
            (pl.col("weather_wind_direction_10m") * (math.pi / 180)).cos().alias("wind_dir_cos"),
            pl.col("weather_weather_code").map_elements(map_weather_code, return_dtype=pl.Utf8).alias("weather_code_cat"),
        ])
        pbar.update(1)
       
        pbar.set_description("Adding lag features")
        weather = weather.sort("ts_start").with_columns([
            pl.col("weather_precipitation").shift(1).alias("precipitation_lag_1h"),
            pl.col("weather_rain").shift(1).alias("rain_lag_1h")
        ])
        pbar.update(1)

    _save_checkpoint(weather, checkpoint_path, "weather table")
    return weather


def step_06_create_station_metadata(df: pl.DataFrame, checkpoint_dir: str) -> pl.DataFrame:
    """Step 6: Create station metadata table."""
    print("[Step 6/11] Creating station metadata table...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step06_stations.parquet")
    stations_cached = _load_checkpoint_if_exists(checkpoint_path, "station metadata")
   
    if stations_cached is not None:
        return stations_cached

    with tqdm(total=1, desc="Step 6", leave=False) as pbar:
        pbar.set_description("Extracting station metadata")
        stations = (
            df.select([
                "id_estacion_origen",
                "nombre_estacion_origen",
                "long_estacion_origen",
                "lat_estacion_origen"
            ]).unique("id_estacion_origen", keep="first")
              .rename({
                  "id_estacion_origen": "station_id",
                  "nombre_estacion_origen": "station_name",
                  "long_estacion_origen": "lon",
                  "lat_estacion_origen": "lat"
              })
        )
        pbar.update(1)

    _save_checkpoint(stations, checkpoint_path, "station metadata")
    return stations


def step_07_build_skeleton(df: pl.DataFrame, stations: pl.DataFrame, dt_minutes: int, checkpoint_dir: str) -> pl.DataFrame:
    """Step 7: Build full station x time skeleton."""
    print("[Step 7/11] Building full station x time skeleton...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step07_skeleton.parquet")
    skeleton_cached = _load_checkpoint_if_exists(checkpoint_path, "skeleton")
   
    if skeleton_cached is not None:
        return skeleton_cached

    step_tasks = ["Computing time range", "Building skeleton"]
    with tqdm(total=len(step_tasks), desc="Step 7", leave=False) as pbar:
        pbar.set_description("Computing time range")
        min_ts = df["ts_dep"].min()
        max_ts = df["ts_dep"].max() + timedelta(minutes=dt_minutes)
        freq = f"{dt_minutes}m"
        pbar.update(1)
       
        pbar.set_description("Building skeleton")
        skeleton = _build_skeleton(stations, min_ts, max_ts, freq)
        pbar.update(1)

    _save_checkpoint(skeleton, checkpoint_path, "skeleton")
    return skeleton


def step_08_join_base_features(checkpoint_dir: str, has_weather: bool) -> pl.DataFrame:
    """Step 8: Join data onto skeleton."""
    print("[Step 8/11] Joining data onto skeleton...")
   
    # Check for final checkpoint first
    final_checkpoint = "step08_joined_with_weather.parquet" if has_weather else "step08_joined_no_weather.parquet"
    final_path = os.path.join(checkpoint_dir, final_checkpoint)
    df_feat_cached = _load_checkpoint_if_exists(final_path, "joined features")
   
    if df_feat_cached is not None:
        return df_feat_cached

    # Load required components
    skeleton = pl.read_parquet(os.path.join(checkpoint_dir, "step07_skeleton.parquet"))
    dep = pl.read_parquet(os.path.join(checkpoint_dir, "step02_departures.parquet"))
    arr = pl.read_parquet(os.path.join(checkpoint_dir, "step03_arrivals.parquet"))
    arr_next = pl.read_parquet(os.path.join(checkpoint_dir, "step04_shifted_arrivals.parquet"))

    step_tasks = ["Joining departures", "Joining arrivals", "Joining targets"]
    if has_weather:
        step_tasks.append("Joining weather")
       
    with tqdm(total=len(step_tasks), desc="Step 8", leave=False) as pbar:
        pbar.set_description("Joining departures")
        df_feat = skeleton.join(dep, on=["station_id", "ts_start"], how="left")
        pbar.update(1)
       
        pbar.set_description("Joining arrivals")
        df_feat = df_feat.join(arr, on=["station_id", "ts_start"], how="left")
        pbar.update(1)
       
        pbar.set_description("Joining targets")
        df_feat = df_feat.join(arr_next, on=["station_id", "ts_start"], how="left")
        pbar.update(1)
       
        if has_weather:
            pbar.set_description("Joining weather")
            weather = pl.read_parquet(os.path.join(checkpoint_dir, "step05_weather.parquet"))
            df_feat = df_feat.join(weather, on="ts_start", how="left")
            pbar.update(1)

    _save_checkpoint(df_feat, final_path, "joined features")
    return df_feat


def step_09_temporal_features(df_feat: pl.DataFrame, holiday_dates: list, checkpoint_dir: str) -> pl.DataFrame:
    """Step 9: Engineering temporal and calendar features."""
    print("[Step 9/11] Engineering temporal and calendar features...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step09_temporal.parquet")
    df_cached = _load_checkpoint_if_exists(checkpoint_path, "temporal features")
   
    if df_cached is not None:
        return df_cached

    df_feat = _create_temporal_features(df_feat, checkpoint_dir, checkpoint_path, holiday_dates)
    return df_feat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_safe_thread_pool():
    """Configure Polars to use a safe number of threads to prevent system overload.
    Uses at most 75% of available cores, with a maximum of 4 threads."""
    if os.getenv("POLARS_MAX_THREADS") is None:
        try:
            # Use at most 75% of cores, but never more than 4
            safe_cores = min(4, max(1, int(os.cpu_count() * 0.75)))
            pl.Config.set_global_pool_size(safe_cores)
            print(f"  -> Using {safe_cores} CPU cores for Polars operations")
        except AttributeError:  # older Polars (<0.19) fallback silently
            pass

_ensure_safe_thread_pool()

# ---------------------------------------------------------------------------
# Step‑10 feature engineering (rolling demand stats)
# ---------------------------------------------------------------------------

def step_10_rolling_features_fast(
    df_feat: pl.DataFrame | pl.LazyFrame,
    dt_minutes: int,
    checkpoint_dir: str,
    chunk_size: int | None = None,        # kept only for signature compatibility
) -> pl.DataFrame:
    """
    Native-Polars implementation – single pass, streaming, no Python loops.
    Writes a checkpoint parquet and returns the resulting frame.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "step10_rolling_trip.parquet")

    # ------------------------------------------------------------------ #
    # 0. Re-use checkpoint if present
    # ------------------------------------------------------------------ #
    if os.path.exists(checkpoint_path):
        return pl.read_parquet(checkpoint_path)

    # ------------------------------------------------------------------ #
    # 1. Ensure we are in Lazy mode (so Polars can stream)
    # ------------------------------------------------------------------ #
    lf = df_feat.lazy() if isinstance(df_feat, pl.DataFrame) else df_feat

    # ------------------------------------------------------------------ #
    # 2. Select only the columns we actually need for the rolling pass
    #    (cuts memory when df_feat has lots of weather columns)
    # ------------------------------------------------------------------ #
    lf_sel = lf.select([
        "station_id",
        "ts_start",
        "dep_last_DT",              # the series we roll on
        pl.exclude(["station_id", "ts_start", "dep_last_DT"]),  # keep rest for later join
    ])

    # ------------------------------------------------------------------ #
    # 3. Decide whether to use row-based or time-based rolling
    # ------------------------------------------------------------------ #
    window_rows = (24 * 60) // dt_minutes  # 48 for 30-min bins
    use_row_based = (
        lf.select(
            (pl.col("ts_start")
            .diff()
            .drop_nulls()
            .n_unique()            # count distinct spacings
            ).alias("n_deltas")
        )
        .collect()
        .item() == 1                  # exactly one unique spacing → regular grid
    )

    if use_row_based:
        # ---------- Regular grid -------------------------------------- #
        rolling_part = (
            lf_sel.sort(["station_id", "ts_start"])
                  .with_columns([
                      pl.col("dep_last_DT")
                        .rolling_mean(window_rows)
                        .over("station_id")
                        .cast(pl.Float32)
                        .alias("dep_ma_24h"),

                      pl.col("dep_last_DT")
                        .rolling_std(window_rows)
                        .over("station_id")
                        .cast(pl.Float32)
                        .alias("dep_std_24h"),
                  ])
        )
    else:
        # ---------- Irregular grid – use true 24-hour window ---------- #
        rolling_part = (
            lf_sel.sort(["station_id", "ts_start"])
                .rolling(                       #  ← nuevo
                    index_column="ts_start",
                    period="24h",
                    group_by="station_id",      #  ← antes `by`
                    closed="both",
                )
                .agg([
                    pl.mean("dep_last_DT").alias("dep_ma_24h"),
                    pl.std("dep_last_DT").alias("dep_std_24h"),
                ])
        )

    # ------------------------------------------------------------------ #
    # 4. Join the rolling stats back, compute the ratio, and stream to disk
    # ------------------------------------------------------------------ #
    lf_out = (
        lf.join(rolling_part, on=["station_id", "ts_start"], how="left")
          .with_columns(
              (pl.col("dep_last_DT") / (pl.col("dep_ma_24h") + 1.0))
                .cast(pl.Float32)
                .alias("dep_ratio_DT_24h")
          )
    )

    # Stream straight to parquet (no .collect() ⇒ constant RAM)
    lf_out.sink_parquet(
        checkpoint_path,
        compression="zstd",
        compression_level=3,
        row_group_size=5_000,
    )

    # If the caller really wants the frame in memory:
    return pl.read_parquet(checkpoint_path)

def _calculate_safe_neighbor_limit(n_stations: int, requested_neighbors: int) -> int:
    """Calculate a safe number of neighbors based on dataset size to prevent memory issues."""
    # Base limits
    max_absolute_neighbors = 10
   
    # For very large datasets, be more conservative
    if n_stations > 2000:
        max_safe = min(3, requested_neighbors)
    elif n_stations > 1000:
        max_safe = min(5, requested_neighbors)
    elif n_stations > 500:
        max_safe = min(8, requested_neighbors)
    else:
        max_safe = min(max_absolute_neighbors, requested_neighbors)
   
    # Final safety check based on total pairs
    total_pairs = n_stations * max_safe
    if total_pairs > 150000:  # Conservative threshold
        max_safe = max(1, 150000 // n_stations)
       
    return max_safe


def step_11_neighbor_features(df_feat: pl.DataFrame, n_neighbors: int, checkpoint_dir: str, bike_models: list) -> pl.DataFrame:
    """Step 11: Calculate neighbor demand and final cleanup."""
    print("[Step 11/11] Calculating neighbor demand and final cleanup...")
   
    checkpoint_path = os.path.join(checkpoint_dir, "step11_final.parquet")
    df_cached = _load_checkpoint_if_exists(checkpoint_path, "final features")
   
    if df_cached is not None:
        return df_cached

    # Log initial memory usage
    _log_memory_usage("step 11 start")

    # Load stations for neighbor calculation
    stations = pl.read_parquet(os.path.join(checkpoint_dir, "step06_stations.parquet"))
   
    step_tasks = ["Adding coordinates", "Computing neighbors", "Calculating neighbor demand", "Adding lag features", "Final cleanup"]
    with tqdm(total=len(step_tasks), desc="Step 11", leave=False) as pbar:
        pbar.set_description("Adding coordinates")
        df_feat = df_feat.join(stations.select(["station_id", "lat", "lon"]),
                               on="station_id", how="left")
        pbar.update(1)
        _log_memory_usage("after coordinates join")

        if n_neighbors > 0 and stations.height > 0:
            pbar.set_description("Computing neighbors")
            coords_pd = stations.drop_nulls(subset=["lat", "lon"]).to_pandas()
           
            # Calculate safe number of neighbors based on dataset size
            max_safe_neighbors = _calculate_safe_neighbor_limit(len(coords_pd), n_neighbors)
            if max_safe_neighbors < n_neighbors:
                print(f"  -> Dataset has {len(coords_pd)} stations. Limiting neighbors from {n_neighbors} to {max_safe_neighbors} to prevent memory issues.")

            try:
                nbrs = NearestNeighbors(
                    n_neighbors=min(max_safe_neighbors + 1, len(coords_pd)),
                    algorithm="ball_tree",
                    metric=lambda a, b: _haversine_np(a[1], a[0], b[1], b[0])
                )
                nbrs.fit(coords_pd[["lat", "lon"]].to_numpy())
                _, ind = nbrs.kneighbors(coords_pd[["lat", "lon"]].to_numpy())

                # Create neighbor mapping with smaller chunks to avoid memory issues
                neighbor_map = pl.DataFrame({
                    "station_id": np.repeat(coords_pd["station_id"].values,
                                            ind.shape[1] - 1),
                    "neighbor_id": coords_pd["station_id"].values[ind[:, 1:].ravel()]
                })
               
                _log_memory_usage("after neighbor computation")
                pbar.update(1)

                pbar.set_description("Calculating neighbor demand")
               
                # Process neighbor demand in smaller chunks to avoid memory explosion
                neighbor_dep = (
                    df_feat.select(["ts_start", "station_id", "dep_last_DT"])
                           .rename({"station_id": "neighbor_id",
                                    "dep_last_DT": "neighbor_dep"})
                )
               
                # Process join in chunks if the neighbor_map is very large
                chunk_size = 50000  # Process 50k neighbor relationships at a time
                if neighbor_map.height > chunk_size:
                    print(f"  -> Processing {neighbor_map.height} neighbor relationships in chunks of {chunk_size}")
                   
                    near_sum_chunks = []
                    for i in range(0, neighbor_map.height, chunk_size):
                        chunk_end = min(i + chunk_size, neighbor_map.height)
                        neighbor_chunk = neighbor_map.slice(i, chunk_end - i)
                       
                        chunk_sum = (
                            neighbor_chunk.join(neighbor_dep, on="neighbor_id", how="inner")
                                         .group_by(["station_id", "ts_start"])
                                        .agg(pl.sum("neighbor_dep").alias("near_dep_sum_DT_chunk"))
                        )
                        near_sum_chunks.append(chunk_sum)
                       
                        # Force garbage collection between chunks
                        gc.collect()
                   
                    # Combine all chunks
                    if near_sum_chunks:
                        near_sum_combined = pl.concat(near_sum_chunks)
                        near_sum = (
                            near_sum_combined.group_by(["station_id", "ts_start"])
                                           .agg(pl.sum("near_dep_sum_DT_chunk").alias("near_dep_sum_DT"))
                        )
                    else:
                        # Create empty result if no chunks
                        near_sum = pl.DataFrame({
                            "station_id": [],
                            "ts_start": [],
                            "near_dep_sum_DT": []
                        }).cast({"station_id": pl.Int64, "ts_start": df_feat["ts_start"].dtype, "near_dep_sum_DT": pl.Float64})
                else:
                    # Process normally for smaller datasets
                    near_sum = (
                        neighbor_map.join(neighbor_dep, on="neighbor_id", how="inner")
                                     .group_by(["station_id", "ts_start"])
                                    .agg(pl.sum("neighbor_dep").alias("near_dep_sum_DT"))
                    )

                df_feat = df_feat.join(near_sum, on=["station_id", "ts_start"], how="left")
                _log_memory_usage("after neighbor demand calculation")
                pbar.update(1)
               
                # Clean up intermediate objects
                del neighbor_map, neighbor_dep, near_sum
                gc.collect()
               
            except Exception as e:
                print(f"  -> Error in neighbor calculation: {e}")
                print("  -> Continuing without neighbor features...")
                # Add empty neighbor column
                df_feat = df_feat.with_columns(pl.lit(None).alias("near_dep_sum_DT").cast(pl.Float64))
                pbar.update(1)
        else:
            # Add empty neighbor column when no neighbors requested
            df_feat = df_feat.with_columns(pl.lit(None).alias("near_dep_sum_DT").cast(pl.Float64))
            pbar.update(2)  # Skip neighbor computation steps
       
        pbar.set_description("Adding lag features")
        df_feat = df_feat.with_columns(
            pl.col("near_dep_sum_DT").shift(1).over("station_id").alias("near_dep_lag_1")
        )
        pbar.update(1)
        _log_memory_usage("after lag features")

        pbar.set_description("Final cleanup")
        lag_cols = [c for c in df_feat.columns if c.startswith("dep_lag_") or c.startswith("arr_lag_")]
        fill_zero = [
            "dep_last_DT", "y_arrivals_next_DT", "near_dep_sum_DT",
            "arr_last_DT", "near_dep_lag_1"
        ] + lag_cols
       
        if bike_models:
            fill_zero.extend([f"model_{model}_cnt" for model in bike_models if model])

        existing_fill = [c for c in fill_zero if c in df_feat.columns]
        df_feat = df_feat.with_columns([pl.col(c).fill_null(0) for c in existing_fill])

        # Weather features cleanup if they exist
        weather_cols_to_fill = [c for c in df_feat.columns if c.startswith("weather_") and c != "weather_code_cat"]
        if weather_cols_to_fill:
            df_feat = df_feat.sort("ts_start")
            df_feat = df_feat.with_columns([
                pl.col(c).fill_null(strategy="forward") for c in weather_cols_to_fill
            ])
            for col in weather_cols_to_fill:
                 if df_feat[col].is_null().any():
                     mean_val = df_feat[col].mean()
                     df_feat = df_feat.with_columns(pl.col(col).fill_null(mean_val))
       
        # Rellenar rolling window NaNs
        for col in ["dep_ma_24h", "dep_std_24h", "dep_ratio_DT_24h"]:
            if col in df_feat.columns:
                df_feat = df_feat.with_columns(pl.col(col).fill_null(0))

        # One-hot encode weather categories
        if "weather_code_cat" in df_feat.columns:
             df_feat = df_feat.to_dummies(columns=["weather_code_cat"])

        # Drop auxiliary columns
        cols_to_drop = ["lon", "lat", "hour", "day", "month", "dow"]
        existing_cols_to_drop = [c for c in cols_to_drop if c in df_feat.columns]
        df_feat = df_feat.drop(existing_cols_to_drop)
        pbar.update(1)
        _log_memory_usage("after final cleanup")

    # Force garbage collection before saving
    gc.collect()
    _save_checkpoint(df_feat, checkpoint_path, "final features", use_streaming=True)
    _log_memory_usage("step 11 complete")
   
    return df_feat


def engineer_ecobici_features(
    df_raw: pd.DataFrame,
    dt_minutes: int = 30,
    n_neighbors: int = 5,
    clear_checkpoints: bool = False,
    rolling_chunk_size: int = 50,
    enable_validation: bool = True,
    abort_on_validation_failure: bool = False
) -> pd.DataFrame:
    """
    Parametrized pipeline for feature-engineering optimized with Polars.
   
    Each step is now a separate function with proper checkpoint management
    that only loads what's needed for that specific step.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original trips + weather data
    dt_minutes : int, default 30
        Time window granularity in minutes
    n_neighbors : int, default 5
        Number of nearest neighbor stations for demand features
    clear_checkpoints : bool, default False
        Whether to clear all checkpoints and start fresh
    rolling_chunk_size : int, default 50
        Number of stations to process at once in step 10 to avoid memory issues
    enable_validation : bool, default True
        Whether to run data quality validation at each step
    abort_on_validation_failure : bool, default False
        Whether to abort pipeline on validation failures (vs. continue with warnings)

    Returns
    -------
    pd.DataFrame
        Final feature dataset
    """
    print("🚴‍♂️  Parametrized Feature Engineering Pipeline (11 steps)")
    print("=" * 60)

    # Setup checkpointing
    checkpoint_dir = "feature_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
   
    if clear_checkpoints:
        clear_feature_checkpoints(checkpoint_dir)
   
    print(f"  Checkpoints directory: '{checkpoint_dir}/'")
    print(f"  Input raw data shape: {df_raw.shape}")
    print(f"  Validation enabled: {enable_validation}")
    print(f"  Abort on validation failure: {abort_on_validation_failure}")
   
    # Initialize data quality tester
    tester = None
    if enable_validation and DataQualityTester is not None:
        try:
            tester = DataQualityTester(checkpoint_dir)
            print("  ✅ Data quality tester initialized")
        except Exception as e:
            print(f"  ⚠️ Could not initialize tester: {e}")
   
    _log_memory_usage("initialization")

    # Validate raw data first
    if tester is not None:
        print("\n🔍 Validating raw data quality...")
        if not _validate_step_with_action(
            tester,
            lambda: tester.test_raw_data_quality(df_raw),
            "Raw Data Quality",
            abort_on_validation_failure
        ):
            raise ValueError("Raw data validation failed and abort_on_failure=True")

    # Check for final result first
    final_result_path = os.path.join(checkpoint_dir, "df_feat_final_result.parquet")
    if os.path.exists(final_result_path) and not clear_checkpoints:
        print(f"  🎉 Final result already exists! Loading from: {final_result_path}")
        try:
            df_final = pl.read_parquet(final_result_path)
            print(f"  ✅ Successfully loaded final result with shape: {df_final.shape}")
           
            # Validate final result if tester available
            if tester is not None:
                _validate_step_with_action(
                    tester,
                    tester.test_step_11_neighbor_features,
                    "Final Result Validation",
                    abort_on_validation_failure
                )
           
            return df_final.to_pandas(use_pyarrow_extension_array=True)
        except Exception as e:
            print(f"  ⚠️ Error loading final result: {e}. Will recreate...")
            os.remove(final_result_path)

    # Execute pipeline steps with validation
    print("\n[Step 1/11] Converting to Polars and normalizing dates...")
    df, bike_models = step_01_convert_to_polars(df_raw, dt_minutes, checkpoint_dir)
   
    if not _validate_step_with_action(
        tester,
        lambda: tester.test_step_01_polars_conversion(df_raw, dt_minutes) if tester else True,
        "Step 1: Polars Conversion",
        abort_on_validation_failure
    ):
        raise ValueError("Step 1 validation failed")
   
    print("\n[Step 2/11] Creating departures table...")
    step_02_create_departures_table(df, bike_models, checkpoint_dir)
    if not _validate_step_with_action(
        tester,
        tester.test_step_02_departures_table if tester else lambda: True,
        "Step 2: Departures Table",
        abort_on_validation_failure
    ):
        raise ValueError("Step 2 validation failed")
   
    print("\n[Step 3/11] Creating arrivals table...")
    step_03_create_arrivals_table(df, checkpoint_dir)
    if not _validate_step_with_action(
        tester,
        tester.test_step_03_arrivals_table if tester else lambda: True,
        "Step 3: Arrivals Table",
        abort_on_validation_failure
    ):
        raise ValueError("Step 3 validation failed")
   
    print("\n[Step 4/11] Creating shifted arrivals...")
    step_04_create_shifted_arrivals(dt_minutes, checkpoint_dir)
    if not _validate_step_with_action(
        tester,
        tester.test_step_04_shifted_arrivals if tester else lambda: True,
        "Step 4: Shifted Arrivals",
        abort_on_validation_failure
    ):
        raise ValueError("Step 4 validation failed")
   
    print("\n[Step 5/11] Creating weather table...")
    weather = step_05_create_weather_table(df, checkpoint_dir)
    has_weather = weather is not None
    if not _validate_step_with_action(
        tester,
        tester.test_step_05_weather_table if tester else lambda: True,
        "Step 5: Weather Table",
        abort_on_validation_failure
    ):
        raise ValueError("Step 5 validation failed")
   
    print("\n[Step 6/11] Creating station metadata...")
    stations = step_06_create_station_metadata(df, checkpoint_dir)
    if not _validate_step_with_action(
        tester,
        tester.test_step_06_station_metadata if tester else lambda: True,
        "Step 6: Station Metadata",
        abort_on_validation_failure
    ):
        raise ValueError("Step 6 validation failed")
   
    print("\n[Step 7/11] Building skeleton...")
    step_07_build_skeleton(df, stations, dt_minutes, checkpoint_dir)
    if not _validate_step_with_action(
        tester,
        tester.test_step_07_skeleton if tester else lambda: True,
        "Step 7: Skeleton",
        abort_on_validation_failure
    ):
        raise ValueError("Step 7 validation failed")
   
    # Clear main dataframe from memory after extracting what we need
    del df
    gc.collect()
    _log_memory_usage("after clearing main dataframe")
   
    # Continue with feature assembly
    print("\n[Step 8/11] Joining base features...")
    df_feat = step_08_join_base_features(checkpoint_dir, has_weather)
    if not _validate_step_with_action(
        tester,
        tester.test_step_08_joined_features if tester else lambda: True,
        "Step 8: Joined Features",
        abort_on_validation_failure
    ):
        raise ValueError("Step 8 validation failed")
   
    print("\n[Step 9/11] Creating temporal features...")
    df_feat = step_09_temporal_features(df_feat, holiday_dates=[], checkpoint_dir=checkpoint_dir)
    if not _validate_step_with_action(
        tester,
        tester.test_step_09_temporal_features if tester else lambda: True,
        "Step 9: Temporal Features",
        abort_on_validation_failure
    ):
        raise ValueError("Step 9 validation failed")
   
    print("\n[Step 10/11] Creating rolling features...")
    df_feat = step_10_rolling_features_fast(
        df_feat,
        dt_minutes=dt_minutes,
        checkpoint_dir=checkpoint_dir,
        chunk_size=rolling_chunk_size,
    )

    if not _validate_step_with_action(
        tester,
        tester.test_step_10_rolling_features if tester else lambda: True,
        "Step 10: Rolling Features",
        abort_on_validation_failure
    ):
        raise ValueError("Step 10 validation failed")
   
    print("\n[Step 11/11] Creating neighbor features...")
    df_feat = step_11_neighbor_features(df_feat, n_neighbors, checkpoint_dir, bike_models)
    if not _validate_step_with_action(
        tester,
        tester.test_step_11_neighbor_features if tester else lambda: True,
        "Step 11: Neighbor Features",
        abort_on_validation_failure
    ):
        raise ValueError("Step 11 validation failed")
   
    # Save final result
    try:
        df_feat.lazy().sink_parquet(final_result_path, compression="snappy", row_group_size=10000)
        print(f"  ✅ Saved final result: {final_result_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save final result: {e}")
   
    # Final validation summary
    if tester is not None:
        print("\n🔍 Running final pipeline validation summary...")
        try:
            tester.print_summary()
        except Exception as e:
            print(f"  ⚠️ Could not print validation summary: {e}")
   
    return df_feat.to_pandas(use_pyarrow_extension_array=True)


def pivot_features_for_global_model(
    df_long: Union[pd.DataFrame, pl.DataFrame],
    checkpoint_dir: str = "feature_checkpoints",
    enable_validation: bool = True
) -> None:
    """
    Transforms the long-format feature DataFrame into a wide format suitable for a
    global prediction model and saves the results as Parquet files.

    In this format, each row represents a single timestamp. The feature columns
    contain the station-specific features for all stations, concatenated in a
    consistent order, followed by the global features for that timestamp. The
    target variable is similarly a single row containing the concatenated
    arrival predictions for all stations.

    This structure allows a model to learn from the state of the entire system
    at time `t` to predict the entire system's state at time `t+1`.

    The final `X_wide.parquet` and `y_wide.parquet` files are saved in the
    `checkpoint_dir`.

    Args:
        df_long (Union[pd.DataFrame, pl.DataFrame]):
            The output from `engineer_ecobici_features`. It must be in a long
            format with 'ts_start' and 'station_id' columns.
        checkpoint_dir (str):
            Directory to save intermediate and final Parquet files.
    """
    print("🔄 Pivoting data to wide format for global model...")

    # Initialize tester for validation
    tester = None
    if enable_validation and DataQualityTester is not None:
        try:
            tester = DataQualityTester(checkpoint_dir)
        except Exception as e:
            print(f"  ⚠️ Could not initialize tester for pivot validation: {e}")    # --- File Paths ---
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    x_wide_path = os.path.join(checkpoint_dir, "X_wide.parquet")
    y_wide_path = os.path.join(checkpoint_dir, "y_wide.parquet")

    if os.path.exists(x_wide_path) and os.path.exists(y_wide_path):
        print(f"  -> Wide data already exists in '{checkpoint_dir}/'. Skipping.")
       
        # Validate existing pivot files
        if tester is not None:
            _validate_step_with_action(
                tester,
                tester.test_pivot_features,
                "Existing Pivot Validation",
                abort_on_failure=False  # Don't abort on pivot validation failure
            )
       
        print("✅ Pivoting complete (loaded from cache).")
        return

    # Allow either Pandas or Polars input for flexibility
    if isinstance(df_long, pl.DataFrame):
        df = df_long.clone()
    elif isinstance(df_long, pd.DataFrame):
        df = pl.from_pandas(df_long)
    else:
        raise TypeError(
            "df_long must be a pandas.DataFrame or polars.DataFrame."
        )

    df = df.sort("ts_start", "station_id")

    # --- Column Identification ---
    print("\n[Step 1/3] Identifying columns...")
    target_col = "y_arrivals_next_DT"
    id_cols = ["ts_start", "station_id"]

    station_specific_patterns = [
        "dep_", "arr_", "share_", "trip_", "model_", "y_arrivals", "near_dep"
    ]

    station_specific_cols = [
        c for c in df.columns
        if any(pat in c for pat in station_specific_patterns) and c != target_col
    ]

    global_cols = [
        c for c in df.columns
        if c not in station_specific_cols + id_cols + [target_col]
    ]

    print(f"  -> Identified {len(station_specific_cols)} station-specific and {len(global_cols)} global features.")

    # --- Feature (X) Matrix Construction ---
    print("\n[Step 2/3] Constructing feature (X) matrix...")

    # 1. Pivot station-specific features
    x_station_wide_path = os.path.join(checkpoint_dir, "X_station_wide.parquet")
    if os.path.exists(x_station_wide_path):
        print(f"  -> Loading station-specific features from checkpoint: {x_station_wide_path}")
        X_station_wide = pl.read_parquet(x_station_wide_path)
    else:
        print("  -> Pivoting station-specific features...")
        X_station_wide = df.pivot(
            index="ts_start",
            columns="station_id",
            values=station_specific_cols
        )
        print(f"  -> Saving checkpoint: {x_station_wide_path}")
        X_station_wide.write_parquet(x_station_wide_path)

    # 2. Get global features (cheap, no checkpoint needed)
    print("  -> Extracting global features...")
    # Ensure 'ts_start' is retained for the subsequent join
    X_global = (
        df.group_by("ts_start").first().select(["ts_start"] + global_cols)
    )

    # 3. Join them together
    print("  -> Joining global and station-specific features...")
    X_wide = X_global.join(X_station_wide, on="ts_start", how="inner")

    print(f"  -> Saving final features matrix: {x_wide_path}")
    X_wide.write_parquet(x_wide_path)

    # --- Target (y) Matrix Construction ---
    print("\n[Step 3/3] Constructing target (y) matrix...")

    # Pivot target variable
    y_wide_unaligned_path = os.path.join(checkpoint_dir, "y_wide_unaligned.parquet")
    if os.path.exists(y_wide_unaligned_path):
        print(f"  -> Loading unaligned target from checkpoint: {y_wide_unaligned_path}")
        y_wide = pl.read_parquet(y_wide_unaligned_path)
    else:
        print("  -> Pivoting target variable...")
        y_wide = df.pivot(
            index="ts_start",
            columns="station_id",
            values=target_col
        )
        print(f"  -> Saving checkpoint: {y_wide_unaligned_path}")
        y_wide.write_parquet(y_wide_unaligned_path)

    # Ensure y_wide has the same timestamps and order as X_wide
    print("  -> Aligning target matrix with feature matrix timestamps...")
    y_wide = X_wide.select("ts_start").join(y_wide, on="ts_start", how="left")

    print(f"  -> Saving final target matrix: {y_wide_path}")
    y_wide.write_parquet(y_wide_path)

    print("-" * 50)
    print(f"  Original long shape: {df.shape}")
    print(f"  Pivoted X_wide shape: {X_wide.shape} -> saved to X_wide.parquet")
    print(f"  Pivoted y_wide shape: {y_wide.shape} -> saved to y_wide.parquet")
   
    # Validate pivot results
    if tester is not None:
        print("\n🔍 Validating pivot results...")
        _validate_step_with_action(
            tester,
            tester.test_pivot_features,
            "Pivot Results Validation",
            abort_on_failure=False  # Don't abort on pivot validation failure
        )
   
    print("✅ Pivoting complete.")


def _check_memory_and_abort_if_needed(operation_name: str, max_memory_mb: int = 8000) -> bool:
    """Check if memory usage is too high and recommend aborting to prevent kernel crash."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
       
        if memory_mb > max_memory_mb:
            print(f"  ⚠️ WARNING: Memory usage ({memory_mb:.1f} MB) too high for {operation_name}")
            print(f"  ⚠️ Recommended: Reduce n_neighbors or use smaller dataset to prevent kernel crash")
            return True  # Abort recommended
        return False  # Continue
    except ImportError:
        return False  # Continue if psutil not available
    except Exception:
        return False  # Continue on any error


def _join_base_features_trip(dep: pl.DataFrame, arr: pl.DataFrame, arr_next: pl.DataFrame, weather: pl.DataFrame | None) -> pl.DataFrame:
    """Lightweight join of base tables (no full skeleton).

    The result mirrors the structure produced by `step_08_join_base_features`,
    but is restricted to (station_id, ts_start) combinations *present in the
    original trips dataset*. This dramatically reduces the row count and avoids
    the costly station × time skeleton.
    """
    df_feat = dep.join(arr, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(arr_next, on=["station_id", "ts_start"], how="left")
    if weather is not None:
        df_feat = df_feat.join(weather, on="ts_start", how="left")
    return df_feat


# ---------------------------------------------------------------------------
# Trip-level pipeline (avoids full skeleton)  
# ---------------------------------------------------------------------------

def engineer_ecobici_features_per_trip(
    df_raw: pd.DataFrame,
    dt_minutes: int = 30,
    n_neighbors: int = 5,
    clear_checkpoints: bool = False,
    rolling_chunk_size: int = 50,
    enable_validation: bool = True,
    abort_on_validation_failure: bool = False,
) -> pd.DataFrame:
    """Memory-efficient trip-level feature-engineering pipeline.

    This function reproduces (as faithfully as possible) the feature set
    generated by `engineer_ecobici_features`, but **returns one row per trip**    instead of one row per station–time interval.  The implementation re-uses
    optimized helper steps (departures, arrivals, weather, rolling, neighbour
    demand, …) while *skipping* the expensive **station × full-timeline
    skeleton**.  All computations are therefore restricted to the
    `(station_id, ts_start)` pairs that actually occur in the data, leading to
    drastically lower memory consumption.
    """
    
    print("🚴‍♂️  Trip-level Feature Engineering (lightweight mode)")
    print("=" * 60)

    checkpoint_dir = "feature_checkpoints"
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if clear_checkpoints:
        clear_feature_checkpoints(checkpoint_dir)

    # ------------------------------------------------------------------
    # 1. Polars conversion & basic preprocessing
    # ------------------------------------------------------------------
    df_polars, bike_models = step_01_convert_to_polars(df_raw, dt_minutes, checkpoint_dir)

    # ------------------------------------------------------------------
    # 2-6. Core aggregations (same as full pipeline)
    # ------------------------------------------------------------------
    dep = step_02_create_departures_table(df_polars, bike_models, checkpoint_dir)
    arr = step_03_create_arrivals_table(df_polars, checkpoint_dir)
    arr_next = step_04_create_shifted_arrivals(dt_minutes, checkpoint_dir)
    weather = step_05_create_weather_table(df_polars, checkpoint_dir)
    stations = step_06_create_station_metadata(df_polars, checkpoint_dir)

    # ------------------------------------------------------------------
    # 7. Join base features WITHOUT a full skeleton
    # ------------------------------------------------------------------
    has_weather = weather is not None
    df_feat = _join_base_features_trip(dep, arr, arr_next, weather)

    # ------------------------------------------------------------------
    # 8. Temporal/calendar features (windowed, memory-safe)
    # ------------------------------------------------------------------
    df_feat = _create_temporal_features(
        df_feat,
        checkpoint_dir=checkpoint_dir,
        temporal_path=os.path.join(checkpoint_dir, "step09_temporal_trip.parquet"),
        holiday_dates=[],
    )

    # ------------------------------------------------------------------
    # 9. Rolling demand history (24-h) – same helper as full pipeline
    # ------------------------------------------------------------------
    df_feat = step_10_rolling_features_fast(
        df_feat,
        dt_minutes=dt_minutes,
        checkpoint_dir=checkpoint_dir,
        chunk_size=rolling_chunk_size,
    )    # ------------------------------------------------------------------
    # 10. Neighbour demand features & final cleanup
    # ------------------------------------------------------------------
    df_feat = step_11_neighbor_features(
        df_feat,
        n_neighbors=n_neighbors,
        checkpoint_dir=checkpoint_dir,
        bike_models=bike_models,
    )
    
    # Ensure consistent data types for df_feat before merging
    print("🔧 Standardizing data types in feature DataFrame...")
    df_feat = df_feat.with_columns([
        pl.col("station_id").cast(pl.Int32),  # Ensure station_id is int32
        pl.col("ts_start").cast(pl.Datetime(time_unit="us"))  # Ensure microsecond precision
    ])
    _log_memory_usage("after type standardization")

    # ------------------------------------------------------------------
    # 11. Merge engineered station-time features back onto *every trip*
    # ------------------------------------------------------------------
    print("\n🔗  Merging features onto trip records (memory-efficient chunked processing)…")
    
    # Immediate memory cleanup before starting merge
    print("🧹 Performing immediate memory cleanup...")
    gc.collect()
    _log_memory_usage("immediate cleanup")    # Build keys that match the aggregated tables with optimized dtypes
    df_trips = df_raw.copy()
    df_trips["ts_start"] = pd.to_datetime(df_trips["fecha_origen_recorrido"]).dt.floor(f"{dt_minutes}min")
    df_trips["station_id"] = df_trips["id_estacion_origen"].astype("int32")  # Use int32 instead of int64 to save memory
      # Optimize other numeric columns in df_trips to reduce memory
    for col in df_trips.select_dtypes(include=['float64']).columns:
        if df_trips[col].max() < 3.4e38 and df_trips[col].min() > -3.4e38:
            df_trips[col] = df_trips[col].astype('float32')
    
    for col in df_trips.select_dtypes(include=['int64']).columns:
        if col != 'station_id':  # Already handled station_id
            col_max = df_trips[col].max()
            col_min = df_trips[col].min()
            if col_min >= -2147483648 and col_max <= 2147483647:
                df_trips[col] = df_trips[col].astype('int32')
    
    _log_memory_usage("after data type optimization")# EMERGENCY: Check if we're in a critical memory situation
    try:
        process = psutil.Process(os.getpid())
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024 / 1024 / 1024
        
        # If we're using >6GB and have <2GB available, we need EMERGENCY mode
        if current_memory_mb > 6000 and available_gb < 2:
            print(f"🚨 EMERGENCY MODE: Current usage {current_memory_mb:.1f}MB, Available {available_gb:.1f}GB")
            print("🚨 Attempting aggressive memory cleanup before proceeding...")
              # Force aggressive garbage collection
            for i in range(3):
                gc.collect()
                
            # Check memory again
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            available_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
            print(f"🚨 After cleanup: {current_memory_mb:.1f}MB used, {available_gb:.1f}GB available")
            
            # Use EXTREMELY small chunks in emergency mode
            chunk_size = 5000
            print(f"🚨 Using EMERGENCY chunk size: {chunk_size:,} trips per chunk")
            
        elif available_gb > 8:
            chunk_size = 150000  # Larger chunks if plenty of memory
        elif available_gb > 4:
            chunk_size = 100000  # Standard chunk size
        elif available_gb > 2:
            chunk_size = 50000   # Smaller chunks if limited memory
        else:
            chunk_size = 10000   # Very small chunks if very limited memory
            print(f"  ⚠️  Low system memory ({available_gb:.1f} GB), using small chunks")
    except:
        chunk_size = 5000  # Ultra-safe default if can't determine memory    
    total_trips = len(df_trips)
    n_chunks = (total_trips + chunk_size - 1) // chunk_size
    print(f"  Processing {total_trips:,} trips in {n_chunks} chunks of {chunk_size:,}")
    _log_memory_usage("before chunked merge")
    
    # EMERGENCY CHECK: If memory situation is too bad, suggest alternatives
    try:
        process = psutil.Process(os.getpid())
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024 / 1024 / 1024
        
        if current_memory_mb > 7000 and available_gb < 1.5:
            print("\n🚨🚨🚨 CRITICAL MEMORY SITUATION 🚨🚨🚨")
            print("Your system is at risk of crashing. Here are your options:")
            print("\n1. 🔄 RESTART KERNEL (RECOMMENDED):")
            print("   - Restart your Jupyter kernel to free all memory")
            print("   - Then run ONLY the feature engineering cell")
            print("\n2. 💾 FREE UP SYSTEM MEMORY:")
            print("   - Close other applications (browsers, etc.)")
            print("   - Wait a few minutes and try again")
            print("\n3. ⚡ CONTINUE WITH EXTREME CAUTION:")
            print("   - Will use micro-chunks but may still crash")
            print("   - Save your work first!")
            
            user_choice = input("\nType 'continue' to proceed anyway, or 'abort' to stop: ").lower().strip()
            if user_choice != 'continue':
                print("🛑 Aborting to prevent system crash. Please restart kernel.")
                return df_raw  # Return original data unchanged
            else:
                print("⚠️ Proceeding with extreme caution...")
                chunk_size = min(chunk_size, 2000)  # Micro-chunks
                print(f"⚠️ Using micro-chunks of {chunk_size:,} trips")
                
    except:
        pass
    
    # Initialize list to store results
    merged_chunks = []
    
    # Process trips in chunks to avoid memory issues
    for i in tqdm(range(n_chunks), desc="Merging chunks"):
        # Check memory before processing each chunk
        if _check_memory_and_abort_if_needed(f"chunk {i+1}/{n_chunks}", max_memory_mb=15700):
            print(f"  ⚠️  Aborting due to high memory usage. Consider restarting kernel and reducing chunk_size.")
            break
            
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_trips)
        
        # Get chunk of trips
        trips_chunk = df_trips.iloc[start_idx:end_idx].copy()
          # Get unique station-time combinations for this chunk
        chunk_keys = trips_chunk[["station_id", "ts_start"]].drop_duplicates()
          # Convert to Polars with proper datetime precision matching
        chunk_keys_pl = pl.from_pandas(chunk_keys).with_columns([
            pl.col("ts_start").cast(pl.Datetime(time_unit="us")),  # Convert to microseconds to match df_feat
            pl.col("station_id").cast(pl.Int32)  # Match the int32 type from df_trips
        ])
        
        # Filter features DataFrame to only relevant keys for this chunk
        df_feat_filtered = df_feat.join(
            chunk_keys_pl,
            on=["station_id", "ts_start"],
            how="inner"
        )
          # Convert only the filtered subset to pandas with optimized dtypes
        df_feat_chunk_pd = df_feat_filtered.to_pandas(use_pyarrow_extension_array=False)
        
        # Optimize data types to reduce memory usage
        for col in df_feat_chunk_pd.columns:
            if df_feat_chunk_pd[col].dtype == 'float64':
                # Check if we can safely downcast to float32
                if df_feat_chunk_pd[col].max() < 3.4e38 and df_feat_chunk_pd[col].min() > -3.4e38:
                    df_feat_chunk_pd[col] = df_feat_chunk_pd[col].astype('float32')
            elif df_feat_chunk_pd[col].dtype == 'int64':
                # Check if we can safely downcast to smaller int types
                col_max = df_feat_chunk_pd[col].max()
                col_min = df_feat_chunk_pd[col].min()
                if col_min >= -128 and col_max <= 127:
                    df_feat_chunk_pd[col] = df_feat_chunk_pd[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df_feat_chunk_pd[col] = df_feat_chunk_pd[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df_feat_chunk_pd[col] = df_feat_chunk_pd[col].astype('int32')
        
        # Merge this chunk
        chunk_merged = trips_chunk.merge(
            df_feat_chunk_pd,
            how="left",
            on=["station_id", "ts_start"],
            copy=False,
        )
        
        merged_chunks.append(chunk_merged)
        
        # Force garbage collection between chunks
        del trips_chunk, chunk_keys, chunk_keys_pl, df_feat_filtered, df_feat_chunk_pd, chunk_merged
        gc.collect()
      # Combine all chunks
    print("  Combining all chunks...")
    df_trip_feat = pd.concat(merged_chunks, ignore_index=True)
    
    # Clean up
    del merged_chunks, df_trips
    gc.collect()
    _log_memory_usage("after chunked merge complete")

    # ------------------------------------------------------------------
    # 12. Per-trip temporal context (hour, cyclical encoding, …)  
    # We re-use the same formulas but operate directly on precise timestamps.
    # ------------------------------------------------------------------
    ts_series = pd.to_datetime(df_trip_feat["fecha_origen_recorrido"])
    df_trip_feat["hour"] = ts_series.dt.hour.astype("int8")
    df_trip_feat["dow"] = ts_series.dt.dayofweek.astype("int8")
    df_trip_feat["month"] = ts_series.dt.month.astype("int8")
    df_trip_feat["day"] = ts_series.dt.day.astype("int8")

    # Cyclical encodings
    df_trip_feat["sin_hour"] = np.sin(df_trip_feat["hour"] * 2 * math.pi / 24).astype("float32")
    df_trip_feat["cos_hour"] = np.cos(df_trip_feat["hour"] * 2 * math.pi / 24).astype("float32")
    df_trip_feat["sin_dow"] = np.sin(df_trip_feat["dow"] * 2 * math.pi / 7).astype("float32")
    df_trip_feat["cos_dow"] = np.cos(df_trip_feat["dow"] * 2 * math.pi / 7).astype("float32")
    df_trip_feat["sin_month"] = np.sin(df_trip_feat["month"] * 2 * math.pi / 12).astype("float32")
    df_trip_feat["cos_month"] = np.cos(df_trip_feat["month"] * 2 * math.pi / 12).astype("float32")

    # Weekend / payday / vacation flags (Int8)
    df_trip_feat["is_weekend"] = df_trip_feat["dow"].isin([5, 6]).astype("int8")
    df_trip_feat["payday_flag"] = df_trip_feat["day"].isin([1, 15]).astype("int8")
    df_trip_feat["vacation_season"] = df_trip_feat["month"].isin([1, 7]).astype("int8")

    # Peak-commute (same rule as base pipeline)
    df_trip_feat["peak_commute"] = (
        df_trip_feat["hour"].isin([7, 8, 9, 17, 18, 19]) & (df_trip_feat["is_weekend"] == 0)
    ).astype("int8")    # Drop helper / duplicate columns to keep dataset tidy
    df_trip_feat.drop(columns=["ts_start"], inplace=True, errors="ignore")

    # ------------------------------------------------------------------
    # 13. Simple null checker (with null counts)
    # ------------------------------------------------------------------
    null_cols = df_trip_feat.columns[df_trip_feat.isnull().any()].tolist()
    if null_cols:
        null_counts = df_trip_feat[null_cols].isnull().sum()
        print("⚠️ Columns with null values:")
        for col in null_cols:
            print(f"   - {col}: {null_counts[col]} nulls")
    else:
        print("✅ No columns with null values.")

    # ------------------------------------------------------------------
    # 14. Final data type optimization to minimize memory usage
    # ------------------------------------------------------------------
    print("🔧 Final data type optimization...")
    
    # Optimize all remaining float64 and int64 columns
    for col in df_trip_feat.select_dtypes(include=['float64']).columns:
        if df_trip_feat[col].max() < 3.4e38 and df_trip_feat[col].min() > -3.4e38:
            df_trip_feat[col] = df_trip_feat[col].astype('float32')
    
    for col in df_trip_feat.select_dtypes(include=['int64']).columns:
        col_max = df_trip_feat[col].max()
        col_min = df_trip_feat[col].min()
        if col_min >= -128 and col_max <= 127:
            df_trip_feat[col] = df_trip_feat[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df_trip_feat[col] = df_trip_feat[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df_trip_feat[col] = df_trip_feat[col].astype('int32')
    
    _log_memory_usage("final optimization")

    print(f"  ✅ Final trip-level feature set shape: {df_trip_feat.shape}")
    return df_trip_feat
