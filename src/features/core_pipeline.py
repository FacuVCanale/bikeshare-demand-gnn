"""
Core feature engineering pipeline for EcoBici data.

This module contains the main feature engineering pipeline with checkpointing,
memory management, and optimized processing steps.
"""

import polars as pl
import numpy as np
import os
import gc
import psutil
import holidays
import math
from datetime import timedelta
from typing import Tuple, List, Optional
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def _log_memory_usage(step_name: str):
    """Log current memory usage for monitoring."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage after {step_name}: {memory_mb:.1f} MB")
    except:
        print(f"  Memory usage after {step_name}: Unable to measure")


def _get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


def clear_feature_checkpoints(checkpoint_dir: str = "feature_checkpoints"):
    """
    Clear all checkpoint files to force fresh computation.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    """
    print(f"Clearing checkpoints in {checkpoint_dir}/...")
    
    if not os.path.exists(checkpoint_dir):
        print(f"  -> Directory {checkpoint_dir}/ doesn't exist")
        return
    
    removed_count = 0
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.parquet'):
            file_path = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"  -> Error removing {filename}: {e}")
    
    print(f"  -> Removed {removed_count} checkpoint files")


def _load_checkpoint_if_exists(path: str, description: str) -> Optional[pl.DataFrame]:
    """Load checkpoint if it exists, otherwise return None."""
    if os.path.exists(path):
        print(f"  -> Loading {description} from checkpoint: {path}")
        return pl.read_parquet(path)
    return None


def _save_checkpoint(df: pl.DataFrame, path: str, description: str, use_streaming: bool = True):
    """Save checkpoint with proper error handling and chunked writing."""
    try:
        # For large datasets, write in chunks to avoid memory issues
        df_size_mb = df.estimated_size('mb')
        
        if df_size_mb > 500:  # If larger than 500MB, write in chunks
            print(f"  -> Large dataset ({df_size_mb:.1f}MB), writing in chunks...")
            _write_parquet_in_chunks(df, path, chunk_rows=50000)
        elif use_streaming:
            df.lazy().sink_parquet(path, compression="snappy", row_group_size=10000)
        else:
            df.write_parquet(path, compression="snappy", row_group_size=10000)
        print(f"  -> Saved {description} checkpoint: {path}")
    except Exception as e:
        print(f"  -> Warning: Could not save checkpoint {path}: {e}")


def _write_parquet_in_chunks(df: pl.DataFrame, path: str, chunk_rows: int = 50000):
    """Write large DataFrame to parquet in chunks to avoid memory issues."""
    total_rows = df.height
    num_chunks = (total_rows + chunk_rows - 1) // chunk_rows
    
    print(f"     -> Writing {total_rows:,} rows in {num_chunks} chunks of {chunk_rows:,} rows each")
    
    # Write first chunk to create the file
    first_chunk = df.slice(0, min(chunk_rows, total_rows))
    first_chunk.write_parquet(path, compression="snappy", row_group_size=10000)
    del first_chunk
    
    # Append remaining chunks
    for i in range(1, num_chunks):
        start_idx = i * chunk_rows
        end_idx = min(start_idx + chunk_rows, total_rows)
        
        chunk = df.slice(start_idx, end_idx - start_idx)
        
        # Read existing data, combine with chunk, and write back
        existing_df = pl.read_parquet(path)
        combined_df = pl.concat([existing_df, chunk], how="vertical")
        combined_df.write_parquet(path, compression="snappy", row_group_size=10000)
        
        del chunk, existing_df, combined_df
        import gc
        gc.collect()
        
        print(f"     -> Wrote chunk {i+1}/{num_chunks}")


def _append_parquet_chunk(existing_path: str, new_chunk: pl.DataFrame):
    """Efficiently append a chunk to an existing parquet file."""
    try:
        # Read existing file lazily
        existing_lazy = pl.scan_parquet(existing_path)
        
        # Combine and write back in streaming mode
        combined = pl.concat([existing_lazy, new_chunk.lazy()], how="vertical")
        combined.sink_parquet(existing_path, compression="snappy", row_group_size=10000)
        
    except Exception as e:
        print(f"     -> Warning: Could not append chunk: {e}")
        # Fallback to regular method
        existing_df = pl.read_parquet(existing_path)
        combined_df = pl.concat([existing_df, new_chunk], how="vertical")
        combined_df.write_parquet(existing_path, compression="snappy", row_group_size=10000)
        del existing_df, combined_df


def _create_temporal_features(df_feat: pl.DataFrame, checkpoint_dir: str, temporal_path: str, holiday_dates: list) -> pl.DataFrame:
    """Create temporal features like hour, day of week, holidays, etc."""
    
    # check if temporal checkpoint exists
    temporal_checkpoint = _load_checkpoint_if_exists(temporal_path, "temporal features")
    if temporal_checkpoint is not None:
        return temporal_checkpoint
    
    print("  -> Creating temporal features...")
    
    # create temporal features
    df_feat = df_feat.with_columns([
        # basic time components
        pl.col("ts_start").dt.hour().alias("hour"),
        pl.col("ts_start").dt.weekday().alias("day_of_week"),  # 1=Monday, 7=Sunday
        pl.col("ts_start").dt.month().alias("month"),
        pl.col("ts_start").dt.day().alias("day"),
        pl.col("ts_start").dt.year().alias("year"),
        
        # cyclical encodings for better ML performance
        (2 * np.pi * pl.col("ts_start").dt.hour() / 24).sin().alias("hour_sin"),
        (2 * np.pi * pl.col("ts_start").dt.hour() / 24).cos().alias("hour_cos"),
        (2 * np.pi * pl.col("ts_start").dt.weekday() / 7).sin().alias("dow_sin"),
        (2 * np.pi * pl.col("ts_start").dt.weekday() / 7).cos().alias("dow_cos"),
        (2 * np.pi * pl.col("ts_start").dt.month() / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("ts_start").dt.month() / 12).cos().alias("month_cos"),
        
        # business logic features
        pl.col("ts_start").dt.weekday().is_in([6, 7]).alias("is_weekend"),
        pl.col("ts_start").dt.hour().is_between(7, 9).alias("is_morning_rush"),
        pl.col("ts_start").dt.hour().is_between(17, 19).alias("is_evening_rush"),
        pl.col("ts_start").dt.hour().is_between(6, 22).alias("is_daytime"),
        
        # seasonal features
        pl.col("ts_start").dt.month().is_in([12, 1, 2]).alias("is_summer"),
        pl.col("ts_start").dt.month().is_in([6, 7, 8]).alias("is_winter"),
        pl.col("ts_start").dt.month().is_in([3, 4, 5]).alias("is_autumn"),
        pl.col("ts_start").dt.month().is_in([9, 10, 11]).alias("is_spring")
    ])
    
    # add holiday features if holiday dates provided
    if holiday_dates:
        print(f"  -> Adding {len(holiday_dates)} holiday dates...")
        holiday_dates_pl = pl.Series(holiday_dates)
        df_feat = df_feat.with_columns([
            pl.col("ts_start").dt.date().is_in(holiday_dates_pl).alias("is_holiday")
        ])
    else:
        # create Argentina holidays automatically
        print("  -> Creating Argentina holidays...")
        
        # get date range from data
        start_year = df_feat.select(pl.col("ts_start").dt.year().min()).item()
        end_year = df_feat.select(pl.col("ts_start").dt.year().max()).item()
        
        # generate Argentina holidays
        arg_holidays = []
        for year in range(start_year, end_year + 1):
            arg_holidays.extend(holidays.Argentina(years=year).keys())
        
        if arg_holidays:
            df_feat = df_feat.with_columns([
                pl.col("ts_start").dt.date().is_in(arg_holidays).alias("is_holiday")
            ])
        else:
            df_feat = df_feat.with_columns([
                pl.lit(False).alias("is_holiday")
            ])
    
    # save temporal features checkpoint
    _save_checkpoint(df_feat, temporal_path, "temporal features", use_streaming=True)
    
    return df_feat


def step_01_normalize_polars_data(df_raw: pl.DataFrame, dt_minutes: int, checkpoint_dir: str) -> Tuple[pl.DataFrame, List]:
    """
    Step 1: Normalize and clean the raw data.
    
    Args:
        df_raw: Raw polars DataFrame
        dt_minutes: Time granularity in minutes
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (normalized_df, bike_models_list)
    """
    print("Step 1: Normalizing data...")
    print(f"  -> Input shape: {df_raw.shape}")
    print(f"  -> Available columns: {df_raw.columns[:10]}{'...' if len(df_raw.columns) > 10 else ''}")
    
    # show data types for key columns  
    key_cols = ["fecha_origen_recorrido", "id_estacion_origen", "duracion_recorrido"]
    for col in key_cols:
        if col in df_raw.columns:
            dtype = df_raw.select(col).dtypes[0]
            print(f"  -> {col}: {dtype}")
    
    checkpoint_path = os.path.join(checkpoint_dir, "step_01_normalized.parquet")
    
    # check for existing checkpoint
    df_checkpoint = _load_checkpoint_if_exists(checkpoint_path, "normalized data")
    if df_checkpoint is not None:
        # extract bike models from the data
        bike_models = df_checkpoint.select("modelo_bicicleta").unique().to_series().to_list()
        return df_checkpoint, bike_models
    
    # normalize data
    df = df_raw.clone()
    
    # detect if fecha_origen_recorrido is already datetime or string
    fecha_col_dtype = df.select("fecha_origen_recorrido").dtypes[0]
    
    # standardize column names and types
    column_transformations = [
        pl.col("id_estacion_origen").cast(pl.Int32).alias("station_id_origin"),
        pl.col("id_estacion_destino").cast(pl.Int32).alias("station_id_dest"),
        pl.col("duracion_recorrido").cast(pl.Float32).alias("trip_duration"),
        pl.col("id_usuario").cast(pl.Int32).alias("user_id"),
        pl.col("modelo_bicicleta").cast(pl.Utf8).alias("bike_model")
    ]
    
    # handle datetime column based on its current type
    if fecha_col_dtype in [pl.Utf8, pl.String]:
        # string type - needs parsing
        column_transformations.append(
            pl.col("fecha_origen_recorrido").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S").alias("ts_start")
        )
    elif fecha_col_dtype in [pl.Datetime, pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")]:
        # already datetime - normalize to microseconds and rename
        column_transformations.append(
            pl.col("fecha_origen_recorrido").cast(pl.Datetime("us")).alias("ts_start")
        )
    else:
        # try to cast to datetime as fallback - normalize to microseconds
        column_transformations.append(
            pl.col("fecha_origen_recorrido").cast(pl.Datetime("us")).alias("ts_start")
        )
    
    df = df.with_columns(column_transformations)
    
    # round timestamps to specified granularity
    if dt_minutes != 1:
        df = df.with_columns([
            (pl.col("ts_start").dt.truncate(f"{dt_minutes}m")).alias("ts_start")
        ])
    
    # filter out invalid data
    df = df.filter(
        (pl.col("station_id_origin").is_not_null()) &
        (pl.col("station_id_dest").is_not_null()) &
        (pl.col("trip_duration") > 0) &
        (pl.col("trip_duration") < 24 * 60)  # less than 24 hours
    )
    
    # extract bike models
    bike_models = df.select("bike_model").unique().to_series().to_list()
    
    print(f"  -> Normalized data shape: {df.shape}")
    print(f"  -> Found {len(bike_models)} bike models")
    
    # save checkpoint
    _save_checkpoint(df, checkpoint_path, "normalized data", use_streaming=True)
    
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
        # check which column names are available (normalized vs original)
        duration_col = "trip_duration" if "trip_duration" in df.columns else "duracion_recorrido"
        gender_col = "genero" if "genero" in df.columns else None
        model_col = "bike_model" if "bike_model" in df.columns else "modelo_bicicleta"
        station_origin_col = "station_id_origin" if "station_id_origin" in df.columns else "id_estacion_origen" 
        
        aggs = [
            pl.len().alias("dep_last_DT"),
            pl.mean(duration_col).alias("trip_dur_mean_last_DT"),
        ]
        
        # add gender aggregations if available
        if gender_col and gender_col in df.columns:
            aggs.extend([
                (pl.col(gender_col) == "MALE").sum().alias("male_cnt"),
                (pl.col(gender_col) == "FEMALE").sum().alias("female_cnt"),
                (pl.col(gender_col) == "OTHER").sum().alias("other_cnt"),
            ])
        else:
            # add zero counts if no gender data
            aggs.extend([
                pl.lit(0).alias("male_cnt"),
                pl.lit(0).alias("female_cnt"), 
                pl.lit(0).alias("other_cnt"),
            ])
        
        # add bike model aggregations if available
        if model_col in df.columns:
            for model in bike_models:
                if model:
                    aggs.append((pl.col(model_col) == model).sum().alias(f"model_{model}_cnt"))
        pbar.update(1)

        pbar.set_description("Computing departures")
        
        # create time column for grouping - check if ts_start exists or use fecha_origen_recorrido
        if "ts_start" not in df.columns:
            df = df.with_columns(pl.col("fecha_origen_recorrido").alias("ts_start"))
        
        dep = (
            df.group_by([station_origin_col, "ts_start"])
              .agg(aggs)
              .rename({station_origin_col: "station_id"})
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
        
        # check which column names are available (normalized vs original)
        station_dest_col = "station_id_dest" if "station_id_dest" in df.columns else "id_estacion_destino"
        
        # ensure time column for grouping
        if "ts_start" not in df.columns:
            df = df.with_columns(pl.col("fecha_origen_recorrido").alias("ts_start"))
        
        arr = (
            df.group_by([station_dest_col, "ts_start"])
              .agg(pl.len().alias("arr_last_DT"))
              .rename({station_dest_col: "station_id"})
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


def step_05_create_weather_table(df: pl.DataFrame, checkpoint_dir: str) -> Optional[pl.DataFrame]:
    """
    Step 5: Create weather features table if weather data is available.
    
    Args:
        df: Normalized DataFrame
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Weather DataFrame or None if no weather data
    """
    print("Step 5: Creating weather table...")
    
    checkpoint_path = os.path.join(checkpoint_dir, "step_05_weather.parquet")
    
    # check for existing checkpoint
    weather_checkpoint = _load_checkpoint_if_exists(checkpoint_path, "weather table")
    if weather_checkpoint is not None:
        return weather_checkpoint
    
    # check if weather columns exist (both with and without "weather_" prefix)
    weather_columns = [col for col in df.columns if col.startswith(('temperature', 'precipitation', 'wind', 'humidity', 'weather_'))]
    
    if not weather_columns:
        print("  -> No weather data found, skipping weather features")
        return None
    
    # create weather aggregation
    weather = (
        df.group_by("ts_start")
        .agg([
            pl.col(col).first().alias(col) for col in weather_columns
        ])
    )
    
    print(f"  -> Weather table shape: {weather.shape}")
    
    # save checkpoint
    _save_checkpoint(weather, checkpoint_path, "weather table", use_streaming=True)
    
    return weather


def step_06_create_station_metadata(df: pl.DataFrame, checkpoint_dir: str) -> pl.DataFrame:
    """
    Step 6: Create station metadata table.
    
    Args:
        df: Normalized DataFrame
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Station metadata DataFrame
    """
    print("Step 6: Creating station metadata...")
    
    checkpoint_path = os.path.join(checkpoint_dir, "step_06_stations.parquet")
    
    # check for existing checkpoint
    stations_checkpoint = _load_checkpoint_if_exists(checkpoint_path, "station metadata")
    if stations_checkpoint is not None:
        return stations_checkpoint
    
    # check available column names (normalized vs original)
    origin_id_col = "station_id_origin" if "station_id_origin" in df.columns else "id_estacion_origen"
    dest_id_col = "station_id_dest" if "station_id_dest" in df.columns else "id_estacion_destino"
    
    # extract unique stations with coordinates
    stations_origin = df.select([
        origin_id_col,
        "lat_estacion_origen",
        "long_estacion_origen",
        "nombre_estacion_origen"
    ]).unique(origin_id_col).rename({
        origin_id_col: "station_id",
        "lat_estacion_origen": "lat",
        "long_estacion_origen": "lon",
        "nombre_estacion_origen": "station_name"
    })
    
    stations_dest = df.select([
        dest_id_col,
        "lat_estacion_destino", 
        "long_estacion_destino",
        "nombre_estacion_destino"
    ]).unique(dest_id_col).rename({
        dest_id_col: "station_id",
        "lat_estacion_destino": "lat",
        "long_estacion_destino": "lon",
        "nombre_estacion_destino": "station_name"
    })
    
    # combine and deduplicate
    stations = (
        pl.concat([stations_origin, stations_dest])
        .unique("station_id", keep="first")
        .drop_nulls(["lat", "lon"])
        .sort("station_id")
    )
    
    print(f"  -> Station metadata shape: {stations.shape}")
    
    # save checkpoint
    _save_checkpoint(stations, checkpoint_path, "station metadata", use_streaming=True)
    
    return stations


def _build_skeleton(stations_df: pl.DataFrame, start_ts: pl.Series,
                    end_ts: pl.Series, freq: str) -> pl.DataFrame:
    """
    Return a DataFrame with all combinations (station_id, ts_start)
    between start and end inclusive at the specified frequency.
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
        # use ts_start column which should exist after normalization
        time_col = "ts_start" if "ts_start" in df.columns else "fecha_origen_recorrido"
        min_ts = df[time_col].min()
        max_ts = df[time_col].max() + timedelta(minutes=dt_minutes)
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

    # Use the temporal processing function from data/temporal_processing.py
    from data.temporal_processing import create_temporal_features_optimized
    df_feat = create_temporal_features_optimized(df_feat, checkpoint_dir, checkpoint_path, holiday_dates)
    return df_feat


def engineer_ecobici_features_optimized(
    df_raw: pl.DataFrame,
    dt_minutes: int = 30,
    n_neighbors: int = 5,
    clear_checkpoints: bool = False,
    max_memory_mb: int = 3000,  # More conservative limit for 8GB systems
    chunk_size_days: int = 5,   # Smaller chunks to prevent memory overflow
    enable_streaming: bool = True,
    process_in_chunks_from_start: bool = False  # New parameter for ultra-low memory processing
) -> pl.DataFrame:
    """
    Memory-optimized parametrized pipeline for feature-engineering with Polars.
    
    Key optimizations:
    - Processes data in temporal chunks to avoid memory overflow
    - Monitors memory usage and aborts if limits are exceeded
    - Uses streaming operations where possible
    - Aggressive garbage collection
    - Smaller data types to reduce memory footprint
    - Cross-platform compatibility (works on M1, Intel, AMD)
    
    Args:
        df_raw: Original trips + weather data (must be Polars DataFrame)
        dt_minutes: Time window granularity in minutes
        n_neighbors: Number of nearest neighbor stations for demand features
        clear_checkpoints: Whether to clear all checkpoints and start fresh
        max_memory_mb: Maximum memory usage before aborting (MB)
        chunk_size_days: Number of days to process at once
        enable_streaming: Use streaming operations for large datasets
        process_in_chunks_from_start: Whether to process data in chunks from the start

    Returns:
        Final feature dataset
    """
    # validate input type
    if not isinstance(df_raw, pl.DataFrame):
        raise TypeError(
            f"df_raw must be a polars.DataFrame, got {type(df_raw)}. "
            f"Convert your data to Polars using pl.from_pandas() or pl.read_csv() before calling this function."
        )
    
    print("🚴‍♂️  MEMORY-OPTIMIZED Feature Engineering Pipeline")
    print("=" * 60)
    print(f"  Max memory limit: {max_memory_mb} MB")
    print(f"  Chunk size: {chunk_size_days} days")
    print(f"  Streaming enabled: {enable_streaming}")

    # setup checkpointing
    checkpoint_dir = "feature_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if clear_checkpoints:
        clear_feature_checkpoints(checkpoint_dir)
    
    print(f"  Checkpoints directory: '{checkpoint_dir}/'")
    print(f"  Input raw data shape: {df_raw.shape}")
    _log_memory_usage("initialization")

    # check memory before starting
    if _check_memory_and_abort_if_needed("initialization", max_memory_mb):
        raise MemoryError(f"Initial memory usage too high. Consider reducing data size or increasing max_memory_mb.")

    # check for final result first
    final_result_path = os.path.join(checkpoint_dir, "df_feat_final_result_optimized.parquet")
    if os.path.exists(final_result_path):
        print(f"  🎉 Final optimized result already exists! Loading from: {final_result_path}")
        try:
            if enable_streaming:
                df_final = pl.scan_parquet(final_result_path).collect()
            else:
                df_final = pl.read_parquet(final_result_path)
            print(f"  ✅ Successfully loaded final result with shape: {df_final.shape}")
            return df_final
        except Exception as e:
            print(f"  ⚠️ Error loading final result: {e}. Will recreate...")
            os.remove(final_result_path)

    # Check if we need to process in chunks from the start due to memory constraints
    estimated_memory_mb = df_raw.estimated_size('mb')
    if estimated_memory_mb > max_memory_mb * 0.7 or process_in_chunks_from_start:
        print(f"🔄 CHUNKED PROCESSING MODE ACTIVATED")
        print(f"   Dataset size: {estimated_memory_mb:.1f} MB > {max_memory_mb * 0.7:.1f} MB threshold")
        print(f"   Will process data in temporal chunks from the start...")
        
        return _process_in_temporal_chunks_from_start(
            df_raw, dt_minutes, n_neighbors, checkpoint_dir, 
            max_memory_mb, chunk_size_days, enable_streaming
        )
    
    # Continue with normal processing if memory allows
    print(f"📊 NORMAL PROCESSING MODE")
    print(f"   Dataset size: {estimated_memory_mb:.1f} MB fits in memory limit")
    
    # execute pipeline steps with memory monitoring
    print("[Step 1/11] Normalizing Polars data...")
    df, bike_models = step_01_normalize_polars_data(df_raw, dt_minutes, checkpoint_dir)
    _log_memory_usage("after step 1")
    
    # Check memory after step 1 and switch to chunked mode if needed
    current_memory = _get_memory_usage_mb()
    if current_memory > max_memory_mb:
        print(f"⚠️  SWITCHING TO CHUNKED MODE after Step 1")
        print(f"   Memory usage: {current_memory:.1f} MB > {max_memory_mb} MB")
        print(f"   Will continue with chunked processing...")
        
        # Save the normalized data and switch to chunked processing
        temp_norm_path = os.path.join(checkpoint_dir, "temp_normalized_for_chunked.parquet")
        df.write_parquet(temp_norm_path)
        del df, df_raw
        import gc
        gc.collect()
        
        # Load back as lazy frame and continue with chunked processing
        df_norm_lazy = pl.scan_parquet(temp_norm_path)
        return _continue_with_chunked_processing(
            df_norm_lazy, dt_minutes, n_neighbors, checkpoint_dir,
            max_memory_mb, chunk_size_days, enable_streaming, temp_norm_path
        )
    
    # create feature tables with memory optimization
    print("[Step 2/11] Creating departures table with lags & user profiles...")
    departures = step_02_create_departures_table(df, bike_models, checkpoint_dir)
    _log_memory_usage("step 2")
    
    print("[Step 3/11] Creating arrivals table with lags...")
    arrivals = step_03_create_arrivals_table(df, checkpoint_dir)
    _log_memory_usage("step 3")
    
    print("[Step 4/11] Creating shifted arrivals table for target...")
    shifted_arrivals = step_04_create_shifted_arrivals(dt_minutes, checkpoint_dir)
    _log_memory_usage("step 4")
    
    print("[Step 5/11] Creating enhanced weather table...")
    weather = step_05_create_weather_table(df, checkpoint_dir)
    _log_memory_usage("step 5")
    
    print("[Step 6/11] Creating station metadata table...")
    stations = step_06_create_station_metadata(df, checkpoint_dir)
    _log_memory_usage("step 6")
    
    # clear main dataframe from memory after extracting what we need
    del df
    gc.collect()
    _log_memory_usage("after clearing main dataframe")
    
    if _check_memory_and_abort_if_needed("after feature table creation", max_memory_mb):
        raise MemoryError("Memory limit exceeded after feature table creation")
    
    # OPTIMIZED SKELETON BUILDING - Process in temporal chunks
    print("[Step 7/11] Building full station x time skeleton...")
    skeleton_path = os.path.join(checkpoint_dir, "step07_skeleton_optimized.parquet")
    
    if os.path.exists(skeleton_path):
        print(f"  -> Loading skeleton from checkpoint: {skeleton_path}")
        if enable_streaming:
            df_feat = pl.scan_parquet(skeleton_path).collect()
        else:
            df_feat = pl.read_parquet(skeleton_path)
    else:
        df_feat = _build_skeleton_chunked(
            departures, stations, dt_minutes, chunk_size_days, 
            skeleton_path, max_memory_mb, enable_streaming
        )
    
    _log_memory_usage("after skeleton creation")
    
    # OPTIMIZED JOINING - Process in chunks
    print("[Step 8/11] Joining data onto skeleton...")
    joined_path = os.path.join(checkpoint_dir, "step08_joined_optimized.parquet")
    
    if os.path.exists(joined_path):
        print(f"  -> Loading joined features from checkpoint: {joined_path}")
        if enable_streaming:
            df_feat = pl.scan_parquet(joined_path).collect()
        else:
            df_feat = pl.read_parquet(joined_path)
    else:
        df_feat = _join_features_chunked(
            df_feat, departures, arrivals, shifted_arrivals, weather, stations,
            joined_path, chunk_size_days, max_memory_mb, enable_streaming
        )
    
    _log_memory_usage("after joining")
    
    # OPTIMIZED TEMPORAL FEATURES
    print("[Step 9/11] Engineering temporal and calendar features...")
    temporal_path = os.path.join(checkpoint_dir, "step09_temporal_optimized.parquet")
    
    if os.path.exists(temporal_path):
        print(f"  -> Loading temporal features from checkpoint: {temporal_path}")
        if enable_streaming:
            df_feat = pl.scan_parquet(temporal_path).collect()
        else:
            df_feat = pl.read_parquet(temporal_path)
    else:
        from data.temporal_processing import create_temporal_features_optimized
        df_feat = create_temporal_features_optimized(df_feat, checkpoint_dir, temporal_path, holiday_dates=[])
    
    _log_memory_usage("after temporal features")
    
    print(f"[Step 10/11] Final data cleaning and optimization...")
    # Optimize data types to save memory
    df_feat = _optimize_data_types(df_feat)
    _log_memory_usage("after data type optimization")
    
    print(f"[Step 11/11] Saving final result...")
    print(f"  -> Final feature dataset shape: {df_feat.shape}")
    
    # Check dataset size and use appropriate saving method
    final_size_mb = df_feat.estimated_size('mb')
    print(f"  -> Final dataset size: {final_size_mb:.1f} MB")
    
    try:
        if final_size_mb > 500:  # Use chunked writing for very large datasets
            print(f"  -> Large dataset detected, using chunked writing...")
            _write_parquet_in_chunks(df_feat, final_result_path, chunk_rows=50000)
        elif enable_streaming:
            print(f"  -> Using streaming mode...")
            df_feat.lazy().sink_parquet(
                final_result_path, 
                compression="zstd", 
                compression_level=3,
                row_group_size=10000
            )
        else:
            print(f"  -> Using standard writing...")
            df_feat.write_parquet(
                final_result_path,
                compression="zstd",
                compression_level=3,
                row_group_size=10000
            )
        print(f"  ✅ Saved optimized final result: {final_result_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save final result: {e}")
        # Try fallback method
        try:
            print(f"  -> Trying fallback chunked writing...")
            _write_parquet_in_chunks(df_feat, final_result_path, chunk_rows=25000)
            print(f"  ✅ Saved using fallback method: {final_result_path}")
        except Exception as e2:
            print(f"  ❌ Fallback also failed: {e2}")
    
    _log_memory_usage("final")
    print("🎉 Memory-optimized pipeline completed successfully!")
    
    return df_feat


def _build_skeleton_chunked(
    departures: pl.DataFrame, 
    stations: pl.DataFrame, 
    dt_minutes: int,
    chunk_size_days: int,
    skeleton_path: str,
    max_memory_mb: int,
    enable_streaming: bool
) -> pl.DataFrame:
    """Build the time x station skeleton in temporal chunks to avoid memory issues."""
    
    # get time range
    time_range = departures.select([
        pl.col("ts_start").min().alias("start"),
        pl.col("ts_start").max().alias("end")
    ]).row(0)
    
    start_time = time_range[0]
    end_time = time_range[1]
    
    print(f"  -> Building skeleton from {start_time} to {end_time}")
    print(f"  -> Processing in {chunk_size_days}-day chunks...")
    
    # create chunks directory
    chunks_dir = os.path.join(os.path.dirname(skeleton_path), "skeleton_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    chunk_files = []
    current_time = start_time
    chunk_num = 0
    
    while current_time < end_time:
        chunk_end = min(current_time + timedelta(days=chunk_size_days), end_time)
        chunk_file = os.path.join(chunks_dir, f"skeleton_chunk_{chunk_num:04d}.parquet")
        chunk_files.append(chunk_file)
        
        print(f"  -> Processing chunk {chunk_num}: {current_time} to {chunk_end}")
        
        # create time range for this chunk - ensuring consistent datetime precision
        time_chunk = pl.datetime_range(
            start=current_time,
            end=chunk_end,
            interval=f"{dt_minutes}m",
            eager=True,
            time_unit="us"  # added to match _build_skeleton precision
        ).alias("ts_start").to_frame()
        
        # cross join with stations (smaller operation per chunk)
        chunk_skeleton = (
            time_chunk
            .join(stations.select("station_id"), how="cross")
            .sort(["ts_start", "station_id"])
        )
        
        # save chunk immediately
        if enable_streaming:
            chunk_skeleton.lazy().sink_parquet(
                chunk_file, 
                compression="snappy",
                row_group_size=5000
            )
        else:
            chunk_skeleton.write_parquet(chunk_file, compression="snappy")
        
        current_time = chunk_end
        chunk_num += 1
        
        # memory check
        if _check_memory_and_abort_if_needed(f"skeleton chunk {chunk_num}", max_memory_mb):
            # cleanup and abort
            for f in chunk_files:
                try:
                    os.remove(f)
                except:
                    pass
            raise MemoryError(f"Memory limit exceeded during skeleton building at chunk {chunk_num}")
        
        gc.collect()
    
    print(f"  -> Combining {len(chunk_files)} skeleton chunks using iterative approach...")
    
    if len(chunk_files) <= 3:
        # Very small number of skeleton chunks - can combine directly
        print(f"  -> Direct combination of {len(chunk_files)} skeleton chunks...")
        combined_chunks = []
        for chunk_file in chunk_files:
            chunk_df = pl.read_parquet(chunk_file)
            combined_chunks.append(chunk_df)
        
        result_df = pl.concat(combined_chunks, how="vertical")
        
        # cleanup chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
            
        return result_df
    
    # Large number of skeleton chunks - use iterative combination (2-3 at a time)
    print(f"  -> Using iterative combination for {len(chunk_files)} skeleton chunks...")
    print(f"  -> Will combine 2-3 files at a time to minimize memory usage...")
    
    # Start with the first file as base
    current_files = chunk_files.copy()
    iteration = 0
    
    while len(current_files) > 1:
        iteration += 1
        print(f"  -> Iteration {iteration}: {len(current_files)} files remaining...")
        
        new_files = []
        files_processed = 0
        
        # Process files in groups of 2-3
        i = 0
        while i < len(current_files):
            # Take 2-3 files at a time
            batch_size = min(3, len(current_files) - i)
            batch_files = current_files[i:i + batch_size]
            
            if len(batch_files) == 1:
                # Single file left - just add to new_files
                new_files.append(batch_files[0])
            else:
                # Combine 2-3 files
                temp_file = os.path.join(chunks_dir, f"skel_iter_{iteration}_{i//batch_size}.parquet")
                
                print(f"    -> Combining {len(batch_files)} files into {os.path.basename(temp_file)}")
                
                # Load only these 2-3 files
                batch_dfs = []
                for file_path in batch_files:
                    batch_dfs.append(pl.read_parquet(file_path))
                
                # Combine and save immediately
                combined = pl.concat(batch_dfs, how="vertical")
                combined.write_parquet(temp_file)
                new_files.append(temp_file)
                
                # Clean up immediately
                del batch_dfs, combined
                for file_path in batch_files:
                    if file_path != temp_file:  # Don't delete the file we just created
                        try:
                            os.remove(file_path)
                        except:
                            pass
                
                files_processed += len(batch_files)
                
                # Force garbage collection after each batch
                gc.collect()
            
            i += batch_size
        
        current_files = new_files
        print(f"    -> Processed {files_processed} files, {len(current_files)} files remain")
    
    # Should have exactly one file left
    if len(current_files) == 1:
        result_df = pl.read_parquet(current_files[0])
        os.remove(current_files[0])  # Clean up the final temp file
    else:
        raise RuntimeError(f"Expected 1 final file, got {len(current_files)}")
    
    print(f"  -> Successfully combined all skeleton chunks iteratively. Result shape: {result_df.shape}")
    return result_df


def _join_features_chunked(
    df_skeleton: pl.DataFrame,
    departures: pl.DataFrame,
    arrivals: pl.DataFrame, 
    shifted_arrivals: pl.DataFrame,
    weather: Optional[pl.DataFrame],
    stations: pl.DataFrame,
    joined_path: str,
    chunk_size_days: int,
    max_memory_mb: int,
    enable_streaming: bool
) -> pl.DataFrame:
    """Join all feature tables to skeleton in temporal chunks."""
    
    # get time range
    time_range = df_skeleton.select([
        pl.col("ts_start").min().alias("start"),
        pl.col("ts_start").max().alias("end")
    ]).row(0)
    
    start_time = time_range[0]
    end_time = time_range[1]
    
    print(f"  -> Joining features in {chunk_size_days}-day chunks...")
    
    # create chunks directory
    chunks_dir = os.path.join(os.path.dirname(joined_path), "joined_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    chunk_files = []
    current_time = start_time
    chunk_num = 0
    
    while current_time < end_time:
        chunk_end = min(current_time + timedelta(days=chunk_size_days), end_time)
        chunk_file = os.path.join(chunks_dir, f"joined_chunk_{chunk_num:04d}.parquet")
        chunk_files.append(chunk_file)
        
        print(f"  -> Processing chunk {chunk_num}: {current_time} to {chunk_end}")
        
        # get skeleton chunk
        skeleton_chunk = df_skeleton.filter(
            (pl.col("ts_start") >= current_time) & 
            (pl.col("ts_start") < chunk_end)
        )
        
        if skeleton_chunk.is_empty():
            current_time = chunk_end
            chunk_num += 1
            continue
        
        # join departures
        chunk_feat = skeleton_chunk.join(
            departures.filter(
                (pl.col("ts_start") >= current_time) & 
                (pl.col("ts_start") < chunk_end)
            ),
            on=["station_id", "ts_start"],
            how="left"
        )
        
        # join arrivals
        chunk_feat = chunk_feat.join(
            arrivals.filter(
                (pl.col("ts_start") >= current_time) & 
                (pl.col("ts_start") < chunk_end)
            ),
            on=["station_id", "ts_start"],
            how="left"
        )
        
        # join shifted arrivals (target)
        chunk_feat = chunk_feat.join(
            shifted_arrivals.filter(
                (pl.col("ts_start") >= current_time) & 
                (pl.col("ts_start") < chunk_end)
            ),
            on=["station_id", "ts_start"],
            how="left"
        )
        
        # join weather if available
        if weather is not None:
            chunk_feat = chunk_feat.join(
                weather.filter(
                    (pl.col("ts_start") >= current_time) & 
                    (pl.col("ts_start") < chunk_end)
                ),
                on="ts_start",
                how="left"
            )
        
        # join station metadata
        chunk_feat = chunk_feat.join(
            stations,
            on="station_id",
            how="left"
        )
        
        # fill nulls with appropriate values
        chunk_feat = chunk_feat.with_columns([
            pl.col("dep_last_DT").fill_null(0),
            pl.col("arr_last_DT").fill_null(0),
            pl.col("y_arrivals_next_DT").fill_null(0),
            pl.col("trip_dur_mean_last_DT").fill_null(0.0)
        ])
        
        # save chunk
        if enable_streaming:
            chunk_feat.lazy().sink_parquet(
                chunk_file,
                compression="snappy",
                row_group_size=5000
            )
        else:
            chunk_feat.write_parquet(chunk_file, compression="snappy")
        
        current_time = chunk_end
        chunk_num += 1
        
        # memory check
        if _check_memory_and_abort_if_needed(f"join chunk {chunk_num}", max_memory_mb):
            # cleanup and abort
            for f in chunk_files:
                try:
                    os.remove(f)
                except:
                    pass
            raise MemoryError(f"Memory limit exceeded during feature joining at chunk {chunk_num}")
        
        gc.collect()
    
    print(f"  -> Combining {len(chunk_files)} joined chunks using iterative approach...")
    
    if len(chunk_files) <= 3:
        # Very small number of chunks - can combine directly
        print(f"  -> Direct combination of {len(chunk_files)} chunks...")
        combined_chunks = []
        for chunk_file in chunk_files:
            chunk_df = pl.read_parquet(chunk_file)
            combined_chunks.append(chunk_df)
        
        result_df = pl.concat(combined_chunks, how="vertical")
        
        # cleanup chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
            
        return result_df
    
    # Large number of chunks - use iterative combination (2-3 at a time)
    print(f"  -> Using iterative combination for {len(chunk_files)} chunks...")
    print(f"  -> Will combine 2-3 files at a time to minimize memory usage...")
    
    # Start with all files
    current_files = chunk_files.copy()
    iteration = 0
    
    while len(current_files) > 1:
        iteration += 1
        print(f"  -> Iteration {iteration}: {len(current_files)} files remaining...")
        
        new_files = []
        files_processed = 0
        
        # Process files in groups of 2-3 (more conservative for feature chunks)
        i = 0
        while i < len(current_files):
            # Take 2-3 files at a time (2 for feature chunks to be extra safe)
            batch_size = min(2, len(current_files) - i)
            batch_files = current_files[i:i + batch_size]
            
            if len(batch_files) == 1:
                # Single file left - just add to new_files
                new_files.append(batch_files[0])
            else:
                # Combine 2 files
                temp_file = os.path.join(chunks_dir, f"feat_iter_{iteration}_{i//batch_size}.parquet")
                
                print(f"    -> Combining {len(batch_files)} files into {os.path.basename(temp_file)}")
                
                # Load only these 2 files
                batch_dfs = []
                for file_path in batch_files:
                    batch_dfs.append(pl.read_parquet(file_path))
                
                # Combine and save immediately
                combined = pl.concat(batch_dfs, how="vertical")
                combined.write_parquet(temp_file)
                new_files.append(temp_file)
                
                # Clean up immediately
                del batch_dfs, combined
                for file_path in batch_files:
                    if file_path != temp_file:  # Don't delete the file we just created
                        try:
                            os.remove(file_path)
                        except:
                            pass
                
                files_processed += len(batch_files)
                
                # Force garbage collection after each batch
                gc.collect()
            
            i += batch_size
        
        current_files = new_files
        print(f"    -> Processed {files_processed} files, {len(current_files)} files remain")
        
        # Extra safety: if we're not making progress, break
        if files_processed == 0 and len(current_files) > 1:
            print(f"    -> Warning: No progress made, forcing final combination...")
            break
    
    # Should have exactly one file left
    if len(current_files) == 1:
        result_df = pl.read_parquet(current_files[0])
        os.remove(current_files[0])  # Clean up the final temp file
    elif len(current_files) <= 3:
        # Final fallback: combine remaining files directly
        print(f"  -> Final combination of {len(current_files)} remaining files...")
        final_dfs = []
        for f in current_files:
            final_dfs.append(pl.read_parquet(f))
        result_df = pl.concat(final_dfs, how="vertical")
        for f in current_files:
            os.remove(f)
        del final_dfs
    else:
        raise RuntimeError(f"Too many files remaining: {len(current_files)}")
    
    print(f"  -> Successfully combined all chunks iteratively. Result shape: {result_df.shape}")
    return result_df


def _optimize_data_types(df: pl.DataFrame) -> pl.DataFrame:
    """Optimize data types to reduce memory usage."""
    
    print("  -> Optimizing data types for memory efficiency...")
    
    # Get column info
    schema = df.schema
    optimizations = []
    
    for col_name, dtype in schema.items():
        if dtype == pl.Int64:
            # Check if we can downcast to smaller int types
            col_min = df.select(pl.col(col_name).min()).item()
            col_max = df.select(pl.col(col_name).max()).item()
            
            if col_min is not None and col_max is not None:
                if col_min >= 0 and col_max <= 255:
                    optimizations.append(pl.col(col_name).cast(pl.UInt8))
                elif col_min >= -128 and col_max <= 127:
                    optimizations.append(pl.col(col_name).cast(pl.Int8))
                elif col_min >= 0 and col_max <= 65535:
                    optimizations.append(pl.col(col_name).cast(pl.UInt16))
                elif col_min >= -32768 and col_max <= 32767:
                    optimizations.append(pl.col(col_name).cast(pl.Int16))
                elif col_min >= 0 and col_max <= 4294967295:
                    optimizations.append(pl.col(col_name).cast(pl.UInt32))
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimizations.append(pl.col(col_name).cast(pl.Int32))
        
        elif dtype == pl.Float64:
            # Downcast to Float32 if precision allows
            optimizations.append(pl.col(col_name).cast(pl.Float32))
    
    if optimizations:
        df = df.with_columns(optimizations)
        print(f"  -> Optimized {len(optimizations)} columns")
    
    return df


# Keep the original function for backward compatibility
def engineer_ecobici_features(
    df_raw: pl.DataFrame,
    dt_minutes: int = 30,
    n_neighbors: int = 5,
    clear_checkpoints: bool = False,
    rolling_chunk_size: int = 50
) -> pl.DataFrame:
    """
    Original function - now calls the optimized version with safe defaults.
    
    This wrapper maintains backward compatibility while using the new optimized pipeline.
    """
    print("⚠️  Using backward compatibility wrapper - calling optimized version...")
    
    # Detect available memory and set conservative limits
    try:
        import psutil
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        max_memory_mb = min(6000, int(available_memory_mb * 0.7))  # Use 70% of available memory, max 6GB
        print(f"  -> Detected {available_memory_mb:.0f} MB available memory")
        print(f"  -> Setting memory limit to {max_memory_mb} MB")
    except:
        max_memory_mb = 4000  # Conservative default
        print(f"  -> Could not detect memory, using conservative limit: {max_memory_mb} MB")
    
    return engineer_ecobici_features_optimized(
        df_raw=df_raw,
        dt_minutes=dt_minutes,
        n_neighbors=n_neighbors,
        clear_checkpoints=clear_checkpoints,
        max_memory_mb=max_memory_mb,
        chunk_size_days=15,  # Conservative chunk size
        enable_streaming=True
    )


def _check_memory_and_abort_if_needed(operation_name: str, max_memory_mb: int = 8000) -> bool:
    """Check if memory usage is too high and recommend aborting to prevent kernel crash."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > max_memory_mb:
            print(f"⚠️  HIGH MEMORY WARNING: {memory_mb:.1f} MB > {max_memory_mb} MB")
            print(f"   Operation: {operation_name}")
            print(f"   Consider reducing rolling_chunk_size or clearing memory")
            return True
        return False
    except:
        return False


def _process_in_temporal_chunks_from_start(
    df_raw: pl.DataFrame,
    dt_minutes: int,
    n_neighbors: int,
    checkpoint_dir: str,
    max_memory_mb: int,
    chunk_size_days: int,
    enable_streaming: bool
) -> pl.DataFrame:
    """Process the entire pipeline in temporal chunks from the very beginning."""
    
    print(f"🔄 PROCESSING IN TEMPORAL CHUNKS FROM START")
    print(f"   Chunk size: {chunk_size_days} days")
    print(f"   Memory limit: {max_memory_mb} MB")
    
    # Get time range from raw data
    time_range = df_raw.select([
        pl.col("fecha_origen_recorrido").min().alias("start"),
        pl.col("fecha_origen_recorrido").max().alias("end")
    ]).row(0)
    
    start_time = time_range[0]
    end_time = time_range[1]
    
    print(f"   Time range: {start_time} to {end_time}")
    
    # Create temporal chunks
    chunk_start = start_time
    chunk_results = []
    chunk_idx = 0
    
    while chunk_start < end_time:
        chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end_time)
        
        print(f"   Processing chunk {chunk_idx}: {chunk_start} to {chunk_end}")
        
        # Filter data for this time chunk
        chunk_data = df_raw.filter(
            pl.col("fecha_origen_recorrido").is_between(chunk_start, chunk_end, closed="left")
        )
        
        if chunk_data.height == 0:
            print(f"     -> Empty chunk, skipping...")
            chunk_start = chunk_end
            continue
        
        print(f"     -> Chunk size: {chunk_data.height:,} rows")
        
        # Process this chunk through the entire pipeline
        try:
            chunk_result = _process_single_chunk_full_pipeline(
                chunk_data, dt_minutes, n_neighbors, checkpoint_dir, 
                max_memory_mb, enable_streaming, chunk_idx
            )
            
            # Save chunk result
            chunk_file = os.path.join(checkpoint_dir, f"final_chunk_{chunk_idx}.parquet")
            chunk_result.write_parquet(chunk_file)
            chunk_results.append(chunk_file)
            
            print(f"     -> Saved chunk result: {chunk_result.shape}")
            
            # Clean up memory
            del chunk_data, chunk_result
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"     -> Error processing chunk {chunk_idx}: {e}")
            # Continue with next chunk
        
        chunk_start = chunk_end
        chunk_idx += 1
    
    # Combine all chunk results
    print(f"🔗 COMBINING {len(chunk_results)} FINAL CHUNKS...")
    return _combine_final_chunks_iteratively(chunk_results, checkpoint_dir)


def _continue_with_chunked_processing(
    df_norm_lazy: pl.LazyFrame,
    dt_minutes: int,
    n_neighbors: int,
    checkpoint_dir: str,
    max_memory_mb: int,
    chunk_size_days: int,
    enable_streaming: bool,
    temp_norm_path: str
) -> pl.DataFrame:
    """Continue processing with chunked approach after step 1."""
    
    print(f"🔄 CONTINUING WITH CHUNKED PROCESSING")
    
    # Get time range from normalized data
    time_range = df_norm_lazy.select([
        pl.col("ts_start").min().alias("start"),
        pl.col("ts_start").max().alias("end")
    ]).collect().row(0)
    
    start_time = time_range[0]
    end_time = time_range[1]
    
    print(f"   Time range: {start_time} to {end_time}")
    
    # Process in temporal chunks
    chunk_start = start_time
    chunk_results = []
    chunk_idx = 0
    
    while chunk_start < end_time:
        chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end_time)
        
        print(f"   Processing normalized chunk {chunk_idx}: {chunk_start} to {chunk_end}")
        
        # Filter normalized data for this time chunk
        chunk_data = df_norm_lazy.filter(
            pl.col("ts_start").is_between(chunk_start, chunk_end, closed="left")
        ).collect()
        
        if chunk_data.height == 0:
            print(f"     -> Empty chunk, skipping...")
            chunk_start = chunk_end
            continue
        
        print(f"     -> Chunk size: {chunk_data.height:,} rows")
        
        # Process this chunk through steps 2-11
        try:
            chunk_result = _process_normalized_chunk_steps_2_to_11(
                chunk_data, dt_minutes, n_neighbors, checkpoint_dir,
                max_memory_mb, enable_streaming, chunk_idx
            )
            
            # Save chunk result
            chunk_file = os.path.join(checkpoint_dir, f"final_chunk_{chunk_idx}.parquet")
            chunk_result.write_parquet(chunk_file)
            chunk_results.append(chunk_file)
            
            print(f"     -> Saved chunk result: {chunk_result.shape}")
            
            # Clean up memory
            del chunk_data, chunk_result
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"     -> Error processing chunk {chunk_idx}: {e}")
        
        chunk_start = chunk_end
        chunk_idx += 1
    
    # Clean up temp file
    try:
        os.remove(temp_norm_path)
    except:
        pass
    
    # Combine all chunk results
    print(f"🔗 COMBINING {len(chunk_results)} FINAL CHUNKS...")
    return _combine_final_chunks_iteratively(chunk_results, checkpoint_dir)


def _combine_final_chunks_iteratively(chunk_files: list, checkpoint_dir: str) -> pl.DataFrame:
    """Combine final chunk results using streaming and chunked writing."""
    
    if len(chunk_files) == 0:
        raise ValueError("No chunk files to combine")
    
    if len(chunk_files) <= 2:
        print(f"   -> Direct streaming combination of {len(chunk_files)} final chunks...")
        return _combine_chunks_with_streaming(chunk_files, checkpoint_dir)
    
    print(f"   -> Iterative streaming combination of {len(chunk_files)} final chunks...")
    
    current_files = chunk_files.copy()
    iteration = 0
    
    while len(current_files) > 1:
        iteration += 1
        print(f"     -> Iteration {iteration}: {len(current_files)} files remaining...")
        
        new_files = []
        i = 0
        
        while i < len(current_files):
            batch_size = min(2, len(current_files) - i)
            batch_files = current_files[i:i + batch_size]
            
            if len(batch_files) == 1:
                new_files.append(batch_files[0])
            else:
                temp_file = os.path.join(checkpoint_dir, f"final_iter_{iteration}_{i//batch_size}.parquet")
                
                print(f"       -> Combining batch {i//batch_size + 1} with streaming...")
                _combine_two_files_streaming(batch_files[0], batch_files[1], temp_file)
                new_files.append(temp_file)
                
                # Clean up original files
                for f in batch_files:
                    if f != temp_file and os.path.exists(f):
                        os.remove(f)
                
                import gc
                gc.collect()
            
            i += batch_size
        
        current_files = new_files
    
    # Read final result with streaming
    print(f"   -> Loading final combined result...")
    result_df = pl.read_parquet(current_files[0])
    os.remove(current_files[0])
    
    return result_df


def _combine_chunks_with_streaming(chunk_files: list, checkpoint_dir: str) -> pl.DataFrame:
    """Combine small number of chunks using streaming approach."""
    if len(chunk_files) == 1:
        result_df = pl.read_parquet(chunk_files[0])
        os.remove(chunk_files[0])
        return result_df
    
    # Use streaming for 2 files
    temp_combined = os.path.join(checkpoint_dir, "temp_streaming_combined.parquet")
    _combine_two_files_streaming(chunk_files[0], chunk_files[1], temp_combined)
    
    # Clean up original files
    for f in chunk_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Load result
    result_df = pl.read_parquet(temp_combined)
    os.remove(temp_combined)
    
    return result_df


def _combine_two_files_streaming(file1: str, file2: str, output_file: str):
    """Combine two parquet files using streaming to minimize memory usage."""
    try:
        # Try streaming approach first
        lazy1 = pl.scan_parquet(file1)
        lazy2 = pl.scan_parquet(file2)
        
        combined_lazy = pl.concat([lazy1, lazy2], how="vertical")
        combined_lazy.sink_parquet(output_file, compression="snappy", row_group_size=10000)
        
        print(f"       -> Successfully combined using streaming")
        
    except Exception as e:
        print(f"       -> Streaming failed ({e}), using chunked approach...")
        _combine_two_files_chunked(file1, file2, output_file)


def _combine_two_files_chunked(file1: str, file2: str, output_file: str, chunk_size: int = 25000):
    """Combine two parquet files by reading and writing in chunks."""
    
    # Start with the first file
    df1 = pl.read_parquet(file1)
    df1_size = df1.estimated_size('mb')
    
    if df1_size > 200:  # If first file is large, write it in chunks
        print(f"       -> First file is large ({df1_size:.1f}MB), writing in chunks...")
        _write_parquet_in_chunks(df1, output_file, chunk_rows=chunk_size)
    else:
        df1.write_parquet(output_file, compression="snappy", row_group_size=10000)
    
    del df1
    import gc
    gc.collect()
    
    # Now append the second file in chunks
    df2 = pl.read_parquet(file2)
    df2_size = df2.estimated_size('mb')
    
    if df2_size > 200:  # If second file is large, append it in chunks
        print(f"       -> Second file is large ({df2_size:.1f}MB), appending in chunks...")
        total_rows = df2.height
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df2.slice(start_idx, end_idx - start_idx)
            
            _append_parquet_chunk(output_file, chunk)
            del chunk
            gc.collect()
            
            print(f"       -> Appended chunk {i+1}/{num_chunks}")
    else:
        _append_parquet_chunk(output_file, df2)
    
    del df2
    gc.collect() 


def _process_single_chunk_full_pipeline(
    chunk_data: pl.DataFrame,
    dt_minutes: int,
    n_neighbors: int,
    checkpoint_dir: str,
    max_memory_mb: int,
    enable_streaming: bool,
    chunk_idx: int
) -> pl.DataFrame:
    """Process a single temporal chunk through the entire feature engineering pipeline."""
    
    print(f"     -> Processing chunk {chunk_idx} through full pipeline...")
    
    # Step 1: Normalize (simplified for chunk)
    df_norm, bike_models = step_01_normalize_polars_data(chunk_data, dt_minutes, checkpoint_dir)
    del chunk_data
    import gc
    gc.collect()
    
    if df_norm.height == 0:
        print(f"     -> Empty after normalization, returning empty result")
        return pl.DataFrame()
    
    # Simplified feature engineering for chunk
    # Create basic departures and arrivals tables
    departures = step_02_create_departures_table(df_norm, bike_models, checkpoint_dir)
    arrivals = step_03_create_arrivals_table(df_norm, checkpoint_dir)
    shifted_arrivals = step_04_create_shifted_arrivals(dt_minutes, checkpoint_dir)
    
    # Create simplified weather and stations (if they have data)
    weather = step_05_create_weather_table(df_norm, checkpoint_dir)
    stations = step_06_create_station_metadata(df_norm, checkpoint_dir)
    
    del df_norm
    gc.collect()
    
    # Create a simplified skeleton for this chunk
    time_range = departures.select([
        pl.col("ts_start").min().alias("start"),
        pl.col("ts_start").max().alias("end")
    ]).row(0)
    
    if time_range[0] is None or time_range[1] is None:
        print(f"     -> No valid time range, returning empty result")
        return pl.DataFrame()
    
    # Build skeleton for this chunk only
    from datetime import timedelta
    skeleton = _build_skeleton(stations, time_range[0], time_range[1] + timedelta(minutes=dt_minutes), f"{dt_minutes}m")
    
    # Join features for this chunk
    df_feat = skeleton.join(departures, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(arrivals, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(shifted_arrivals, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(weather, on="ts_start", how="left")
    
    # Fill nulls
    df_feat = df_feat.with_columns([
        pl.col("dep_last_DT").fill_null(0),
        pl.col("arr_last_DT").fill_null(0),
        pl.col("y_arrivals_next_DT").fill_null(0),
        pl.col("trip_dur_mean_last_DT").fill_null(0.0)
    ])
    
    # Add basic temporal features (simplified)
    df_feat = df_feat.with_columns([
        pl.col("ts_start").dt.hour().sin().alias("sin_hour"),
        pl.col("ts_start").dt.hour().cos().alias("cos_hour"),
        pl.col("ts_start").dt.weekday().sin().alias("sin_dow"),
        pl.col("ts_start").dt.weekday().cos().alias("cos_dow"),
        pl.col("ts_start").dt.month().sin().alias("sin_month"),
        pl.col("ts_start").dt.month().cos().alias("cos_month"),
        (pl.col("ts_start").dt.weekday() >= 6).alias("is_weekend")
    ])
    
    print(f"     -> Chunk {chunk_idx} processed: {df_feat.shape}")
    return df_feat


def _process_normalized_chunk_steps_2_to_11(
    chunk_data: pl.DataFrame,
    dt_minutes: int,
    n_neighbors: int,
    checkpoint_dir: str,
    max_memory_mb: int,
    enable_streaming: bool,
    chunk_idx: int
) -> pl.DataFrame:
    """Process a normalized chunk through steps 2-11 of the pipeline."""
    
    print(f"     -> Processing normalized chunk {chunk_idx} through steps 2-11...")
    
    # Similar to above but starting from normalized data
    # Extract bike models first
    bike_models = chunk_data.select("bike_model").unique().to_series().to_list() if "bike_model" in chunk_data.columns else []
    
    departures = step_02_create_departures_table(chunk_data, bike_models, checkpoint_dir)
    arrivals = step_03_create_arrivals_table(chunk_data, checkpoint_dir)
    shifted_arrivals = step_04_create_shifted_arrivals(dt_minutes, checkpoint_dir)
    
    weather = step_05_create_weather_table(chunk_data, checkpoint_dir)
    stations = step_06_create_station_metadata(chunk_data, checkpoint_dir)
    
    del chunk_data
    import gc
    gc.collect()
    
    # Build skeleton and join features
    time_range = departures.select([
        pl.col("ts_start").min(),
        pl.col("ts_start").max()
    ]).row(0)
    from datetime import timedelta
    skeleton = _build_skeleton(stations, time_range[0], time_range[1] + timedelta(minutes=dt_minutes), f"{dt_minutes}m")
    
    df_feat = skeleton.join(departures, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(arrivals, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(shifted_arrivals, on=["station_id", "ts_start"], how="left")
    df_feat = df_feat.join(weather, on="ts_start", how="left")
    
    # Fill nulls and add basic features
    df_feat = df_feat.with_columns([
        pl.col("dep_last_DT").fill_null(0),
        pl.col("arr_last_DT").fill_null(0),
        pl.col("y_arrivals_next_DT").fill_null(0),
        pl.col("trip_dur_mean_last_DT").fill_null(0.0)
    ])
    
    # Add temporal features
    df_feat = df_feat.with_columns([
        pl.col("ts_start").dt.hour().sin().alias("sin_hour"),
        pl.col("ts_start").dt.hour().cos().alias("cos_hour"),
        pl.col("ts_start").dt.weekday().sin().alias("sin_dow"),
        pl.col("ts_start").dt.weekday().cos().alias("cos_dow"),
        pl.col("ts_start").dt.month().sin().alias("sin_month"),
        pl.col("ts_start").dt.month().cos().alias("cos_month"),
        (pl.col("ts_start").dt.weekday() >= 6).alias("is_weekend")
    ])
    
    print(f"     -> Normalized chunk {chunk_idx} processed: {df_feat.shape}")
    return df_feat 