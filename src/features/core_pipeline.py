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
    """Save checkpoint with proper error handling."""
    try:
        if use_streaming:
            df.lazy().sink_parquet(path, compression="snappy", row_group_size=10000)
        else:
            df.write_parquet(path, compression="snappy")
        print(f"  -> Saved {description} checkpoint: {path}")
    except Exception as e:
        print(f"  -> Warning: Could not save checkpoint {path}: {e}")


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
    
    checkpoint_path = os.path.join(checkpoint_dir, "step_01_normalized.parquet")
    
    # check for existing checkpoint
    df_checkpoint = _load_checkpoint_if_exists(checkpoint_path, "normalized data")
    if df_checkpoint is not None:
        # extract bike models from the data
        bike_models = df_checkpoint.select("modelo_bicicleta").unique().to_series().to_list()
        return df_checkpoint, bike_models
    
    # normalize data
    df = df_raw.clone()
    
    # standardize column names and types
    df = df.with_columns([
        pl.col("fecha_origen_recorrido").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("ts_start"),
        pl.col("id_estacion_origen").cast(pl.Int32).alias("station_id_origin"),
        pl.col("id_estacion_destino").cast(pl.Int32).alias("station_id_dest"),
        pl.col("duracion_recorrido").cast(pl.Float32).alias("trip_duration"),
        pl.col("id_usuario").cast(pl.Int32).alias("user_id"),
        pl.col("modelo_bicicleta").cast(pl.Utf8).alias("bike_model")
    ])
    
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
    
    # check if weather columns exist
    weather_columns = [col for col in df.columns if col.startswith(('temperature', 'precipitation', 'wind', 'humidity'))]
    
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
    
    # extract unique stations with coordinates
    stations_origin = df.select([
        "station_id_origin",
        "lat_estacion_origen",
        "long_estacion_origen",
        "nombre_estacion_origen"
    ]).unique("station_id_origin").rename({
        "station_id_origin": "station_id",
        "lat_estacion_origen": "lat",
        "long_estacion_origen": "lon",
        "nombre_estacion_origen": "station_name"
    })
    
    stations_dest = df.select([
        "station_id_dest",
        "lat_estacion_destino", 
        "long_estacion_destino",
        "nombre_estacion_destino"
    ]).unique("station_id_dest").rename({
        "station_id_dest": "station_id",
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

    # Use the temporal processing function from data/temporal_processing.py
    from ..data.temporal_processing import create_temporal_features_optimized
    df_feat = create_temporal_features_optimized(df_feat, checkpoint_dir, checkpoint_path, holiday_dates)
    return df_feat


def engineer_ecobici_features(
    df_raw: pl.DataFrame,
    dt_minutes: int = 30,
    n_neighbors: int = 5,
    clear_checkpoints: bool = False,
    rolling_chunk_size: int = 50
) -> pl.DataFrame:
    """
    Parametrized pipeline for feature-engineering optimized with Polars.
    
    Args:
        df_raw: Original trips + weather data (must be Polars DataFrame)
        dt_minutes: Time window granularity in minutes
        n_neighbors: Number of nearest neighbor stations for demand features
        clear_checkpoints: Whether to clear all checkpoints and start fresh
        rolling_chunk_size: Number of stations to process at once to avoid memory issues

    Returns:
        Final feature dataset
    """
    # validate input type
    if not isinstance(df_raw, pl.DataFrame):
        raise TypeError(
            f"df_raw must be a polars.DataFrame, got {type(df_raw)}. "
            f"Convert your data to Polars using pl.from_pandas() or pl.read_csv() before calling this function."
        )
    
    print("🚴‍♂️  Parametrized Feature Engineering Pipeline")
    print("=" * 60)

    # setup checkpointing
    checkpoint_dir = "feature_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if clear_checkpoints:
        clear_feature_checkpoints(checkpoint_dir)
    
    print(f"  Checkpoints directory: '{checkpoint_dir}/'")
    print(f"  Input raw data shape: {df_raw.shape}")
    _log_memory_usage("initialization")

    # check for final result first
    final_result_path = os.path.join(checkpoint_dir, "df_feat_final_result.parquet")
    if os.path.exists(final_result_path):
        print(f"  🎉 Final result already exists! Loading from: {final_result_path}")
        try:
            df_final = pl.read_parquet(final_result_path)
            print(f"  ✅ Successfully loaded final result with shape: {df_final.shape}")
            return df_final
        except Exception as e:
            print(f"  ⚠️ Error loading final result: {e}. Will recreate...")
            os.remove(final_result_path)

    # execute pipeline steps
    df, bike_models = step_01_normalize_polars_data(df_raw, dt_minutes, checkpoint_dir)
    
    # create feature tables
    departures = step_02_create_departures_table(df, bike_models, checkpoint_dir)
    arrivals = step_03_create_arrivals_table(df, checkpoint_dir)
    shifted_arrivals = step_04_create_shifted_arrivals(dt_minutes, checkpoint_dir)
    weather = step_05_create_weather_table(df, checkpoint_dir)
    stations = step_06_create_station_metadata(df, checkpoint_dir)
    
    # clear main dataframe from memory after extracting what we need
    del df
    gc.collect()
    _log_memory_usage("after clearing main dataframe")
    
    # build feature skeleton (time x station grid)
    print("Step 7: Building feature skeleton...")
    
    # create time skeleton
    time_range = departures.select([
        pl.col("ts_start").min().alias("start"),
        pl.col("ts_start").max().alias("end")
    ]).row(0)
    
    time_skeleton = pl.datetime_range(
        start=time_range[0],
        end=time_range[1],
        interval=f"{dt_minutes}m"
    ).alias("ts_start").to_frame()
    
    # cross join with stations to create full skeleton
    df_feat = (
        time_skeleton
        .join(stations.select("station_id"), how="cross")
        .sort(["ts_start", "station_id"])
    )
    
    print(f"  -> Feature skeleton shape: {df_feat.shape}")
    
    # join all feature tables
    print("Step 8: Joining feature tables...")
    
    # join departures
    df_feat = df_feat.join(
        departures,
        on=["station_id", "ts_start"],
        how="left"
    )
    
    # join arrivals
    df_feat = df_feat.join(
        arrivals,
        on=["station_id", "ts_start"],
        how="left"
    )
    
    # join shifted arrivals (target)
    df_feat = df_feat.join(
        shifted_arrivals,
        on=["station_id", "ts_start"],
        how="left"
    )
    
    # join weather if available
    if weather is not None:
        df_feat = df_feat.join(
            weather,
            on="ts_start",
            how="left"
        )
    
    # join station metadata
    df_feat = df_feat.join(
        stations,
        on="station_id",
        how="left"
    )
    
    # fill nulls with appropriate values
    df_feat = df_feat.with_columns([
        pl.col("dep_count").fill_null(0).alias("dep_last_DT"),
        pl.col("arr_count").fill_null(0).alias("arr_last_DT"),
        pl.col("y_arrivals_next_DT").fill_null(0),
        pl.col("trip_dur_mean").fill_null(0.0).alias("trip_dur_mean_last_DT")
    ])
    
    # add temporal features
    print("Step 9: Adding temporal features...")
    temporal_path = os.path.join(checkpoint_dir, "step_09_temporal.parquet")
    df_feat = _create_temporal_features(df_feat, checkpoint_dir, temporal_path, holiday_dates=[])
    
    print(f"  -> Final feature dataset shape: {df_feat.shape}")
    
    # save final result
    try:
        df_feat.lazy().sink_parquet(final_result_path, compression="snappy", row_group_size=10000)
        print(f"  ✅ Saved final result: {final_result_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save final result: {e}")
    
    return df_feat


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