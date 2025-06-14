"""
Temporal data processing utilities for EcoBici-AI.

This module contains functions for:
- Temporal data splitting (train/validation/test)
- Time-based filtering
- Temporal feature engineering helpers
- Holiday and seasonal feature creation
"""

import pandas as pd
import numpy as np
import polars as pl
from datetime import timedelta
from typing import Tuple, Dict, List
import holidays
import math
import os
import gc


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


def get_argentina_holidays(start_year: int = 2020, end_year: int = 2024) -> list:
    """
    Get list of Argentina holidays for the specified year range.
    
    Args:
        start_year: Start year for holiday calculation
        end_year: End year for holiday calculation
        
    Returns:
        List of holiday dates
    """
    holiday_dates = []
    
    for year in range(start_year, end_year + 1):
        # Get Argentina holidays for this year
        ar_holidays = holidays.Argentina(years=year)
        
        # Convert to list of datetime objects
        year_holidays = [date for date in ar_holidays.keys()]
        holiday_dates.extend(year_holidays)
    
    return sorted(holiday_dates)


def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Create temporal features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame with added temporal features
    """
    df_time = df.copy()
    
    # Ensure datetime column is properly typed
    if not pd.api.types.is_datetime64_any_dtype(df_time[datetime_col]):
        df_time[datetime_col] = pd.to_datetime(df_time[datetime_col], errors='coerce')
    
    dt_series = df_time[datetime_col]
    
    # Basic time features
    df_time['hour'] = dt_series.dt.hour
    df_time['day_of_week'] = dt_series.dt.dayofweek  # 0=Monday
    df_time['month'] = dt_series.dt.month
    df_time['year'] = dt_series.dt.year
    df_time['day_of_year'] = dt_series.dt.dayofyear
    df_time['week_of_year'] = dt_series.dt.isocalendar().week
    
    # Cyclical features
    df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
    df_time['dow_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['dow_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
    
    # Weekend flag
    df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
    
    # Time of day categories
    df_time['time_of_day'] = pd.cut(df_time['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'],
                                   include_lowest=True)
    
    return df_time


def add_holiday_features(df: pd.DataFrame, datetime_col: str, 
                        holiday_dates: list = None) -> pd.DataFrame:
    """
    Add holiday-related features to DataFrame.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
        holiday_dates: List of holiday dates (auto-generated if None)
        
    Returns:
        DataFrame with added holiday features
    """
    df_holiday = df.copy()
    
    # Ensure datetime column is properly typed
    if not pd.api.types.is_datetime64_any_dtype(df_holiday[datetime_col]):
        df_holiday[datetime_col] = pd.to_datetime(df_holiday[datetime_col], errors='coerce')
    
    if holiday_dates is None:
        # Get holidays for the date range in the data
        min_year = df_holiday[datetime_col].dt.year.min()
        max_year = df_holiday[datetime_col].dt.year.max()
        holiday_dates = get_argentina_holidays(min_year, max_year)
    
    # Convert holiday dates to set for faster lookup
    holiday_set = set(pd.to_datetime(holiday_dates).date)
    
    # Create holiday flag
    df_holiday['is_holiday'] = df_holiday[datetime_col].dt.date.isin(holiday_set).astype(int)
    
    # Days before/after holiday
    df_holiday['days_to_holiday'] = (
        df_holiday[datetime_col].dt.date.apply(
            lambda x: min([abs((x - h).days) for h in holiday_set] + [365])
        )
    )
    
    # Weekend or holiday flag
    df_holiday['is_weekend_or_holiday'] = (
        ((df_holiday[datetime_col].dt.dayofweek >= 5) | 
         (df_holiday['is_holiday'] == 1)).astype(int)
    )
    
    return df_holiday


def get_season(month: int) -> str:
    """
    Get season based on month (Southern Hemisphere).
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name
    """
    if month in [12, 1, 2]:
        return 'summer'
    elif month in [3, 4, 5]:
        return 'autumn'
    elif month in [6, 7, 8]:
        return 'winter'
    else:  # 9, 10, 11
        return 'spring'


def add_season_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Add seasonal features to DataFrame.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame with added seasonal features
    """
    df_season = df.copy()
    
    # Ensure datetime column is properly typed
    if not pd.api.types.is_datetime64_any_dtype(df_season[datetime_col]):
        df_season[datetime_col] = pd.to_datetime(df_season[datetime_col], errors='coerce')
    
    # Add season
    df_season['season'] = df_season[datetime_col].dt.month.apply(get_season)
    
    # Season encoding
    season_mapping = {'summer': 0, 'autumn': 1, 'winter': 2, 'spring': 3}
    df_season['season_encoded'] = df_season['season'].map(season_mapping)
    
    # Vacation seasons (summer and winter holidays in Argentina)
    df_season['is_vacation_season'] = (
        df_season['season'].isin(['summer', 'winter']).astype(int)
    )
    
    return df_season


def create_temporal_features_optimized(df_feat: pl.DataFrame, checkpoint_dir: str, temporal_path: str, holiday_dates: list) -> pl.DataFrame:
    """Create temporal features with memory-efficient operations.
    
    Changes:
      • Processes data in time windows to reduce memory pressure
      • Uses smaller data types (Int8, Float32) to save memory
      • Uses streaming operations for all heavy computations
      • Saves intermediate results to disk
      • Uses aggressive compression
    """
    print("  -> Creating temporal features with memory optimization...")
    
    # Get time range for windowing
    min_ts = df_feat["ts_start"].min()
    max_ts = df_feat["ts_start"].max()
    time_window = timedelta(days=7)  # Process one week at a time
    
    # Holidays for Argentina - handle safely
    try:
        if not holiday_dates:  # Only compute if not provided
            min_year = df_feat["ts_start"].dt.year().min()
            max_year = df_feat["ts_start"].dt.year().max()
            ar_holidays = holidays.AR(years=range(min_year, max_year + 1))
            holiday_dates = [date for date in ar_holidays.keys()]
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
        
        # log memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"  -> Memory usage after window until {window_end}: {memory_mb:.1f} MB")
        except:
            pass
    
    print("  -> Combining all temporal windows...")
    # Combine all windows using streaming
    df_feat_final = pl.concat(
        [pl.scan_parquet(f) for f in chunk_paths],
        how="vertical"
    ).collect()
    
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
        print("  -> Temporal features checkpoint saved")
    except Exception as e:
        print(f"  -> Error saving temporal features: {e}")
    
    return df_feat_final 