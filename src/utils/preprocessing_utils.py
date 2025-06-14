"""
Preprocessing utilities for EcoBici data.

This module provides functionality for:
- Data cleaning and validation
- Missing data handling  
- Feature scaling and normalization
- Data type conversions
- Quality checks and filtering
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings


def clean_column_names(df: pl.DataFrame, lowercase: bool = True, remove_spaces: bool = True) -> pl.DataFrame:
    """
    Clean column names for consistency.
    
    Args:
        df: Input DataFrame
        lowercase: Whether to convert to lowercase
        remove_spaces: Whether to replace spaces with underscores
        
    Returns:
        DataFrame with cleaned column names
    """
    print("Cleaning column names...")
    
    old_columns = df.columns
    new_columns = old_columns.copy()
    
    if lowercase:
        new_columns = [col.lower() for col in new_columns]
    
    if remove_spaces:
        new_columns = [col.replace(' ', '_').replace('-', '_') for col in new_columns]
    
    # create mapping for renaming
    rename_mapping = {old: new for old, new in zip(old_columns, new_columns) if old != new}
    
    if rename_mapping:
        df = df.rename(rename_mapping)
        print(f"  -> Renamed {len(rename_mapping)} columns")
    else:
        print("  -> No column names needed cleaning")
    
    return df


def standardize_datetime_columns(df: pl.DataFrame, 
                                datetime_cols: List[str],
                                date_format: Optional[str] = None) -> pl.DataFrame:
    """
    Standardize datetime columns to proper Polars datetime format.
    
    Args:
        df: Input DataFrame
        datetime_cols: List of datetime column names
        date_format: Optional specific date format to parse
        
    Returns:
        DataFrame with standardized datetime columns
    """
    print(f"Standardizing {len(datetime_cols)} datetime columns...")
    
    df_clean = df.clone()
    
    for col in datetime_cols:
        if col not in df.columns:
            print(f"  -> Warning: Column '{col}' not found, skipping")
            continue
            
        print(f"  -> Processing {col}...")
        
        try:
            if date_format:
                df_clean = df_clean.with_columns([
                    pl.col(col).str.strptime(pl.Datetime, date_format).alias(col)
                ])
            else:
                # try automatic parsing
                df_clean = df_clean.with_columns([
                    pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias(col)
                ])
        except:
            try:
                # fallback to pandas-style parsing
                df_clean = df_clean.with_columns([
                    pl.col(col).str.strptime(pl.Datetime).alias(col)
                ])
            except Exception as e:
                print(f"    -> Failed to parse {col}: {e}")
                continue
    
    return df_clean


def handle_missing_values(df: pl.DataFrame, 
                         strategy: Dict[str, str] = None,
                         drop_threshold: float = 0.5) -> pl.DataFrame:
    """
    Handle missing values with various strategies.
    
    Args:
        df: Input DataFrame
        strategy: Dict mapping column names to fill strategies ('mean', 'median', 'mode', 'zero', 'drop')
        drop_threshold: Proportion of missing values above which to drop the column
        
    Returns:
        DataFrame with handled missing values
    """
    print("Handling missing values...")
    
    if strategy is None:
        strategy = {}
    
    df_clean = df.clone()
    
    # analyze missing values
    missing_info = []
    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = null_count / df.height * 100
        missing_info.append({
            'column': col,
            'null_count': null_count,
            'null_percentage': null_pct
        })
    
    missing_df = pd.DataFrame(missing_info).sort_values('null_percentage', ascending=False)
    
    print("  -> Missing value analysis:")
    for _, row in missing_df.head(10).iterrows():
        if row['null_count'] > 0:
            print(f"     {row['column']}: {row['null_count']} ({row['null_percentage']:.1f}%)")
    
    # drop columns with too many missing values
    cols_to_drop = []
    for _, row in missing_df.iterrows():
        if row['null_percentage'] > drop_threshold * 100:
            cols_to_drop.append(row['column'])
    
    if cols_to_drop:
        print(f"  -> Dropping {len(cols_to_drop)} columns with >{drop_threshold*100}% missing values")
        df_clean = df_clean.drop(cols_to_drop)
    
    # handle remaining missing values
    for col in df_clean.columns:
        if col in strategy:
            fill_strategy = strategy[col]
            
            if fill_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif fill_strategy == 'median':
                fill_value = df_clean[col].median()
            elif fill_strategy == 'mode':
                fill_value = df_clean[col].mode().first()
            elif fill_strategy == 'zero':
                fill_value = 0
            elif fill_strategy == 'drop':
                df_clean = df_clean.drop_nulls(subset=[col])
                continue
            else:
                print(f"    -> Unknown strategy '{fill_strategy}' for {col}")
                continue
                
            df_clean = df_clean.with_columns([
                pl.col(col).fill_null(fill_value)
            ])
            
            print(f"    -> Filled {col} with {fill_strategy} strategy")
    
    return df_clean


def detect_and_fix_outliers(df: pl.DataFrame,
                           numerical_cols: List[str],
                           method: str = 'iqr',
                           threshold: float = 1.5) -> pl.DataFrame:
    """
    Detect and fix outliers in numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    print(f"Detecting outliers using {method} method...")
    
    df_clean = df.clone()
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
            
        print(f"  -> Processing {col}...")
        
        if method == 'iqr':
            # interquartile range method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # count outliers
            outliers = df.filter(
                (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
            ).height
            
            # cap outliers
            df_clean = df_clean.with_columns([
                pl.col(col).clip(lower_bound, upper_bound)
            ])
            
        elif method == 'zscore':
            # z-score method
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # count outliers
            outliers = df.filter(
                (pl.col(col) - mean_val).abs() > threshold * std_val
            ).height
            
            # cap outliers
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            df_clean = df_clean.with_columns([
                pl.col(col).clip(lower_bound, upper_bound)
            ])
        
        if outliers > 0:
            print(f"    -> Fixed {outliers} outliers in {col}")
    
    return df_clean


def validate_data_quality(df: pl.DataFrame, 
                         checks: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: Input DataFrame
        checks: Dictionary of specific checks to perform
        
    Returns:
        Dictionary with data quality results
    """
    print("Performing data quality validation...")
    
    if checks is None:
        checks = {
            'duplicates': True,
            'missing_values': True,
            'data_types': True,
            'value_ranges': True
        }
    
    results = {
        'total_rows': df.height,
        'total_columns': len(df.columns),
        'issues': []
    }
    
    # check for duplicates
    if checks.get('duplicates', False):
        duplicate_count = df.height - df.unique().height
        results['duplicates'] = duplicate_count
        if duplicate_count > 0:
            results['issues'].append(f"Found {duplicate_count} duplicate rows")
    
    # check missing values
    if checks.get('missing_values', False):
        missing_stats = {}
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                missing_stats[col] = {
                    'count': null_count,
                    'percentage': null_count / df.height * 100
                }
        results['missing_values'] = missing_stats
        if missing_stats:
            results['issues'].append(f"Missing values found in {len(missing_stats)} columns")
    
    # check data types
    if checks.get('data_types', False):
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        results['data_types'] = dtypes
    
    # check value ranges for numerical columns
    if checks.get('value_ranges', False):
        numerical_stats = {}
        for col in df.columns:
            if df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                try:
                    stats = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
                    numerical_stats[col] = stats
                    
                    # check for suspicious values
                    if stats['min'] < 0 and col.endswith(('_count', '_duration')):
                        results['issues'].append(f"Negative values found in {col}")
                        
                except:
                    continue
        results['numerical_stats'] = numerical_stats
    
    # summary
    print(f"  -> Data shape: {results['total_rows']:,} rows × {results['total_columns']} columns")
    if results['issues']:
        print(f"  -> Found {len(results['issues'])} potential issues:")
        for issue in results['issues']:
            print(f"     - {issue}")
    else:
        print("  -> No data quality issues detected")
    
    return results


def filter_by_date_range(df: pl.DataFrame,
                        date_col: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pl.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        
    Returns:
        Filtered DataFrame
    """
    print(f"Filtering by date range...")
    
    if date_col not in df.columns:
        print(f"  -> Warning: Date column '{date_col}' not found")
        return df
    
    df_filtered = df.clone()
    original_count = df_filtered.height
    
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        df_filtered = df_filtered.filter(pl.col(date_col) >= start_dt)
        print(f"  -> Applied start date filter: {start_date}")
    
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        df_filtered = df_filtered.filter(pl.col(date_col) <= end_dt)
        print(f"  -> Applied end date filter: {end_date}")
    
    filtered_count = df_filtered.height
    removed_count = original_count - filtered_count
    
    print(f"  -> Filtered: {original_count:,} → {filtered_count:,} rows (-{removed_count:,})")
    
    return df_filtered


def normalize_station_ids(df: pl.DataFrame, 
                         station_cols: List[str]) -> pl.DataFrame:
    """
    Normalize station IDs to ensure consistency.
    
    Args:
        df: Input DataFrame
        station_cols: List of station ID column names
        
    Returns:
        DataFrame with normalized station IDs
    """
    print(f"Normalizing {len(station_cols)} station ID columns...")
    
    df_clean = df.clone()
    
    for col in station_cols:
        if col not in df.columns:
            continue
            
        print(f"  -> Processing {col}...")
        
        # convert to integer and handle nulls
        df_clean = df_clean.with_columns([
            pl.col(col).cast(pl.Int32, strict=False)
        ])
        
        # check for negative or zero station IDs
        invalid_count = df_clean.filter(pl.col(col) <= 0).height
        if invalid_count > 0:
            print(f"    -> Found {invalid_count} invalid station IDs (≤0)")
            df_clean = df_clean.filter(pl.col(col) > 0)
    
    return df_clean


def standardize_coordinate_precision(df: pl.DataFrame,
                                   lat_col: str = "lat",
                                   lon_col: str = "lon",
                                   precision: int = 6) -> pl.DataFrame:
    """
    Standardize coordinate precision to avoid floating point issues.
    
    Args:
        df: Input DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        precision: Number of decimal places to round to
        
    Returns:
        DataFrame with standardized coordinates
    """
    print(f"Standardizing coordinate precision to {precision} decimal places...")
    
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"  -> Warning: Coordinate columns not found")
        return df
    
    df_clean = df.with_columns([
        pl.col(lat_col).round(precision),
        pl.col(lon_col).round(precision)
    ])
    
    # validate coordinate ranges for Buenos Aires
    buenos_aires_bounds = {
        'lat_min': -35.0, 'lat_max': -34.0,
        'lon_min': -59.0, 'lon_max': -58.0
    }
    
    invalid_coords = df_clean.filter(
        (pl.col(lat_col) < buenos_aires_bounds['lat_min']) |
        (pl.col(lat_col) > buenos_aires_bounds['lat_max']) |
        (pl.col(lon_col) < buenos_aires_bounds['lon_min']) |
        (pl.col(lon_col) > buenos_aires_bounds['lon_max'])
    ).height
    
    if invalid_coords > 0:
        print(f"  -> Warning: {invalid_coords} rows have coordinates outside Buenos Aires bounds")
    
    return df_clean


def create_preprocessing_pipeline(steps: List[Dict[str, Any]]) -> callable:
    """
    Create a preprocessing pipeline from a list of steps.
    
    Args:
        steps: List of preprocessing steps, each with 'function' and 'params' keys
        
    Returns:
        Callable preprocessing pipeline function
    """
    print(f"Creating preprocessing pipeline with {len(steps)} steps...")
    
    def pipeline(df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        
        for i, step in enumerate(steps):
            step_name = step.get('name', f'Step {i+1}')
            func = step['function']
            params = step.get('params', {})
            
            print(f"  -> Executing {step_name}...")
            result = func(result, **params)
        
        return result
    
    return pipeline


def quick_data_summary(df: pl.DataFrame) -> None:
    """
    Print a quick summary of the DataFrame.
    
    Args:
        df: Input DataFrame
    """
    print("📊 DATA SUMMARY")
    print("=" * 50)
    print(f"Shape: {df.height:,} rows × {len(df.columns)} columns")
    print(f"Memory usage: ~{df.estimated_size() / 1024 / 1024:.1f} MB")
    
    print(f"\n📋 Column Types:")
    dtype_counts = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count} columns")
    
    print(f"\n🔍 Missing Values:")
    has_missing = False
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            print(f"  {col}: {null_count:,} ({null_count/df.height*100:.1f}%)")
            has_missing = True
    
    if not has_missing:
        print("  No missing values found")
    
    print("=" * 50) 