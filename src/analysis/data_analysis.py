"""
Data analysis utilities for EcoBici-AI.

This module contains functions for:
- User and trip data analysis for visualization
- Statistical analysis and data profiling
- Data quality assessment
- Exploratory data analysis functions
- Raw data analysis and summary
- Weather impact analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re


def analyze_users_for_visualization(users_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze users data for visualization purposes.
    
    Args:
        users_df: Users DataFrame
        
    Returns:
        Dict with processed data for visualization
    """
    processed_data = {}
    
    # Age analysis
    if 'edad_usuario' in users_df.columns:
        users_clean = users_df.copy()
        users_clean['edad_usuario'] = pd.to_numeric(users_clean['edad_usuario'], errors='coerce')
        edad_valid = users_clean['edad_usuario'].dropna()
        edad_valid = edad_valid[(edad_valid >= 0) & (edad_valid <= 150)]
        processed_data['edad_valid'] = edad_valid
        processed_data['edad_promedio'] = edad_valid.mean()
    
    # Gender analysis
    if 'genero_usuario' in users_df.columns:
        processed_data['genero_counts'] = users_df['genero_usuario'].value_counts()
    
    # Registration date analysis
    if 'fecha_alta' in users_df.columns:
        users_temp = users_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(users_temp['fecha_alta']):
            users_temp['fecha_alta'] = pd.to_datetime(users_temp['fecha_alta'], errors='coerce')
        
        processed_data['registros_por_año'] = users_temp['fecha_alta'].dt.year.value_counts().sort_index()
        processed_data['registros_por_mes'] = users_temp['fecha_alta'].dt.month.value_counts().sort_index()
    
    return processed_data


def analyze_trips_for_visualization(trips_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trips data for visualization purposes.
    
    Args:
        trips_df: Trips DataFrame
        
    Returns:
        Dict with processed data for visualization
    """
    processed_data = {}
    
    # Prepare temporal data
    trips_temp = trips_df.copy()
    fecha_col = 'fecha_origen_recorrido' if 'fecha_origen_recorrido' in trips_temp.columns else 'fecha_origen'
    
    if not pd.api.types.is_datetime64_any_dtype(trips_temp[fecha_col]):
        trips_temp[fecha_col] = pd.to_datetime(trips_temp[fecha_col], errors='coerce')
    
    trips_temp = trips_temp.dropna(subset=[fecha_col])
    processed_data['trips_clean'] = trips_temp
    processed_data['fecha_col'] = fecha_col
    
    # Temporal analysis
    processed_data['viajes_por_hora'] = trips_temp[fecha_col].dt.hour.value_counts().sort_index()
    processed_data['viajes_por_dia'] = trips_temp[fecha_col].dt.dayofweek.value_counts().sort_index()
    processed_data['viajes_por_mes'] = trips_temp[fecha_col].dt.month.value_counts().sort_index()
    processed_data['viajes_por_año'] = trips_temp[fecha_col].dt.year.value_counts().sort_index()
    
    # Station analysis
    if 'id_estacion_origen' in trips_temp.columns:
        processed_data['top_origen'] = trips_temp['id_estacion_origen'].value_counts().head(15)
        processed_data['viajes_por_estacion'] = trips_temp['id_estacion_origen'].value_counts()
    
    if 'id_estacion_destino' in trips_temp.columns:
        processed_data['top_destino'] = trips_temp['id_estacion_destino'].value_counts().head(15)
    
    # Duration analysis
    if 'duracion_recorrido' in trips_temp.columns:
        trips_temp['duracion_num'] = pd.to_numeric(
            trips_temp['duracion_recorrido'].astype(str).str.replace(',', ''), 
            errors='coerce'
        )
        duracion_valid = trips_temp['duracion_num'].dropna()
        
        # create different duration ranges for analysis
        # 1. reasonable trips: >2 min, <12 hours (original filter)  
        duracion_reasonable = duracion_valid[(duracion_valid > 120) & (duracion_valid < 60 * 60 * 24)]
        
        # 2. extended range for visualization: >1 min, <24 hours
        duracion_extended = duracion_valid[(duracion_valid > 60) & (duracion_valid < 60 * 60 * 24)]
        
        # 3. full range (for outlier analysis)
        duracion_full = duracion_valid[duracion_valid > 0]
        
        # store all ranges for flexible analysis
        processed_data['duracion_valid'] = duracion_reasonable  # default for compatibility
        processed_data['duracion_extended'] = duracion_extended  # extended range
        processed_data['duracion_full'] = duracion_full  # full range including outliers
        processed_data['duracion_promedio'] = duracion_reasonable.mean()
        
        # add percentile information
        processed_data['duracion_stats'] = {
            'q99': duracion_full.quantile(0.99),
            'q95': duracion_full.quantile(0.95),
            'q90': duracion_full.quantile(0.90),
            'median': duracion_full.median(),
            'mean': duracion_full.mean()
        }
    
    # Heatmap data
    processed_data['hora'] = trips_temp[fecha_col].dt.hour
    processed_data['dia_semana'] = trips_temp[fecha_col].dt.dayofweek
    processed_data['activity_matrix'] = trips_temp.groupby([
        trips_temp[fecha_col].dt.dayofweek, 
        trips_temp[fecha_col].dt.hour
    ]).size().unstack(fill_value=0)
    
    return processed_data


def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with data quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values': {},
        'data_types': df.dtypes.value_counts().to_dict(),
        'completeness_score': 0.0
    }
    
    # Missing values analysis
    missing = df.isnull().sum()
    total_cells = len(df) * len(df.columns)
    total_missing = missing.sum()
    
    metrics['missing_values'] = {
        'total_missing_cells': int(total_missing),
        'missing_percentage': float(total_missing / total_cells * 100),
        'columns_with_missing': missing[missing > 0].to_dict()
    }
    
    # Calculate completeness score
    metrics['completeness_score'] = float((total_cells - total_missing) / total_cells * 100)
    
    return metrics


def analyze_categorical_columns(df: pd.DataFrame, max_unique_values: int = 50) -> Dict[str, Any]:
    """
    Analyze categorical columns in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        max_unique_values: Maximum unique values to consider a column categorical
        
    Returns:
        Dictionary with categorical analysis
    """
    categorical_analysis = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= max_unique_values:
            value_counts = df[col].value_counts()
            
            categorical_analysis[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.head(10).to_dict()
            }
    
    return categorical_analysis


def analyze_numerical_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze numerical columns in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with numerical analysis
    """
    numerical_analysis = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        series = df[col].dropna()
        
        if len(series) > 0:
            numerical_analysis[col] = {
                'count': len(series),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'outliers_iqr': _count_outliers_iqr(series),
                'zeros': int((series == 0).sum()),
                'negatives': int((series < 0).sum())
            }
    
    return numerical_analysis


def _count_outliers_iqr(series: pd.Series) -> int:
    """Count outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)


def analyze_temporal_patterns(df: pd.DataFrame, datetime_col: str) -> Dict[str, Any]:
    """
    Analyze temporal patterns in a DataFrame.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        Dictionary with temporal analysis
    """
    if datetime_col not in df.columns:
        return {}
    
    df_temp = df.copy()
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_temp[datetime_col]):
        df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
    
    df_temp = df_temp.dropna(subset=[datetime_col])
    
    if len(df_temp) == 0:
        return {}
    
    temporal_analysis = {
        'date_range': {
            'start': df_temp[datetime_col].min(),
            'end': df_temp[datetime_col].max(),
            'duration_days': (df_temp[datetime_col].max() - df_temp[datetime_col].min()).days
        },
        'patterns': {
            'hourly': df_temp[datetime_col].dt.hour.value_counts().sort_index().to_dict(),
            'daily': df_temp[datetime_col].dt.dayofweek.value_counts().sort_index().to_dict(),
            'monthly': df_temp[datetime_col].dt.month.value_counts().sort_index().to_dict(),
            'yearly': df_temp[datetime_col].dt.year.value_counts().sort_index().to_dict()
        },
        'seasonality': {
            'weekend_vs_weekday': _analyze_weekend_vs_weekday(df_temp, datetime_col),
            'business_hours': _analyze_business_hours(df_temp, datetime_col)
        }
    }
    
    return temporal_analysis


def _analyze_weekend_vs_weekday(df: pd.DataFrame, datetime_col: str) -> Dict[str, int]:
    """Analyze weekend vs weekday patterns."""
    is_weekend = df[datetime_col].dt.dayofweek >= 5
    return {
        'weekend_count': int(is_weekend.sum()),
        'weekday_count': int((~is_weekend).sum()),
        'weekend_percentage': float(is_weekend.mean() * 100)
    }


def _analyze_business_hours(df: pd.DataFrame, datetime_col: str) -> Dict[str, int]:
    """Analyze business hours patterns."""
    hour = df[datetime_col].dt.hour
    is_business_hours = (hour >= 9) & (hour <= 17)
    return {
        'business_hours_count': int(is_business_hours.sum()),
        'after_hours_count': int((~is_business_hours).sum()),
        'business_hours_percentage': float(is_business_hours.mean() * 100)
    }


def generate_data_profile_report(df: pd.DataFrame, datetime_col: str = None) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile report.
    
    Args:
        df: DataFrame to profile
        datetime_col: Optional datetime column for temporal analysis
        
    Returns:
        Dictionary with comprehensive data profile
    """
    report = {
        'overview': calculate_data_quality_metrics(df),
        'categorical_analysis': analyze_categorical_columns(df),
        'numerical_analysis': analyze_numerical_columns(df)
    }
    
    if datetime_col:
        report['temporal_analysis'] = analyze_temporal_patterns(df, datetime_col)
    
    return report


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                      name1: str = "Dataset 1", name2: str = "Dataset 2") -> Dict[str, Any]:
    """
    Compare two DataFrames and highlight differences.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        name1: Name for first DataFrame
        name2: Name for second DataFrame
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'shape_comparison': {
            name1: df1.shape,
            name2: df2.shape
        },
        'column_comparison': {
            'common_columns': list(set(df1.columns) & set(df2.columns)),
            f'only_in_{name1}': list(set(df1.columns) - set(df2.columns)),
            f'only_in_{name2}': list(set(df2.columns) - set(df1.columns))
        },
        'data_type_differences': {},
        'missing_value_comparison': {}
    }
    
    # Compare data types for common columns
    common_cols = comparison['column_comparison']['common_columns']
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            comparison['data_type_differences'][col] = {
                name1: str(df1[col].dtype),
                name2: str(df2[col].dtype)
            }
    
    # Compare missing values for common columns
    for col in common_cols:
        missing1 = df1[col].isnull().sum()
        missing2 = df2[col].isnull().sum()
        if missing1 != missing2:
            comparison['missing_value_comparison'][col] = {
                name1: int(missing1),
                name2: int(missing2)
            }
    
    return comparison


def analyze_raw_data(df: pd.DataFrame) -> None:
    """
    Analyze raw data and print summary statistics.
    
    Args:
        df: DataFrame to analyze
    """
    print("=" * 60)
    print("RAW DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn types:")
    print(df.dtypes.value_counts())
    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print(missing)
    else:
        print("No missing values found!")
    
    print("\nFirst few rows:")
    print(df.head())
    
    # station-specific analysis if columns exist
    if 'id_estacion_origen' in df.columns:
        print(f"\nUnique origin stations: {df['id_estacion_origen'].nunique()}")
        
    if 'id_estacion_destino' in df.columns:
        print(f"Unique destination stations: {df['id_estacion_destino'].nunique()}")
        
    if 'fecha_origen_recorrido' in df.columns:
        df_temp = df.copy()
        df_temp['fecha_origen_recorrido'] = pd.to_datetime(df_temp['fecha_origen_recorrido'], errors='coerce')
        print(f"Date range: {df_temp['fecha_origen_recorrido'].min()} to {df_temp['fecha_origen_recorrido'].max()}")
    
    print("=" * 60)


def analyze_weather_impact(trips_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Analyze the impact of weather conditions on bike usage.
    
    Args:
        trips_df: DataFrame with trip data and weather conditions
        
    Returns:
        Dictionary with weather impact analysis or None if no weather data
    """
    print("analyzing weather impact on bike usage...")
    
    if 'temperature_2m' not in trips_df.columns:
        print("no weather data found in trips dataframe")
        return None
    
    # group by weather conditions
    temp_bins = [-10, 5, 15, 25, 35, 50]
    temp_labels = ['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot']
    trips_df['temp_category'] = pd.cut(trips_df['temperature_2m'], bins=temp_bins, labels=temp_labels)
    
    rain_conditions = {
        'No Rain': trips_df['precipitation'] == 0,
        'Light Rain': (trips_df['precipitation'] > 0) & (trips_df['precipitation'] <= 1),
        'Moderate Rain': (trips_df['precipitation'] > 1) & (trips_df['precipitation'] <= 5),
        'Heavy Rain': trips_df['precipitation'] > 5
    }
    
    results = {
        'temperature_impact': trips_df.groupby('temp_category').size().to_dict(),
        'precipitation_impact': {
            condition: len(trips_df[mask]) for condition, mask in rain_conditions.items()
        },
        'optimal_conditions': {
            'temperature_range': (15, 25),
            'max_acceptable_rain': 1.0,
            'max_acceptable_wind': 15.0
        }
    }
    
    return results


def print_weather_analysis(results: Dict[str, Any], total_trips: int) -> None:
    """
    Print formatted weather impact analysis.
    
    Args:
        results: Dictionary from analyze_weather_impact
        total_trips: Total number of trips for percentage calculations
    """
    print("\n🌤️  WEATHER IMPACT ANALYSIS")
    print("=" * 50)
    
    print("📊 Temperature Impact:")
    for temp_cat, count in results['temperature_impact'].items():
        percentage = count / total_trips * 100
        print(f"   {temp_cat}: {count:,} trips ({percentage:.1f}%)")
    
    print("\n🌧️  Precipitation Impact:")
    for rain_condition, count in results['precipitation_impact'].items():
        percentage = count / total_trips * 100
        print(f"   {rain_condition}: {count:,} trips ({percentage:.1f}%)")
    
    print(f"\n🎯 Optimal riding conditions:")
    opt = results['optimal_conditions']
    print(f"   Temperature: {opt['temperature_range'][0]}°C - {opt['temperature_range'][1]}°C")
    print(f"   Max rain: {opt['max_acceptable_rain']}mm")
    print(f"   Max wind: {opt['max_acceptable_wind']}km/h") 