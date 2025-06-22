"""
Data processing utilities for EcoBici-AI project.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any
from datetime import datetime, timedelta
import warnings


def process_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw station data to standardize column names and types.
    
    Args:
        df: Raw station DataFrame
        
    Returns:
        Processed DataFrame
    """
    df = df.copy()
    
    # standardize column names (common variations)
    column_mapping = {
        'station_id': 'station_id',
        'id_estacion': 'station_id',
        'estacion_id': 'station_id',
        'station_name': 'station_name',
        'nombre_estacion': 'station_name',
        'estacion_nombre': 'station_name',
        'latitude': 'latitude',
        'lat': 'latitude',
        'latitud': 'latitude',
        'longitude': 'longitude',
        'lon': 'longitude',
        'lng': 'longitude',
        'longitud': 'longitude'
    }
    
    # rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # ensure required columns exist
    required_columns = ['station_id', 'latitude', 'longitude']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # clean and convert data types
    df['station_id'] = df['station_id'].astype(str)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # remove rows with invalid coordinates
    initial_rows = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df) < initial_rows:
        warnings.warn(f"Removed {initial_rows - len(df)} rows with invalid coordinates")
    
    return df


def clean_trip_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw trip data by handling missing values and outliers.
    
    Args:
        df: Raw trip DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # standardize datetime columns
    datetime_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['fecha', 'date', 'inicio', 'fin', 'start', 'end']):
            datetime_columns.append(col)
    
    # convert to datetime
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # remove rows with missing datetime values
    initial_rows = len(df)
    df = df.dropna(subset=datetime_columns)
    
    if len(df) < initial_rows:
        warnings.warn(f"Removed {initial_rows - len(df)} rows with invalid datetime values")
    
    # calculate trip duration if not present
    if 'duracion' not in df.columns and len(datetime_columns) >= 2:
        start_col = datetime_columns[0]
        end_col = datetime_columns[1]
        df['duracion'] = (df[end_col] - df[start_col]).dt.total_seconds()
    
    # remove unrealistic trip durations (< 1 minute or > 24 hours)
    if 'duracion' in df.columns:
        initial_rows = len(df)
        df = df[(df['duracion'] >= 60) & (df['duracion'] <= 86400)]
        
        if len(df) < initial_rows:
            warnings.warn(f"Removed {initial_rows - len(df)} rows with unrealistic trip durations")
    
    return df


def normalize_coordinates(df: pd.DataFrame, lat_col: str = 'latitude', 
                         lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Normalize coordinate columns to a standard range.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with normalized coordinates
    """
    df = df.copy()
    
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns {lat_col} and {lon_col} must exist in DataFrame")
    
    # normalize to [0, 1] range
    df[f'{lat_col}_norm'] = (df[lat_col] - df[lat_col].min()) / (df[lat_col].max() - df[lat_col].min())
    df[f'{lon_col}_norm'] = (df[lon_col] - df[lon_col].min()) / (df[lon_col].max() - df[lon_col].min())
    
    return df


def create_station_metadata_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced station metadata with additional calculated features.
    
    Args:
        df: Station DataFrame
        
    Returns:
        Enhanced station metadata
    """
    df = df.copy()
    
    # calculate basic statistics if trip data is available
    if 'trip_count' not in df.columns:
        df['trip_count'] = 0
    
    # add coordinate-based features
    df = normalize_coordinates(df)
    
    # calculate distance to city center (Buenos Aires)
    # approximate city center coordinates
    center_lat, center_lon = -34.6118, -58.3960
    
    df['distance_to_center'] = np.sqrt(
        (df['latitude'] - center_lat) ** 2 + 
        (df['longitude'] - center_lon) ** 2
    )
    
    # categorize stations by location
    df['location_category'] = pd.cut(
        df['distance_to_center'],
        bins=[0, 0.02, 0.05, np.inf],
        labels=['center', 'inner', 'outer']
    )
    
    return df


def aggregate_trips_by_station(df: pd.DataFrame, 
                              station_col: str = 'station_id',
                              datetime_col: str = 'fecha_inicio') -> pd.DataFrame:
    """
    Aggregate trip data by station and time period.
    
    Args:
        df: Trip DataFrame
        station_col: Station identifier column
        datetime_col: Datetime column for aggregation
        
    Returns:
        Aggregated DataFrame
    """
    df = df.copy()
    
    # ensure datetime column is datetime type
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # create time-based features
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['month'] = df[datetime_col].dt.month
    df['date'] = df[datetime_col].dt.date
    
    # aggregate by station and time
    agg_dict = {
        'trip_id': 'count',  # assuming there's a trip_id column
        'duracion': ['mean', 'std', 'min', 'max'] if 'duracion' in df.columns else 'count'
    }
    
    # group by station and hour
    hourly_agg = df.groupby([station_col, 'date', 'hour']).agg(agg_dict).reset_index()
    
    # flatten column names
    hourly_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in hourly_agg.columns]
    
    return hourly_agg


def validate_data_quality(df: pd.DataFrame, 
                         required_columns: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # check for required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        quality_metrics['missing_required_columns'] = missing_cols
    
    # check data types
    quality_metrics['data_types'] = df.dtypes.to_dict()
    
    # check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers[col] = {
            'count': ((df[col] < lower_bound) | (df[col] > upper_bound)).sum(),
            'percentage': ((df[col] < lower_bound) | (df[col] > upper_bound)).sum() / len(df) * 100
        }
    
    quality_metrics['outliers'] = outliers
    
    return quality_metrics


__all__ = [
    'process_station_data',
    'clean_trip_data',
    'normalize_coordinates',
    'create_station_metadata_enhanced',
    'aggregate_trips_by_station',
    'validate_data_quality'
] 