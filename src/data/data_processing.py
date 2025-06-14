"""
Station data processing utilities for EcoBici-AI.

This module contains functions for:
- Station ID processing and normalization
- Coordinate cleaning and validation
- Station metadata extraction
- Data consistency analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def extract_first_3_digits_station_id(station_id):
    """
    Extract first 3 digits from station ID for consistency.
    
    This function normalizes station IDs by extracting the first 3 digits,
    which helps in handling different ID formats across years.
    
    Args:
        station_id: Original station ID (can be int, float, or string)
        
    Returns:
        int: Normalized station ID (first 3 digits)
    """
    if pd.isna(station_id):
        return station_id
        
    # Remove BAEcobici suffix if present
    station_str = str(station_id).replace('BAEcobici', '')
    
    try:
        # Convert to int first to handle float strings
        station_str = str(int(float(station_str)))
        
        if len(station_str) >= 3:
            return int(station_str[:3])
        else:
            return int(station_str)
    except (ValueError, TypeError):
        return station_id


def process_station_ids_to_3_digits(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Process station IDs in a DataFrame to use 3-digit format.
    
    Args:
        df: DataFrame with station ID columns
        verbose: Whether to print processing information
        
    Returns:
        DataFrame with processed station IDs
    """
    df_processed = df.copy()
    
    station_columns = [
        'id_estacion_origen', 'id_estacion_destino'
    ]
    
    for col in station_columns:
        if col in df_processed.columns:
            if verbose:
                original_unique = df_processed[col].nunique()
                
            # Apply the 3-digit extraction
            df_processed[col] = df_processed[col].apply(extract_first_3_digits_station_id)
            
            if verbose:
                new_unique = df_processed[col].nunique()
                print(f"  {col}: {original_unique} → {new_unique} unique values")
    
    if verbose:
        print(f"processed {len(df_processed)} rows")
        
    return df_processed


def clean_coordinate_pair(latitude, longitude):
    """
    Clean and validate coordinate pairs.
    
    Args:
        latitude: Latitude value (can be string or numeric)
        longitude: Longitude value (can be string or numeric)
        
    Returns:
        tuple: (cleaned_coord_string, issue_type)
    """
    lat_str = str(latitude).strip()
    lon_str = str(longitude).strip()
    
    # Detect malformed coordinates (format: "lat,lon")
    if ',' in lat_str and ',' not in lon_str:
        # Case: latitude contains "lat,lon" and longitude is only one value
        parts = lat_str.split(',')
        if len(parts) == 2:
            clean_lat = parts[0].strip()
            clean_lon = parts[1].strip()
            return f"({clean_lat}, {clean_lon})", "FIXED_FROM_MALFORMED"
    
    # Detect duplicate coordinates
    if lat_str == lon_str:
        return f"({lat_str}, {lon_str})", "DUPLICATE_COORDS"
    
    # Normal format
    return f"({lat_str}, {lon_str})", "NORMAL"


def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str) -> Dict[str, Any]:
    """
    Validate coordinate columns in a DataFrame.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        Dict with validation results
    """
    results = {
        'total_rows': len(df),
        'valid_coords': 0,
        'invalid_coords': 0,
        'null_coords': 0,
        'coordinate_issues': {}
    }
    
    # Check for null coordinates
    null_mask = df[lat_col].isna() | df[lon_col].isna()
    results['null_coords'] = null_mask.sum()
    
    # Validate non-null coordinates
    valid_df = df[~null_mask].copy()
    
    for idx, row in valid_df.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        coord_str, issue_type = clean_coordinate_pair(lat, lon)
        
        if issue_type == "NORMAL":
            results['valid_coords'] += 1
        else:
            results['invalid_coords'] += 1
            if issue_type not in results['coordinate_issues']:
                results['coordinate_issues'][issue_type] = 0
            results['coordinate_issues'][issue_type] += 1
    
    return results


def normalize_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize station metadata by consolidating duplicate stations.
    
    Args:
        df: DataFrame with station data
        
    Returns:
        DataFrame with normalized station metadata
    """
    # Extract unique station information
    station_cols = [
        'id_estacion_origen', 'nombre_estacion_origen', 
        'lat_estacion_origen', 'long_estacion_origen'
    ]
    
    if all(col in df.columns for col in station_cols):
        # Create station metadata from origin data
        origin_meta = df[station_cols].drop_duplicates('id_estacion_origen')
        origin_meta.columns = ['station_id', 'station_name', 'lat', 'lon']
        
        # Extract destination metadata
        dest_cols = [
            'id_estacion_destino', 'nombre_estacion_destino',
            'lat_estacion_destino', 'long_estacion_destino'
        ]
        
        if all(col in df.columns for col in dest_cols):
            dest_meta = df[dest_cols].drop_duplicates('id_estacion_destino')
            dest_meta.columns = ['station_id', 'station_name', 'lat', 'lon']
            
            # Combine and deduplicate
            combined_meta = pd.concat([origin_meta, dest_meta])
            final_meta = combined_meta.drop_duplicates('station_id', keep='first')
            
            return final_meta.sort_values('station_id').reset_index(drop=True)
    
    return pd.DataFrame()


# Note: analyze_raw_data function has been moved to src/analysis/data_analysis.py 