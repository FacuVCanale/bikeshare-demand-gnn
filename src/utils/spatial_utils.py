"""
Spatial and mathematical utilities for EcoBici-AI.

This module contains functions for:
- Geospatial calculations (haversine distance)
- Coordinate transformations and normalization
- Neighbor finding algorithms
- Mathematical utility functions
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional
import math



def haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance calculation (km) - used as metric in NearestNeighbors.
    
    Args:
        lon1, lat1: Longitude and latitude of first point(s)
        lon2, lat2: Longitude and latitude of second point(s)
        
    Returns:
        Distance in kilometers
    """
    # convert to radians
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def haversine_single(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points (single values).
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r


def find_nearest_stations(stations_df: pd.DataFrame, 
                         target_lat: float, target_lon: float,
                         n_neighbors: int = 5,
                         lat_col: str = 'lat', lon_col: str = 'lon',
                         id_col: str = 'station_id') -> pd.DataFrame:
    """
    Find nearest stations to a target location.
    
    Args:
        stations_df: DataFrame with station coordinates
        target_lat: Target latitude
        target_lon: Target longitude
        n_neighbors: Number of nearest neighbors to find
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        id_col: Name of station ID column
        
    Returns:
        DataFrame with nearest stations and distances
    """
    # Calculate distances to all stations
    distances = haversine_np(
        stations_df[lon_col].values, stations_df[lat_col].values,
        target_lon, target_lat
    )
    
    # Create result DataFrame
    result_df = stations_df.copy()
    result_df['distance_km'] = distances
    
    # Sort by distance and take top n
    nearest = result_df.sort_values('distance_km').head(n_neighbors)
    
    return nearest.reset_index(drop=True)


def create_station_neighbor_matrix(stations_df: pd.DataFrame,
                                  n_neighbors: int = 5,
                                  lat_col: str = 'lat', lon_col: str = 'lon',
                                  id_col: str = 'station_id') -> pd.DataFrame:
    """
    Create a matrix of nearest neighbors for all stations.
    
    Args:
        stations_df: DataFrame with station coordinates
        n_neighbors: Number of neighbors to find for each station
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        id_col: Name of station ID column
        
    Returns:
        DataFrame with station neighbors
    """
    # Prepare coordinates for NearestNeighbors
    coords = stations_df[[lat_col, lon_col]].values
    
    # Create NearestNeighbors model with Haversine metric
    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,  # +1 because each station is its own neighbor
        metric='haversine',
        algorithm='ball_tree'
    )
    
    # Convert coordinates to radians for haversine
    coords_rad = np.radians(coords)
    nn_model.fit(coords_rad)
    
    # Find neighbors for all stations
    distances, indices = nn_model.kneighbors(coords_rad)
    
    # Convert distances back to kilometers
    distances_km = distances * 6371.0
    
    # Create result DataFrame
    neighbor_data = []
    
    for i, station_id in enumerate(stations_df[id_col]):
        # Skip the first neighbor (the station itself)
        for j in range(1, n_neighbors + 1):
            if j < len(indices[i]):
                neighbor_idx = indices[i][j]
                neighbor_id = stations_df.iloc[neighbor_idx][id_col]
                distance = distances_km[i][j]
                
                neighbor_data.append({
                    'station_id': station_id,
                    'neighbor_id': neighbor_id,
                    'neighbor_rank': j,
                    'distance_km': distance
                })
    
    return pd.DataFrame(neighbor_data)


def normalize_coordinates_zscore(df: pd.DataFrame, 
                                lat_col: str = 'lat', lon_col: str = 'lon') -> pd.DataFrame:
    """
    Normalize coordinates using z-score normalization.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with normalized coordinates added
    """
    df_norm = df.copy()
    
    # Calculate mean and standard deviation
    lat_mean = df[lat_col].mean()
    lat_std = df[lat_col].std()
    lon_mean = df[lon_col].mean()
    lon_std = df[lon_col].std()
    
    # Add normalized coordinates
    df_norm[f'{lat_col}_normalized'] = (df[lat_col] - lat_mean) / lat_std
    df_norm[f'{lon_col}_normalized'] = (df[lon_col] - lon_mean) / lon_std
    
    return df_norm


def normalize_coordinates_minmax(df: pd.DataFrame,
                                lat_col: str = 'lat', lon_col: str = 'lon') -> pd.DataFrame:
    """
    Normalize coordinates using min-max normalization.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with normalized coordinates added
    """
    df_norm = df.copy()
    
    # Calculate min and max
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    
    # Add normalized coordinates
    df_norm[f'{lat_col}_minmax'] = (df[lat_col] - lat_min) / (lat_max - lat_min)
    df_norm[f'{lon_col}_minmax'] = (df[lon_col] - lon_min) / (lon_max - lon_min)
    
    return df_norm


def create_fourier_features(df: pd.DataFrame, 
                           coordinate_cols: List[str],
                           k_values: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Create Fourier features for periodic coordinate encoding.
    
    Args:
        df: DataFrame with coordinate columns
        coordinate_cols: List of coordinate column names
        k_values: List of k values for Fourier transforms
        
    Returns:
        DataFrame with Fourier features added
    """
    df_fourier = df.copy()
    
    for col in coordinate_cols:
        if col in df.columns:
            for k in k_values:
                # Create sine and cosine features
                df_fourier[f'{col}_sin_{k}'] = np.sin(2 * np.pi * k * df[col])
                df_fourier[f'{col}_cos_{k}'] = np.cos(2 * np.pi * k * df[col])
    
    return df_fourier


def calculate_bounding_box(df: pd.DataFrame, 
                          lat_col: str = 'lat', lon_col: str = 'lon',
                          buffer_km: float = 1.0) -> dict:
    """
    Calculate bounding box for coordinates with optional buffer.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        buffer_km: Buffer distance in kilometers
        
    Returns:
        Dictionary with bounding box coordinates
    """
    # Get min/max coordinates
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
    min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
    
    # Convert buffer to degrees (approximate)
    lat_buffer = buffer_km / 111.32  # 1 degree lat ≈ 111.32 km
    lon_buffer = buffer_km / (111.32 * np.cos(np.radians(np.mean([min_lat, max_lat]))))
    
    return {
        'min_lat': min_lat - lat_buffer,
        'max_lat': max_lat + lat_buffer,
        'min_lon': min_lon - lon_buffer,
        'max_lon': max_lon + lon_buffer
    }


def point_in_bounding_box(lat: float, lon: float, bbox: dict) -> bool:
    """
    Check if a point is within a bounding box.
    
    Args:
        lat: Latitude of point
        lon: Longitude of point
        bbox: Bounding box dictionary
        
    Returns:
        True if point is within bounding box
    """
    return (bbox['min_lat'] <= lat <= bbox['max_lat'] and
            bbox['min_lon'] <= lon <= bbox['max_lon'])


def calculate_centroid(df: pd.DataFrame, 
                      lat_col: str = 'lat', lon_col: str = 'lon') -> Tuple[float, float]:
    """
    Calculate centroid (mean center) of coordinates.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        Tuple of (centroid_lat, centroid_lon)
    """
    centroid_lat = df[lat_col].mean()
    centroid_lon = df[lon_col].mean()
    
    return centroid_lat, centroid_lon


def calculate_weighted_centroid(df: pd.DataFrame,
                               lat_col: str = 'lat', lon_col: str = 'lon',
                               weight_col: str = 'weight') -> Tuple[float, float]:
    """
    Calculate weighted centroid of coordinates.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        weight_col: Name of weight column
        
    Returns:
        Tuple of (weighted_centroid_lat, weighted_centroid_lon)
    """
    total_weight = df[weight_col].sum()
    
    weighted_lat = (df[lat_col] * df[weight_col]).sum() / total_weight
    weighted_lon = (df[lon_col] * df[weight_col]).sum() / total_weight
    
    return weighted_lat, weighted_lon


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180 / np.pi


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing between two points.
    
    Args:
        lat1, lon1: Starting point coordinates
        lat2, lon2: Ending point coordinates
        
    Returns:
        Bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2) - 
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def calculate_safe_neighbor_limit(n_stations: int, requested_neighbors: int) -> int:
    """
    Calculate safe neighbor limit to avoid memory issues.
    
    Args:
        n_stations: Total number of stations
        requested_neighbors: Requested number of neighbors
        
    Returns:
        Safe number of neighbors to use
    """
    # Conservative limit to avoid memory issues
    max_neighbors = min(requested_neighbors, n_stations - 1, 10)
    
    if max_neighbors < requested_neighbors:
        print(f"  -> Limiting neighbors from {requested_neighbors} to {max_neighbors} (stations: {n_stations})")
    
    return max_neighbors 


