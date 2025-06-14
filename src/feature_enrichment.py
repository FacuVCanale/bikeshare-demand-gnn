"""
Feature enrichment and cleaning functions for EcoBici data.

This module contains functions for:
- Adding lat/lon coordinates to stations
- Normalizing coordinates  
- Correcting trip_dur_mean_last_DT
- Creating occurrence flags
- Station embeddings
"""

import polars as pl
import numpy as np
import math
from typing import Dict, List, Optional, Tuple


def create_station_metadata_enhanced(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract station metadata with lat/lon coordinates and additional cleaning.
    
    Args:
        df: DataFrame with trip data containing station information
        
    Returns:
        DataFrame with columns: station_id, lat, lon, station_name
    """
    print("Creating enhanced station metadata...")
    
    # Extract unique station information from origin stations
    origin_meta = (
        df.select([
            "id_estacion_origen",
            "nombre_estacion_origen", 
            "lat_estacion_origen",
            "long_estacion_origen"
        ])
        .unique("id_estacion_origen", keep="first")
        .rename({
            "id_estacion_origen": "station_id",
            "nombre_estacion_origen": "station_name",
            "lat_estacion_origen": "lat", 
            "long_estacion_origen": "lon"
        })
    )
    
    # Extract unique station information from destination stations
    dest_meta = (
        df.select([
            "id_estacion_destino",
            "nombre_estacion_destino",
            "lat_estacion_destino", 
            "long_estacion_destino"
        ])
        .unique("id_estacion_destino", keep="first")
        .rename({
            "id_estacion_destino": "station_id",
            "nombre_estacion_destino": "station_name",
            "lat_estacion_destino": "lat",
            "long_estacion_destino": "lon"
        })
    )
    
    # Combine and deduplicate
    meta_df = (
        pl.concat([origin_meta, dest_meta])
        .unique("station_id", keep="first")
        .drop_nulls(["lat", "lon"])
        .sort("station_id")
    )
    
    print(f"  -> Extracted metadata for {len(meta_df)} stations")
    return meta_df


def add_coordinates_to_features(df_feat: pl.DataFrame, df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Add lat/lon coordinates to feature dataframe by joining station metadata.
    
    Args:
        df_feat: Feature dataframe with station_id column
        df_raw: Raw trips dataframe with coordinate information
        
    Returns:
        Feature dataframe with added lat, lon columns
    """
    print("Adding coordinates to feature dataframe...")
    
    # Create station metadata
    meta_df = create_station_metadata_enhanced(df_raw)
    
    # Join coordinates to feature dataframe
    df_with_coords = df_feat.join(
        meta_df.select(["station_id", "lat", "lon"]), 
        on="station_id", 
        how="left"
    )
    
    # Check for missing coordinates
    missing_coords = df_with_coords.filter(pl.col("lat").is_null() | pl.col("lon").is_null()).height
    if missing_coords > 0:
        print(f"  -> Warning: {missing_coords} rows have missing coordinates")
    else:
        print(f"  -> Successfully added coordinates to all {df_with_coords.height:,} rows")
    
    return df_with_coords


def normalize_coordinates(df: pl.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pl.DataFrame:
    """
    Add normalized latitude and longitude coordinates (z-score normalization).
    
    Args:
        df: DataFrame containing latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with additional columns: lat_z, lon_z
    """
    print("Normalizing coordinates...")
    
    # Calculate statistics
    lat_mean = df[lat_col].mean()
    lat_std = df[lat_col].std()
    lon_mean = df[lon_col].mean() 
    lon_std = df[lon_col].std()
    
    # Add normalized coordinates
    df_norm = df.with_columns([
        ((pl.col(lat_col) - lat_mean) / lat_std).alias("lat_z"),
        ((pl.col(lon_col) - lon_mean) / lon_std).alias("lon_z")
    ])
    
    print(f"  -> Coordinate normalization stats:")
    print(f"     lat: mean={lat_mean:.6f}, std={lat_std:.6f}")
    print(f"     lon: mean={lon_mean:.6f}, std={lon_std:.6f}")
    
    return df_norm


def add_fourier_coordinates(df: pl.DataFrame, k_values: List[int] = [1, 2], 
                           lat_col: str = "lat", lon_col: str = "lon") -> pl.DataFrame:
    """
    Add fourier features for spatial periodicity encoding.
    
    Args:
        df: DataFrame with lat/lon coordinates
        k_values: List of k values for fourier encoding
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with additional fourier coordinate columns
    """
    print(f"Adding fourier coordinates with k_values={k_values}...")
    
    for k in k_values:
        df = df.with_columns([
            (pl.col(lat_col) * k).sin().alias(f"lat_sin_{k}"),
            (pl.col(lat_col) * k).cos().alias(f"lat_cos_{k}"),
            (pl.col(lon_col) * k).sin().alias(f"lon_sin_{k}"),
            (pl.col(lon_col) * k).cos().alias(f"lon_cos_{k}")
        ])
    
    print(f"  -> Added {len(k_values) * 4} fourier coordinate features")
    return df


def correct_trip_duration_mean(df: pl.DataFrame, 
                              duration_col: str = "trip_dur_mean_last_DT",
                              departure_col: str = "dep_last_DT") -> pl.DataFrame:
    """
    Correct trip_dur_mean_last_DT column and add duration validity flag.
    
    Rules:
    - When dep_last_DT == 0 or null -> set trip_dur_mean_last_DT to 0
    - Add flag dur_is_null = 1 if original value was null
    
    Args:
        df: DataFrame containing the duration and departure columns
        duration_col: Name of the trip duration mean column
        departure_col: Name of the departure count column
        
    Returns:
        DataFrame with corrected duration column and dur_is_null flag
    """
    print("Correcting trip duration mean...")
    
    # Create null flag before correction
    df = df.with_columns([
        pl.col(duration_col).is_null().cast(pl.UInt8).alias("dur_is_null")
    ])
    
    # Apply correction logic
    df = df.with_columns([
        pl.when((pl.col(departure_col) == 0) | pl.col(departure_col).is_null())
        .then(0.0)
        .otherwise(pl.col(duration_col).fill_null(0.0))
        .alias(duration_col)
    ])
    
    # Count corrections made
    null_count = df["dur_is_null"].sum()
    zero_dep_count = df.filter((pl.col(departure_col) == 0) | pl.col(departure_col).is_null()).height
    
    print(f"  -> Added dur_is_null flag ({null_count} null values detected)")
    print(f"  -> Corrected {zero_dep_count} cases where dep_last_DT was 0/null")
    
    return df


def create_occurrence_flags(df: pl.DataFrame,
                           departure_col: str = "dep_last_DT", 
                           arrival_col: str = "y_arrivals_next_DT") -> pl.DataFrame:
    """
    Create binary flags for departure and arrival occurrences.
    
    Args:
        df: DataFrame containing departure and arrival columns
        departure_col: Name of departure count column
        arrival_col: Name of arrival count column
        
    Returns:
        DataFrame with additional columns: has_dep, has_arr
    """
    print("Creating occurrence flags...")
    
    df = df.with_columns([
        pl.when(pl.col(departure_col) > 0)
        .then(1)
        .otherwise(0)
        .cast(pl.UInt8)
        .alias("has_dep"),
        
        pl.when(pl.col(arrival_col) > 0)
        .then(1)
        .otherwise(0)
        .cast(pl.UInt8)
        .alias("has_arr")
    ])
    
    has_dep_count = df["has_dep"].sum()
    has_arr_count = df["has_arr"].sum()
    total_rows = df.height
    
    print(f"  -> has_dep: {has_dep_count:,} / {total_rows:,} ({has_dep_count/total_rows*100:.1f}%)")
    print(f"  -> has_arr: {has_arr_count:,} / {total_rows:,} ({has_arr_count/total_rows*100:.1f}%)")
    
    return df


def create_station_embeddings_mapping(df: pl.DataFrame, 
                                     station_col: str = "station_id") -> Tuple[Dict[int, int], int]:
    """
    Create mapping from station_id to 0-based index for embeddings.
    
    Args:
        df: DataFrame containing station_id column
        station_col: Name of station ID column
        
    Returns:
        Tuple of (station_id_to_idx mapping, embedding_dim)
    """
    print("Creating station embeddings mapping...")
    
    # Get unique station IDs
    unique_stations = df[station_col].unique().sort().to_list()
    n_stations = len(unique_stations)
    
    # Calculate embedding dimension (sqrt rule)
    embedding_dim = int(math.sqrt(n_stations))
    
    # Create 0-based mapping
    station_id_to_idx = {station_id: idx for idx, station_id in enumerate(unique_stations)}
    
    print(f"  -> Found {n_stations} unique stations")
    print(f"  -> Recommended embedding dimension: {embedding_dim}")
    print(f"  -> Station ID range: {min(unique_stations)} to {max(unique_stations)}")
    
    return station_id_to_idx, embedding_dim


def add_station_embedding_index(df: pl.DataFrame, 
                               station_id_to_idx: Dict[int, int],
                               station_col: str = "station_id") -> pl.DataFrame:
    """
    Add 0-based station index column for embedding lookup.
    
    Args:
        df: DataFrame containing station_id column
        station_id_to_idx: Mapping from station_id to 0-based index
        station_col: Name of station ID column
        
    Returns:
        DataFrame with additional station_id_idx column
    """
    print("Adding station embedding index...")
    
    # Convert mapping to Polars-friendly format
    mapping_df = pl.DataFrame({
        station_col: list(station_id_to_idx.keys()),
        "station_id_idx": list(station_id_to_idx.values())
    })
    
    # Join to add index
    df = df.join(mapping_df, on=station_col, how="left")
    
    # Check for any missing mappings
    missing_count = df["station_id_idx"].null_count()
    if missing_count > 0:
        print(f"  -> Warning: {missing_count} rows have missing station_id_idx")
    else:
        print(f"  -> Successfully mapped all {df.height:,} rows")
    
    return df


def apply_all_enrichments(df: pl.DataFrame, 
                         df_raw: Optional[pl.DataFrame] = None,
                         add_coords: bool = True,
                         add_fourier: bool = True,
                         k_values: List[int] = [1, 2]) -> Tuple[pl.DataFrame, Dict[int, int], int]:
    """
    Apply all enrichment functions in sequence.
    
    Args:
        df: Input DataFrame with feature engineered data
        df_raw: Raw trips dataframe (needed for coordinates)
        add_coords: Whether to add coordinate normalization
        add_fourier: Whether to add fourier coordinate features  
        k_values: K values for fourier encoding
        
    Returns:
        Tuple of (enriched_df, station_id_to_idx, embedding_dim)
    """
    print("Applying all data enrichments...")
    print("=" * 50)
    
    # Step 1: Correct trip duration
    df = correct_trip_duration_mean(df)
    
    # Step 2: Create occurrence flags
    df = create_occurrence_flags(df)
    
    # Step 3: Station embeddings mapping
    station_id_to_idx, embedding_dim = create_station_embeddings_mapping(df)
    df = add_station_embedding_index(df, station_id_to_idx)
    
    # Step 4: Add coordinates if requested and available
    if add_coords:
        # Check if we need to add coordinate data
        if "lat" not in df.columns or "lon" not in df.columns:
            if df_raw is not None:
                print("  -> Adding coordinates from raw data...")
                df = add_coordinates_to_features(df, df_raw)
            else:
                print("  -> Coordinates not found and no raw data provided, skipping coordinate enrichment")
                add_coords = False
        else:
            print("  -> Coordinates already present in dataframe")
            
        if add_coords:
            df = normalize_coordinates(df)
            if add_fourier:
                df = add_fourier_coordinates(df, k_values=k_values)
    
    print("=" * 50)
    print("✅ All enrichments completed successfully!")
    print(f"   Final shape: {df.shape}")
    print(f"   Station embedding dimension: {embedding_dim}")
    
    return df, station_id_to_idx, embedding_dim 