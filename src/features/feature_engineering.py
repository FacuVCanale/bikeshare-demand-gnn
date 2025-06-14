"""
Feature engineering functions for EcoBici data.

This module contains functions for:
- Station metadata enhancement
- Coordinate normalization and spatial features
- Trip duration corrections
- Occurrence flags and embeddings
- Feature enrichment pipelines
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
    
    # extract unique station information from origin stations
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
    
    # extract unique station information from destination stations
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
    
    # combine and deduplicate
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
    
    # create station metadata
    meta_df = create_station_metadata_enhanced(df_raw)
    
    # join coordinates to feature dataframe
    df_with_coords = df_feat.join(
        meta_df.select(["station_id", "lat", "lon"]), 
        on="station_id", 
        how="left"
    )
    
    # check for missing coordinates
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
    
    # calculate statistics
    lat_mean = df[lat_col].mean()
    lat_std = df[lat_col].std()
    lon_mean = df[lon_col].mean() 
    lon_std = df[lon_col].std()
    
    # add normalized coordinates
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
    
    # create null flag before correction
    df = df.with_columns([
        pl.col(duration_col).is_null().cast(pl.UInt8).alias("dur_is_null")
    ])
    
    # apply correction logic
    df = df.with_columns([
        pl.when((pl.col(departure_col) == 0) | pl.col(departure_col).is_null())
        .then(0.0)
        .otherwise(pl.col(duration_col).fill_null(0.0))
        .alias(duration_col)
    ])
    
    corrected_count = df.filter(pl.col("dur_is_null") == 1).height
    print(f"  -> Corrected {corrected_count} null duration values")
    
    return df


def create_occurrence_flags(df: pl.DataFrame,
                           departure_col: str = "dep_last_DT", 
                           arrival_col: str = "y_arrivals_next_DT") -> pl.DataFrame:
    """
    Create binary occurrence flags for departures and arrivals.
    
    Args:
        df: DataFrame with departure and arrival columns
        departure_col: Name of the departure count column
        arrival_col: Name of the arrival count column
        
    Returns:
        DataFrame with additional columns: dep_occurred, arr_occurred
    """
    print("Creating occurrence flags...")
    
    df = df.with_columns([
        (pl.col(departure_col) > 0).cast(pl.UInt8).alias("dep_occurred"),
        (pl.col(arrival_col) > 0).cast(pl.UInt8).alias("arr_occurred")
    ])
    
    dep_occurred_count = df.filter(pl.col("dep_occurred") == 1).height
    arr_occurred_count = df.filter(pl.col("arr_occurred") == 1).height
    total_rows = df.height
    
    print(f"  -> Departures occurred: {dep_occurred_count:,}/{total_rows:,} ({dep_occurred_count/total_rows*100:.1f}%)")
    print(f"  -> Arrivals occurred: {arr_occurred_count:,}/{total_rows:,} ({arr_occurred_count/total_rows*100:.1f}%)")
    
    return df


def create_station_embeddings_mapping(df: pl.DataFrame, 
                                     station_col: str = "station_id") -> Tuple[Dict[int, int], int]:
    """
    Create mapping from station_id to embedding index for neural networks.
    
    Args:
        df: DataFrame containing station_id column
        station_col: Name of the station ID column
        
    Returns:
        Tuple of (station_id_to_index_dict, num_stations)
    """
    print("Creating station embeddings mapping...")
    
    unique_stations = sorted(df[station_col].unique().to_list())
    station_id_to_idx = {station_id: idx for idx, station_id in enumerate(unique_stations)}
    
    print(f"  -> Created mapping for {len(unique_stations)} unique stations")
    
    return station_id_to_idx, len(unique_stations)


def add_station_embedding_index(df: pl.DataFrame, 
                               station_id_to_idx: Dict[int, int],
                               station_col: str = "station_id") -> pl.DataFrame:
    """
    Add station embedding index column based on station_id mapping.
    
    Args:
        df: DataFrame with station_id column
        station_id_to_idx: Mapping from station_id to embedding index
        station_col: Name of the station ID column
        
    Returns:
        DataFrame with added station_idx column
    """
    print("Adding station embedding indices...")
    
    # convert mapping to polars expression
    mapping_expr = pl.col(station_col).map_dict(station_id_to_idx)
    
    df = df.with_columns([
        mapping_expr.alias("station_idx")
    ])
    
    print(f"  -> Added embedding indices for {df.height:,} rows")
    
    return df


def apply_all_enrichments(df: pl.DataFrame, 
                         df_raw: Optional[pl.DataFrame] = None,
                         add_coords: bool = True,
                         add_fourier: bool = True,
                         k_values: List[int] = [1, 2]) -> Tuple[pl.DataFrame, Dict[int, int], int]:
    """
    Apply all feature enrichments in the correct order.
    
    Args:
        df: Feature dataframe to enrich
        df_raw: Raw trips dataframe for coordinate extraction (required if add_coords=True)
        add_coords: Whether to add coordinate features
        add_fourier: Whether to add fourier coordinate features
        k_values: K values for fourier encoding
        
    Returns:
        Tuple of (enriched_dataframe, station_id_to_idx_mapping, num_stations)
    """
    print("🚀 APPLYING ALL FEATURE ENRICHMENTS")
    print("=" * 50)
    
    # start with input dataframe
    df_enriched = df.clone()
    
    # 1. add coordinates if requested and raw data provided
    if add_coords and df_raw is not None:
        df_enriched = add_coordinates_to_features(df_enriched, df_raw)
        
        # normalize coordinates
        df_enriched = normalize_coordinates(df_enriched)
        
        # add fourier features if requested
        if add_fourier:
            df_enriched = add_fourier_coordinates(df_enriched, k_values=k_values)
    
    # 2. correct trip duration if column exists
    if "trip_dur_mean_last_DT" in df_enriched.columns:
        df_enriched = correct_trip_duration_mean(df_enriched)
    
    # 3. create occurrence flags if relevant columns exist
    if "dep_last_DT" in df_enriched.columns and "y_arrivals_next_DT" in df_enriched.columns:
        df_enriched = create_occurrence_flags(df_enriched)
    
    # 4. create station embeddings mapping
    station_id_to_idx, num_stations = create_station_embeddings_mapping(df_enriched)
    df_enriched = add_station_embedding_index(df_enriched, station_id_to_idx)
    
    print("=" * 50)
    print(f"✅ ENRICHMENT COMPLETE: {df.height:,} → {df_enriched.height:,} rows, {len(df.columns)} → {len(df_enriched.columns)} columns")
    
    return df_enriched, station_id_to_idx, num_stations 