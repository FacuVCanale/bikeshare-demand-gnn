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
    mapping_expr = pl.col(station_col).replace(station_id_to_idx, default=None)
    
    df = df.with_columns([
        mapping_expr.alias("station_idx")
    ])
    
    print(f"  -> Added embedding indices for {df.height:,} rows")
    
    return df


def apply_all_enrichments(df_train: pl.DataFrame, 
                                   df_val: pl.DataFrame,
                                   df_test: pl.DataFrame,
                                   df_raw: Optional[pl.DataFrame] = None,
                                   add_coords: bool = True,
                                   add_fourier: bool = True,
                                   k_values: List[int] = [1, 2]) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[int, int], int]:
    """
    Apply all feature enrichments to train/val/test splits without data leakage.
    
    Computes statistics (means, normalization parameters, etc.) only on training data
    and applies them to validation and test sets to prevent data leakage.
    
    Args:
        df_train: Training feature dataframe
        df_val: Validation feature dataframe  
        df_test: Test feature dataframe
        df_raw: Raw trips dataframe for coordinate extraction (required if add_coords=True)
        add_coords: Whether to add coordinate features
        add_fourier: Whether to add fourier coordinate features
        k_values: K values for fourier encoding
        
    Returns:
        Tuple of (train_enriched, val_enriched, test_enriched, station_id_to_idx_mapping, num_stations)
    """
    print("🚀 APPLYING ALL FEATURE ENRICHMENTS WITHOUT DATA LEAKAGE")
    print("=" * 60)
    print(f"Train: {df_train.shape[0]:,} rows")
    print(f"Val: {df_val.shape[0]:,} rows") 
    print(f"Test: {df_test.shape[0]:,} rows")
    print("=" * 60)
    
    # 1. COORDINATES - NO LEAKAGE (station coordinates are fixed)
    if add_coords and df_raw is not None:
        print("Adding coordinates to all splits...")
        
        # Create station metadata (this is fixed data, no leakage)
        meta_df = create_station_metadata_enhanced(df_raw)
        
        # Add coordinates to all splits
        df_train = df_train.join(meta_df.select(["station_id", "lat", "lon"]), on="station_id", how="left")
        df_val = df_val.join(meta_df.select(["station_id", "lat", "lon"]), on="station_id", how="left")
        df_test = df_test.join(meta_df.select(["station_id", "lat", "lon"]), on="station_id", how="left")
        
        # 2. NORMALIZE COORDINATES - COMPUTE STATS ONLY ON TRAIN
        print("Normalizing coordinates using TRAIN statistics only...")
        
        # Compute normalization stats on train only
        lat_mean = df_train["lat"].mean()
        lat_std = df_train["lat"].std()
        lon_mean = df_train["lon"].mean() 
        lon_std = df_train["lon"].std()
        
        print(f"  -> Normalization stats (from TRAIN only):")
        print(f"     lat: mean={lat_mean:.6f}, std={lat_std:.6f}")
        print(f"     lon: mean={lon_mean:.6f}, std={lon_std:.6f}")
        
        # Apply normalization to all splits using train stats
        for df_split, split_name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
            locals()[f"df_{split_name}"] = df_split.with_columns([
                ((pl.col("lat") - lat_mean) / lat_std).alias("lat_z"),
                ((pl.col("lon") - lon_mean) / lon_std).alias("lon_z")
            ])
        
        # Extract updated dataframes from locals
        df_train = locals()["df_train"]
        df_val = locals()["df_val"] 
        df_test = locals()["df_test"]
        
        # 3. ADD FOURIER FEATURES - NO LEAKAGE (deterministic transformation)
        if add_fourier:
            print(f"Adding fourier coordinates with k_values={k_values}...")
            
            for k in k_values:
                for df_split, split_name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
                    locals()[f"df_{split_name}"] = df_split.with_columns([
                        (pl.col("lat") * k * 2 * math.pi).sin().alias(f"lat_sin_{k}"),
                        (pl.col("lat") * k * 2 * math.pi).cos().alias(f"lat_cos_{k}"),
                        (pl.col("lon") * k * 2 * math.pi).sin().alias(f"lon_sin_{k}"),
                        (pl.col("lon") * k * 2 * math.pi).cos().alias(f"lon_cos_{k}")
                    ])
            
            # Extract updated dataframes from locals
            df_train = locals()["df_train"]
            df_val = locals()["df_val"]
            df_test = locals()["df_test"]
            
            print(f"  -> Added {len(k_values) * 4} fourier coordinate features")
    
    # 4. CORRECT TRIP DURATION - COMPUTE REPLACEMENT VALUE FROM TRAIN ONLY
    if "trip_dur_mean_last_DT" in df_train.columns:
        print("Correcting trip duration using TRAIN statistics only...")
        
        # Compute replacement value from train only
        train_mean_duration = df_train.filter(pl.col("trip_dur_mean_last_DT") > 0)["trip_dur_mean_last_DT"].mean()
        
        if train_mean_duration is not None:
            print(f"  -> Using train mean duration: {train_mean_duration:.1f} seconds")
            
            # Apply correction to all splits using train-derived value
            df_train = df_train.with_columns([
                pl.col("trip_dur_mean_last_DT").fill_null(train_mean_duration),
                pl.col("trip_dur_mean_last_DT").is_null().alias("dur_is_null")
            ])
            
            df_val = df_val.with_columns([
                pl.col("trip_dur_mean_last_DT").fill_null(train_mean_duration),
                pl.col("trip_dur_mean_last_DT").is_null().alias("dur_is_null")
            ])
            
            df_test = df_test.with_columns([
                pl.col("trip_dur_mean_last_DT").fill_null(train_mean_duration),
                pl.col("trip_dur_mean_last_DT").is_null().alias("dur_is_null")
            ])
            
            corrected_train = df_train["dur_is_null"].sum()
            corrected_val = df_val["dur_is_null"].sum()
            corrected_test = df_test["dur_is_null"].sum()
            
            print(f"  -> Corrected nulls: Train={corrected_train:,}, Val={corrected_val:,}, Test={corrected_test:,}")
        else:
            print("  -> No valid durations found in train, adding null flags only")
            for df_split, split_name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
                locals()[f"df_{split_name}"] = df_split.with_columns([
                    pl.col("trip_dur_mean_last_DT").is_null().alias("dur_is_null")
                ])
            df_train = locals()["df_train"]
            df_val = locals()["df_val"]
            df_test = locals()["df_test"]
    
    # 5. CREATE OCCURRENCE FLAGS - NO LEAKAGE (simple binary flags)
    if "dep_last_DT" in df_train.columns and "y_arrivals_next_DT" in df_train.columns:
        print("Creating occurrence flags...")
        
        for df_split, split_name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
            locals()[f"df_{split_name}"] = df_split.with_columns([
                (pl.col("dep_last_DT") > 0).alias("dep_occurred"),
                (pl.col("arr_last_DT") > 0).alias("arr_occurred")
            ])
        
        df_train = locals()["df_train"]
        df_val = locals()["df_val"]
        df_test = locals()["df_test"]
        
        print(f"  -> Departures occurred: Train={df_train['dep_occurred'].sum():,}, Val={df_val['dep_occurred'].sum():,}, Test={df_test['dep_occurred'].sum():,}")
        print(f"  -> Arrivals occurred: Train={df_train['arr_occurred'].sum():,}, Val={df_val['arr_occurred'].sum():,}, Test={df_test['arr_occurred'].sum():,}")
    
    # 6. CREATE STATION EMBEDDINGS - NO LEAKAGE (uses all station IDs, but creates consistent mapping)
    print("Creating station embeddings mapping...")
    
    # Collect all unique station IDs from all splits (this is okay, it's just creating a consistent mapping)
    all_station_ids = pl.concat([
        df_train.select("station_id"),
        df_val.select("station_id"), 
        df_test.select("station_id")
    ]).unique().sort("station_id")["station_id"].to_list()
    
    station_id_to_idx = {station_id: idx for idx, station_id in enumerate(all_station_ids)}
    num_stations = len(all_station_ids)
    
    print(f"  -> Created mapping for {num_stations} unique stations")
    
    # Add embedding indices to all splits
    for df_split, split_name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        locals()[f"df_{split_name}"] = df_split.with_columns([
            pl.col("station_id").map_elements(lambda x: station_id_to_idx[x], return_dtype=pl.Int32).alias("station_idx")
        ])
    
    df_train = locals()["df_train"]
    df_val = locals()["df_val"]
    df_test = locals()["df_test"]
    
    # Calculate new column count (simplified approach)
    base_cols = ["lat", "lon", "lat_z", "lon_z", "dur_is_null", "dep_occurred", "arr_occurred", "station_idx"]
    fourier_cols = [f"{coord}_{trig}_{k}" for k in k_values for coord in ["lat", "lon"] for trig in ["sin", "cos"]]
    new_cols = base_cols + fourier_cols
    
    print("=" * 60)
    print(f"✅ ENRICHMENT COMPLETE (NO DATA LEAKAGE):")
    print(f"   Train: {df_train.shape}")
    print(f"   Val: {df_val.shape}")
    print(f"   Test: {df_test.shape}")
    print(f"   Stations: {num_stations}")
    print(f"   Statistics computed ONLY on training data")
    
    return df_train, df_val, df_test, station_id_to_idx, num_stations 