"""
Data preparation utilities for clustering pipeline.

This module handles:
- Loading and merging datasets
- Temporal splitting without data leakage
- Data validation and quality checks
- Memory-efficient data processing
"""

import polars as pl
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime
import numpy as np


class DataPreparator:
    """Handles data loading, merging, and temporal splitting for clustering pipeline."""
    
    def __init__(self, data_dir: str = "data", log_level: str = "INFO"):
        """Initialize data preparator.
        
        Args:
            data_dir: Root data directory path
            log_level: Logging level
        """
        self.data_dir = Path(data_dir)
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_datasets(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load all required datasets with temporal filtering to exclude September 2024 onward.
        
        August 2024 data is included, but September 2024 onward is filtered out as reserved data.
        
        Returns:
            Tuple of (trips_with_weather, trips, users) DataFrames
        """
        self.logger.info("Loading datasets...")
        
        # cutoff date to exclude september 2024 onward (august 2024 inclusive is allowed)
        cutoff_date = pl.lit("2024-09-01").str.strptime(pl.Date, "%Y-%m-%d")
        
        # load main dataset
        trips_weather_path = self.data_dir / "processed" / "trips_with_weather.parquet"
        self.logger.info(f"Loading trips_with_weather from {trips_weather_path}")
        trips_with_weather = pl.read_parquet(trips_weather_path)
        
        # filter out august 2024 onward
        trips_with_weather_filtered = trips_with_weather.filter(
            pl.col("fecha_origen_recorrido").dt.date() < cutoff_date
        )
        
        filtered_count = trips_with_weather.height - trips_with_weather_filtered.height
        if filtered_count > 0:
            self.logger.info(f"  -> Filtered out {filtered_count:,} trips from September 2024 onward")
        
        self.logger.info(f"  -> trips_with_weather shape after filtering: {trips_with_weather_filtered.shape}")
        
        # load complete trips dataset
        trips_path = self.data_dir / "raw" / "combined" / "trips.parquet"
        self.logger.info(f"Loading trips from {trips_path}")
        trips = pl.read_parquet(trips_path)
        
        # filter out august 2024 onward from trips as well
        trips_filtered = trips.filter(
            pl.col("fecha_origen_recorrido").dt.date() < cutoff_date
        )
        
        filtered_count_trips = trips.height - trips_filtered.height
        if filtered_count_trips > 0:
            self.logger.info(f"  -> Filtered out {filtered_count_trips:,} trips from September 2024 onward")
        
        self.logger.info(f"  -> trips shape after filtering: {trips_filtered.shape}")
        
        # load users dataset
        users_path = self.data_dir / "raw" / "combined" / "users.parquet"
        self.logger.info(f"Loading users from {users_path}")
        users = pl.read_parquet(users_path)
        self.logger.info(f"  -> users shape: {users.shape}")
        
        return trips_with_weather_filtered, trips_filtered, users
        
    def merge_datasets(self, trips_with_weather: pl.DataFrame, 
                      trips: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
        """Merge all datasets with quality flags.
        
        Args:
            trips_with_weather: Main dataset with weather data
            trips: Complete trips dataset
            users: Users information dataset
            
        Returns:
            Merged DataFrame with quality flags
        """
        self.logger.info("Merging datasets...")
        
        # start with main dataset as base
        df_base = trips_with_weather.clone()
        self.logger.info(f"Base dataset shape: {df_base.shape}")
        
        # create user matching flag by joining with users
        self.logger.info("Joining with users dataset...")
        df_with_users = df_base.join(
            users.select([
                "id_usuario", 
                pl.col("genero_usuario").alias("user_gender_full"),
                pl.col("edad_usuario").alias("user_age")
            ]),
            on="id_usuario",
            how="left"
        )
        
        # add user matching flag
        df_with_users = df_with_users.with_columns([
            pl.col("user_gender_full").is_not_null().alias("user_matched"),
            pl.col("user_age").is_not_null().alias("age_available")
        ])
        
        # log matching statistics
        total_trips = df_with_users.height
        user_matched = df_with_users.filter(pl.col("user_matched")).height
        age_available = df_with_users.filter(pl.col("age_available")).height
        
        self.logger.info(f"User matching stats:")
        self.logger.info(f"  -> Total trips: {total_trips:,}")
        self.logger.info(f"  -> User matched: {user_matched:,} ({user_matched/total_trips*100:.1f}%)")
        self.logger.info(f"  -> Age available: {age_available:,} ({age_available/total_trips*100:.1f}%)")
        
        # add weather matching flag (should be 100% for trips_with_weather)
        df_with_users = df_with_users.with_columns([
            pl.col("weather_matched").alias("weather_available")
        ])
        
        weather_available = df_with_users.filter(pl.col("weather_available")).height
        self.logger.info(f"  -> Weather available: {weather_available:,} ({weather_available/total_trips*100:.1f}%)")
        
        # check for missing trips in main dataset by comparing with complete trips
        complete_trip_ids = set(trips.select("id_recorrido").to_series().to_list())
        main_trip_ids = set(df_with_users.select("id_recorrido").to_series().to_list())
        
        missing_from_main = len(complete_trip_ids - main_trip_ids)
        self.logger.info(f"  -> Trips missing from main dataset: {missing_from_main:,}")
        
        # add flag for completeness
        df_with_users = df_with_users.with_columns([
            pl.lit(True).alias("in_main_dataset")  # all rows in main dataset
        ])
        
        self.logger.info(f"Final merged dataset shape: {df_with_users.shape}")
        return df_with_users
        
    def create_temporal_splits(self, df: pl.DataFrame, 
                             train_end_date: str = "2023-01-01",
                             val_end_date: str = "2023-07-01") -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Create temporal train/validation/test splits without data leakage.
        
        Args:
            df: Input DataFrame with datetime column
            train_end_date: End date for training set (format: "YYYY-MM-DD")
            val_end_date: End date for validation set (format: "YYYY-MM-DD")
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Creating temporal splits...")
        
        # ensure datetime column exists
        if "fecha_origen_recorrido" in df.columns:
            date_col = "fecha_origen_recorrido"
        else:
            raise ValueError("No datetime column found in DataFrame")
            
        # convert string dates to datetime
        train_cutoff = pl.lit(train_end_date).str.strptime(pl.Date, "%Y-%m-%d")
        val_cutoff = pl.lit(val_end_date).str.strptime(pl.Date, "%Y-%m-%d")
        
        # create splits
        df_train = df.filter(pl.col(date_col).dt.date() < train_cutoff)
        df_val = df.filter(
            (pl.col(date_col).dt.date() >= train_cutoff) & 
            (pl.col(date_col).dt.date() < val_cutoff)
        )
        df_test = df.filter(pl.col(date_col).dt.date() >= val_cutoff)
        
        # log split statistics
        self.logger.info(f"Temporal split results:")
        self.logger.info(f"  -> Train: {df_train.shape[0]:,} rows (< {train_end_date})")
        self.logger.info(f"  -> Val: {df_val.shape[0]:,} rows ({train_end_date} to {val_end_date})")
        self.logger.info(f"  -> Test: {df_test.shape[0]:,} rows (>= {val_end_date})")
        
        # validate no leakage
        train_max_date = df_train.select(pl.col(date_col).dt.date().max()).item()
        val_min_date = df_val.select(pl.col(date_col).dt.date().min()).item()
        val_max_date = df_val.select(pl.col(date_col).dt.date().max()).item()
        test_min_date = df_test.select(pl.col(date_col).dt.date().min()).item()
        
        self.logger.info(f"Date ranges:")
        self.logger.info(f"  -> Train: ends {train_max_date}")
        self.logger.info(f"  -> Val: {val_min_date} to {val_max_date}")
        self.logger.info(f"  -> Test: starts {test_min_date}")
        
        assert train_max_date < val_min_date, "Data leakage detected: train max >= val min"
        assert val_max_date < test_min_date, "Data leakage detected: val max >= test min"
        
        self.logger.info("✓ No data leakage detected in temporal splits")
        
        return df_train, df_val, df_test
        
    def extract_station_metadata(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract unique station metadata from trips data.
        
        Args:
            df: DataFrame with trip data
            
        Returns:
            DataFrame with station metadata (id, name, lat, lon)
        """
        self.logger.info("Extracting station metadata...")
        
        # extract from origin stations
        origin_stations = df.select([
            pl.col("id_estacion_origen").alias("station_id"),
            pl.col("nombre_estacion_origen").alias("station_name"),
            pl.col("lat_estacion_origen").alias("lat"),
            pl.col("long_estacion_origen").alias("lon")
        ]).unique(subset=["station_id"], keep="first")
        
        # extract from destination stations
        dest_stations = df.select([
            pl.col("id_estacion_destino").alias("station_id"),
            pl.col("nombre_estacion_destino").alias("station_name"),
            pl.col("lat_estacion_destino").alias("lat"),
            pl.col("long_estacion_destino").alias("lon")
        ]).unique(subset=["station_id"], keep="first")
        
        # combine and deduplicate
        stations = pl.concat([origin_stations, dest_stations]).unique(
            subset=["station_id"], keep="first"
        ).drop_nulls(subset=["lat", "lon"]).sort("station_id")
        
        self.logger.info(f"  -> Extracted {stations.height} unique stations")
        self.logger.info(f"  -> Coordinate ranges:")
        self.logger.info(f"     Lat: {stations['lat'].min():.6f} to {stations['lat'].max():.6f}")
        self.logger.info(f"     Lon: {stations['lon'].min():.6f} to {stations['lon'].max():.6f}")
        
        return stations
        
    def validate_data_quality(self, df: pl.DataFrame) -> Dict[str, int]:
        """Validate data quality and return statistics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality statistics
        """
        self.logger.info("Validating data quality...")
        
        stats = {}
        
        # basic stats
        stats['total_rows'] = df.height
        stats['total_columns'] = df.width
        
        # null counts for key columns
        key_columns = [
            'id_recorrido', 'id_estacion_origen', 'id_estacion_destino',
            'lat_estacion_origen', 'long_estacion_origen',
            'lat_estacion_destino', 'long_estacion_destino',
            'id_usuario', 'modelo_bicicleta', 'genero'
        ]
        
        for col in key_columns:
            if col in df.columns:
                null_count = df.select(pl.col(col).is_null().sum()).item()
                stats[f'null_{col}'] = null_count
                
        # coordinate validation
        if 'lat_estacion_origen' in df.columns:
            # buenos aires approximate bounds
            buenos_aires_lat_min, buenos_aires_lat_max = -35.0, -34.0
            buenos_aires_lon_min, buenos_aires_lon_max = -59.0, -58.0
            
            out_of_bounds = df.filter(
                (pl.col('lat_estacion_origen') < buenos_aires_lat_min) |
                (pl.col('lat_estacion_origen') > buenos_aires_lat_max) |
                (pl.col('long_estacion_origen') < buenos_aires_lon_min) |
                (pl.col('long_estacion_origen') > buenos_aires_lon_max)
            ).height
            
            stats['coordinates_out_of_bounds'] = out_of_bounds
            
        # log quality stats
        self.logger.info("Data quality summary:")
        for key, value in stats.items():
            if isinstance(value, int):
                if 'null' in key and stats['total_rows'] > 0:
                    pct = value / stats['total_rows'] * 100
                    self.logger.info(f"  -> {key}: {value:,} ({pct:.1f}%)")
                else:
                    self.logger.info(f"  -> {key}: {value:,}")
                    
        return stats 