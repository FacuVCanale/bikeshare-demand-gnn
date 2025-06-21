"""
Cluster feature generation for internal vs external trips.

This module handles:
- Generation of cluster-level features aggregated by time
- Internal vs external trip classification
- User demographics aggregation by cluster and trip type
- Age bins (young/adult/senior) instead of age statistics to prevent data leakage
- Bike model distribution features
- Ghost trip handling
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import timedelta
from pathlib import Path


class ClusterFeatureGenerator:
    """Generates cluster-level features with internal/external trip distinction."""
    
    def __init__(self, dt_minutes: int = 30, log_level: str = "INFO"):
        """Initialize cluster feature generator.
        
        Args:
            dt_minutes: Time interval in minutes for aggregation
            log_level: Logging level
        """
        self.dt_minutes = dt_minutes
        self.setup_logging(log_level)
        
        # age bins to prevent data leakage - fixed ranges
        self.age_bins = {
            'young': (0, 25),      # 0-25 years old
            'adult': (26, 55),     # 26-55 years old  
            'senior': (56, 120)    # 56+ years old
        }
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def classify_trips(self, trips_df: pl.DataFrame, 
                      station_cluster_mapping: Dict[int, int]) -> pl.DataFrame:
        """Classify trips as internal or external to clusters.
        
        Args:
            trips_df: DataFrame with trip data
            station_cluster_mapping: Mapping from station_id to cluster_id
            
        Returns:
            DataFrame with trip classification
        """
        self.logger.info("Classifying trips as internal/external to clusters...")
        
        # add cluster information for origin and destination stations
        origin_clusters = []
        dest_clusters = []
        
        for row in trips_df.iter_rows():
            origin_station = row[trips_df.columns.index('id_estacion_origen')]
            dest_station = row[trips_df.columns.index('id_estacion_destino')]
            
            origin_cluster = station_cluster_mapping.get(origin_station, -1)
            dest_cluster = station_cluster_mapping.get(dest_station, -1)
            
            origin_clusters.append(origin_cluster)
            dest_clusters.append(dest_cluster)
            
        # add cluster columns
        trips_with_clusters = trips_df.with_columns([
            pl.Series("origin_cluster_id", origin_clusters),
            pl.Series("dest_cluster_id", dest_clusters)
        ])
        
        # classify trips
        trips_classified = trips_with_clusters.with_columns([
            # internal trip: same cluster for origin and destination (and not -1)
            ((pl.col("origin_cluster_id") == pl.col("dest_cluster_id")) & 
             (pl.col("origin_cluster_id") != -1)).alias("is_internal_trip"),
            
            # external trip: different clusters or one cluster is -1 (unmapped station)
            ((pl.col("origin_cluster_id") != pl.col("dest_cluster_id")) | 
             (pl.col("origin_cluster_id") == -1) | 
             (pl.col("dest_cluster_id") == -1)).alias("is_external_trip"),
             
            # ghost trip flag: internal trip within same cluster
            ((pl.col("origin_cluster_id") == pl.col("dest_cluster_id")) & 
             (pl.col("origin_cluster_id") != -1)).alias("is_ghost_trip"),
             
            # unmapped station flag
            ((pl.col("origin_cluster_id") == -1) | 
             (pl.col("dest_cluster_id") == -1)).alias("has_unmapped_station")
        ])
        
        # add age bins to prevent data leakage
        trips_classified = self.add_age_bins(trips_classified)
        
        # log classification statistics
        total_trips = trips_classified.height
        internal_trips = trips_classified.filter(pl.col("is_internal_trip")).height
        external_trips = trips_classified.filter(pl.col("is_external_trip")).height
        ghost_trips = trips_classified.filter(pl.col("is_ghost_trip")).height
        unmapped_trips = trips_classified.filter(pl.col("has_unmapped_station")).height
        
        self.logger.info(f"Trip classification results:")
        self.logger.info(f"  -> Total trips: {total_trips:,}")
        self.logger.info(f"  -> Internal trips: {internal_trips:,} ({internal_trips/total_trips*100:.1f}%)")
        self.logger.info(f"  -> External trips: {external_trips:,} ({external_trips/total_trips*100:.1f}%)")
        self.logger.info(f"  -> Ghost trips: {ghost_trips:,} ({ghost_trips/total_trips*100:.1f}%)")
        self.logger.info(f"  -> Trips with unmapped stations: {unmapped_trips:,} ({unmapped_trips/total_trips*100:.1f}%)")
        
        return trips_classified
        
    def add_age_bins(self, trips_df: pl.DataFrame) -> pl.DataFrame:
        """Add age bin columns to prevent data leakage.
        
        Args:
            trips_df: DataFrame with user_age column
            
        Returns:
            DataFrame with age bin columns
        """
        self.logger.info("Adding age bins (young/adult/senior)...")
        
        # create age bin flags using fixed ranges
        trips_with_age_bins = trips_df.with_columns([
            # young: 0-25 years
            ((pl.col("user_age") >= self.age_bins['young'][0]) & 
             (pl.col("user_age") <= self.age_bins['young'][1])).alias("is_young"),
             
            # adult: 26-55 years  
            ((pl.col("user_age") >= self.age_bins['adult'][0]) & 
             (pl.col("user_age") <= self.age_bins['adult'][1])).alias("is_adult"),
             
            # senior: 56+ years
            ((pl.col("user_age") >= self.age_bins['senior'][0]) & 
             (pl.col("user_age") <= self.age_bins['senior'][1])).alias("is_senior"),
             
            # age available flag
            pl.col("user_age").is_not_null().alias("has_age_data")
        ])
        
        # log age bin distribution
        if "user_age" in trips_df.columns:
            young_count = trips_with_age_bins.filter(pl.col("is_young")).height
            adult_count = trips_with_age_bins.filter(pl.col("is_adult")).height  
            senior_count = trips_with_age_bins.filter(pl.col("is_senior")).height
            has_age_count = trips_with_age_bins.filter(pl.col("has_age_data")).height
            
            self.logger.info(f"  -> Age distribution:")
            self.logger.info(f"     Young (0-25): {young_count:,}")
            self.logger.info(f"     Adult (26-55): {adult_count:,}")
            self.logger.info(f"     Senior (56+): {senior_count:,}")
            self.logger.info(f"     With age data: {has_age_count:,}")
            
        return trips_with_age_bins
        
    def create_time_skeleton(self, trips_df: pl.DataFrame, 
                           cluster_ids: List[int]) -> pl.DataFrame:
        """Create time skeleton for all clusters and time intervals.
        
        Args:
            trips_df: DataFrame with trip data
            cluster_ids: List of cluster IDs
            
        Returns:
            DataFrame with time skeleton
        """
        self.logger.info("Creating time skeleton for cluster aggregation...")
        
        # get time range from data
        start_time = trips_df.select(pl.col("fecha_origen_recorrido").min()).item()
        end_time = trips_df.select(pl.col("fecha_origen_recorrido").max()).item()
        
        self.logger.info(f"  -> Time range: {start_time} to {end_time}")
        self.logger.info(f"  -> Time interval: {self.dt_minutes} minutes")
        self.logger.info(f"  -> Number of clusters: {len(cluster_ids)}")
        
        # create time series
        time_range = pl.datetime_range(
            start=start_time.replace(minute=0, second=0, microsecond=0),
            end=end_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            interval=f"{self.dt_minutes}m",
            eager=True
        )
        
        # create cartesian product of clusters and time intervals
        skeleton_data = []
        for cluster_id in cluster_ids:
            for timestamp in time_range:
                skeleton_data.append([cluster_id, timestamp])
                
        skeleton_df = pl.DataFrame(
            skeleton_data,
            schema=["cluster_id", "ts_start"]
        )
        
        self.logger.info(f"  -> Created skeleton with {skeleton_df.height:,} rows")
        self.logger.info(f"     ({len(cluster_ids)} clusters × {len(time_range)} time intervals)")
        
        return skeleton_df
        
    def aggregate_departures(self, trips_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate departure features by cluster and time.
        
        Args:
            trips_df: Classified trip data
            
        Returns:
            DataFrame with departure aggregations
        """
        self.logger.info("Aggregating departure features...")
        
        # round timestamp to time interval
        trips_with_time = trips_df.with_columns([
            pl.col("fecha_origen_recorrido").dt.truncate(f"{self.dt_minutes}m").alias("ts_start")
        ])
        
        # aggregate internal departures
        internal_deps = (
            trips_with_time
            .filter(pl.col("is_internal_trip"))
            .group_by(["origin_cluster_id", "ts_start"])
            .agg([
                pl.len().alias("dep_internal_count"),
                
                # gender counts for internal departures
                (pl.col("genero") == "MALE").sum().alias("dep_internal_male"),
                (pl.col("genero") == "FEMALE").sum().alias("dep_internal_female"),
                (pl.col("genero") == "OTHER").sum().alias("dep_internal_other"),
                
                # age bins for internal departures - NO STATISTICS, ONLY COUNTS
                pl.col("is_young").sum().alias("dep_internal_young"),
                pl.col("is_adult").sum().alias("dep_internal_adult"),
                pl.col("is_senior").sum().alias("dep_internal_senior"),
                pl.col("has_age_data").sum().alias("dep_internal_age_available"),
                
                # bike model diversity for internal departures
                pl.col("modelo_bicicleta").n_unique().alias("dep_internal_model_types"),
                
                # trip duration stats for internal departures - only basic stats
                pl.col("duracion_recorrido").mean().alias("dep_internal_duration_mean"),
                pl.col("duracion_recorrido").std().alias("dep_internal_duration_std"),
            ])
            .rename({"origin_cluster_id": "cluster_id"})
        )
        
        # aggregate external departures
        external_deps = (
            trips_with_time
            .filter(pl.col("is_external_trip"))
            .group_by(["origin_cluster_id", "ts_start"])
            .agg([
                pl.len().alias("dep_external_count"),
                
                # gender counts for external departures
                (pl.col("genero") == "MALE").sum().alias("dep_external_male"),
                (pl.col("genero") == "FEMALE").sum().alias("dep_external_female"),
                (pl.col("genero") == "OTHER").sum().alias("dep_external_other"),
                
                # age bins for external departures - NO STATISTICS, ONLY COUNTS
                pl.col("is_young").sum().alias("dep_external_young"),
                pl.col("is_adult").sum().alias("dep_external_adult"),
                pl.col("is_senior").sum().alias("dep_external_senior"),
                pl.col("has_age_data").sum().alias("dep_external_age_available"),
                
                # bike model diversity for external departures
                pl.col("modelo_bicicleta").n_unique().alias("dep_external_model_types"),
                
                # trip duration stats for external departures - only basic stats
                pl.col("duracion_recorrido").mean().alias("dep_external_duration_mean"),
                pl.col("duracion_recorrido").std().alias("dep_external_duration_std"),
            ])
            .rename({"origin_cluster_id": "cluster_id"})
        )
        
        # combine internal and external departures
        departures = internal_deps.join(
            external_deps, 
            on=["cluster_id", "ts_start"], 
            how="full"
        ).fill_null(0)
        
        # create has_departure flags
        departures = departures.with_columns([
            (pl.col("dep_internal_count") > 0).alias("has_dep_internal"),
            (pl.col("dep_external_count") > 0).alias("has_dep_external"),
        ])
        
        self.logger.info(f"  -> Departure aggregations shape: {departures.shape}")
        return departures
        
    def aggregate_arrivals(self, trips_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate arrival features by cluster and time.
        
        Args:
            trips_df: Classified trip data
            
        Returns:
            DataFrame with arrival aggregations
        """
        self.logger.info("Aggregating arrival features...")
        
        # use destination time for arrivals (fecha_destino_recorrido if available)
        time_col = "fecha_destino_recorrido" if "fecha_destino_recorrido" in trips_df.columns else "fecha_origen_recorrido"
        
        trips_with_time = trips_df.with_columns([
            pl.col(time_col).dt.truncate(f"{self.dt_minutes}m").alias("ts_start")
        ])
        
        # aggregate internal arrivals
        internal_arrs = (
            trips_with_time
            .filter(pl.col("is_internal_trip"))
            .group_by(["dest_cluster_id", "ts_start"])
            .agg([
                pl.len().alias("arr_internal_count"),
                
                # gender counts for internal arrivals
                (pl.col("genero") == "MALE").sum().alias("arr_internal_male"),
                (pl.col("genero") == "FEMALE").sum().alias("arr_internal_female"),
                (pl.col("genero") == "OTHER").sum().alias("arr_internal_other"),
                
                # age bins for internal arrivals - NO STATISTICS, ONLY COUNTS
                pl.col("is_young").sum().alias("arr_internal_young"),
                pl.col("is_adult").sum().alias("arr_internal_adult"),
                pl.col("is_senior").sum().alias("arr_internal_senior"),
                pl.col("has_age_data").sum().alias("arr_internal_age_available"),
                
                # bike model diversity for internal arrivals  
                pl.col("modelo_bicicleta").n_unique().alias("arr_internal_model_types"),
            ])
            .rename({"dest_cluster_id": "cluster_id"})
        )
        
        # aggregate external arrivals
        external_arrs = (
            trips_with_time
            .filter(pl.col("is_external_trip"))
            .group_by(["dest_cluster_id", "ts_start"])
            .agg([
                pl.len().alias("arr_external_count"),
                
                # gender counts for external arrivals
                (pl.col("genero") == "MALE").sum().alias("arr_external_male"),
                (pl.col("genero") == "FEMALE").sum().alias("arr_external_female"),
                (pl.col("genero") == "OTHER").sum().alias("arr_external_other"),
                
                # age bins for external arrivals - NO STATISTICS, ONLY COUNTS
                pl.col("is_young").sum().alias("arr_external_young"),
                pl.col("is_adult").sum().alias("arr_external_adult"),
                pl.col("is_senior").sum().alias("arr_external_senior"),
                pl.col("has_age_data").sum().alias("arr_external_age_available"),
                
                # bike model diversity for external arrivals
                pl.col("modelo_bicicleta").n_unique().alias("arr_external_model_types"),
            ])
            .rename({"dest_cluster_id": "cluster_id"})
        )
        
        # combine internal and external arrivals
        arrivals = internal_arrs.join(
            external_arrs,
            on=["cluster_id", "ts_start"],
            how="full"
        ).fill_null(0)
        
        # create has_arrival flags
        arrivals = arrivals.with_columns([
            (pl.col("arr_internal_count") > 0).alias("has_arr_internal"),
            (pl.col("arr_external_count") > 0).alias("has_arr_external"),
        ])
        
        self.logger.info(f"  -> Arrival aggregations shape: {arrivals.shape}")
        return arrivals
        
    def create_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal features to the aggregated data.
        
        Args:
            df: DataFrame with ts_start column
            
        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Adding temporal features...")
        
        df_with_temporal = df.with_columns([
            # basic time components
            pl.col("ts_start").dt.hour().alias("hour"),
            pl.col("ts_start").dt.weekday().alias("day_of_week"),
            pl.col("ts_start").dt.month().alias("month"),
            pl.col("ts_start").dt.year().alias("year"),
            
            # cyclical encodings
            (2 * np.pi * pl.col("ts_start").dt.hour() / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col("ts_start").dt.hour() / 24).cos().alias("hour_cos"),
            (2 * np.pi * pl.col("ts_start").dt.weekday() / 7).sin().alias("dow_sin"),
            (2 * np.pi * pl.col("ts_start").dt.weekday() / 7).cos().alias("dow_cos"),
            (2 * np.pi * pl.col("ts_start").dt.month() / 12).sin().alias("month_sin"),
            (2 * np.pi * pl.col("ts_start").dt.month() / 12).cos().alias("month_cos"),
            
            # business logic features
            pl.col("ts_start").dt.weekday().is_in([6, 7]).alias("is_weekend"),
            pl.col("ts_start").dt.hour().is_between(7, 9).alias("is_morning_rush"),
            pl.col("ts_start").dt.hour().is_between(17, 19).alias("is_evening_rush"),
            pl.col("ts_start").dt.hour().is_between(6, 22).alias("is_daytime"),
            
            # seasonal features
            pl.col("ts_start").dt.month().is_in([12, 1, 2]).alias("is_summer"),
            pl.col("ts_start").dt.month().is_in([6, 7, 8]).alias("is_winter"),
        ])
        
        self.logger.info(f"  -> Added temporal features")
        return df_with_temporal
        
    def add_weather_features(self, df: pl.DataFrame, 
                           weather_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Add weather features if available.
        
        Args:
            df: Base DataFrame with ts_start
            weather_df: Weather data (if separate) or None to use existing weather columns
            
        Returns:
            DataFrame with weather features
        """
        self.logger.info("Adding weather features...")
        
        if weather_df is not None:
            # join with separate weather data
            df_with_weather = df.join(
                weather_df, 
                on="ts_start", 
                how="left"
            )
        else:
            # weather features should already be in the original trips data
            # this is handled during the aggregation phase
            df_with_weather = df
            
        # check for weather columns and create weather availability flag
        weather_cols = [
            'temperature_2m', 'relative_humidity_2m', 'precipitation',
            'weather_code', 'wind_speed_10m', 'pressure_msl'
        ]
        
        has_weather_cols = [col for col in weather_cols if col in df_with_weather.columns]
        
        if has_weather_cols:
            self.logger.info(f"  -> Found {len(has_weather_cols)} weather columns")
            # add weather availability flag
            df_with_weather = df_with_weather.with_columns([
                pl.lit(True).alias("weather_data_available")
            ])
        else:
            self.logger.info("  -> No weather columns found, adding placeholder")
            df_with_weather = df_with_weather.with_columns([
                pl.lit(False).alias("weather_data_available")
            ])
            
        return df_with_weather
        
    def generate_cluster_features(self, trips_df: pl.DataFrame,
                                station_cluster_mapping: Dict[int, int],
                                cluster_centroids: Dict[int, Dict[str, float]]) -> pl.DataFrame:
        """Generate complete cluster feature dataset.
        
        Args:
            trips_df: Input trip data
            station_cluster_mapping: Station to cluster mapping
            cluster_centroids: Cluster centroid information
            
        Returns:
            Complete cluster feature dataset
        """
        self.logger.info("Generating complete cluster feature dataset...")
        
        # step 1: classify trips
        trips_classified = self.classify_trips(trips_df, station_cluster_mapping)
        
        # step 2: create time skeleton
        cluster_ids = list(cluster_centroids.keys())
        skeleton = self.create_time_skeleton(trips_classified, cluster_ids)
        
        # step 3: aggregate departures and arrivals
        departures = self.aggregate_departures(trips_classified)
        arrivals = self.aggregate_arrivals(trips_classified)
        
        # step 4: join skeleton with aggregations
        self.logger.info("Joining skeleton with aggregated features...")
        features = skeleton.join(
            departures, on=["cluster_id", "ts_start"], how="left"
        ).join(
            arrivals, on=["cluster_id", "ts_start"], how="left"
        ).fill_null(0)
        
        # step 5: add cluster metadata
        self.logger.info("Adding cluster metadata...")
        cluster_meta_data = []
        for cluster_id in cluster_ids:
            centroid = cluster_centroids[cluster_id]
            cluster_meta_data.append([
                cluster_id,
                centroid['lat'],
                centroid['lon'],
                centroid['station_count']
            ])
            
        cluster_meta_df = pl.DataFrame(
            cluster_meta_data,
            schema=['cluster_id', 'cluster_centroid_lat', 'cluster_centroid_lon', 'cluster_station_count']
        )
        
        features = features.join(cluster_meta_df, on="cluster_id", how="left")
        
        # step 6: add temporal features
        features = self.create_temporal_features(features)
        
        # step 7: add weather features (if available in original data)
        features = self.add_weather_features(features)
        
        # step 8: create summary statistics
        self.logger.info("Computing summary statistics...")
        total_rows = features.height
        non_zero_internal_deps = features.filter(pl.col("dep_internal_count") > 0).height
        non_zero_external_deps = features.filter(pl.col("dep_external_count") > 0).height
        non_zero_internal_arrs = features.filter(pl.col("arr_internal_count") > 0).height
        non_zero_external_arrs = features.filter(pl.col("arr_external_count") > 0).height
        
        self.logger.info(f"Cluster feature generation completed:")
        self.logger.info(f"  -> Total rows: {total_rows:,}")
        self.logger.info(f"  -> Rows with internal departures: {non_zero_internal_deps:,} ({non_zero_internal_deps/total_rows*100:.1f}%)")
        self.logger.info(f"  -> Rows with external departures: {non_zero_external_deps:,} ({non_zero_external_deps/total_rows*100:.1f}%)")
        self.logger.info(f"  -> Rows with internal arrivals: {non_zero_internal_arrs:,} ({non_zero_internal_arrs/total_rows*100:.1f}%)")
        self.logger.info(f"  -> Rows with external arrivals: {non_zero_external_arrs:,} ({non_zero_external_arrs/total_rows*100:.1f}%)")
        self.logger.info(f"  -> Final feature count: {features.width}")
        
        return features 