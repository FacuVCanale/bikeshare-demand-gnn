"""
Feature Engineering for Individual Station Bike Prediction

Converts trip-level data to 30-minute delta T format and creates features for:
- Climate features
- Station-wise features  
- Time-wise features
- User-wise features

Based on the approach document requirements.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for converting trip data to 30-minute delta T format
    and creating station-level features for bike prediction.
    """
    
    def __init__(self, delta_t_minutes=30, fill_nulls=True, fill_strategy='zero'):
        """
        Initialize the feature engineer.
        
        Args:
            delta_t_minutes (int): Time interval in minutes (default: 30)
            fill_nulls (bool): Whether to fill null values (default: True)
            fill_strategy (str): Strategy for filling nulls ('zero', 'mean', 'median', 'forward_fill')
        """
        self.delta_t_minutes = delta_t_minutes
        self.fill_nulls = fill_nulls
        self.fill_strategy = fill_strategy
        self.feature_columns = []
        
    def process_data(self, df, station_id=None):
        """
        Main processing function to convert trip data to delta T format with features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with trip and weather data
            station_id (int, optional): Specific station ID to filter for
            
        Returns:
            pd.DataFrame: Processed DataFrame with delta T format and features
        """
        print("Starting feature engineering...")
        print("Feature types to be created:")
        print("  ✓ Climate: temperature, humidity, apparent temperature + 1w lags")
        print("  ✓ Station: last delta T, 1w lag, weekly moving averages")  
        print("  ✓ Time: cyclical encoding for day_of_week/day_of_month/weekend")
        print("  ✗ User: excluded as requested")
        print(f"Null handling: {'Enabled' if self.fill_nulls else 'Disabled'} " +
              f"(strategy: {self.fill_strategy})" if self.fill_nulls else "")
        
        # Filter for specific station if provided
        if station_id is not None:
            # First check if station exists in the data
            origin_stations = set(df.select(pl.col('id_estacion_origen').unique()).to_series().to_list())
            dest_stations = set(df.select(pl.col('id_estacion_destino').unique()).to_series().to_list())
            all_stations = origin_stations | dest_stations
            
            if station_id not in all_stations:
                available_stations = sorted(list(all_stations))[:10]  # Show first 10
                raise ValueError(f"Station {station_id} not found in data. "
                               f"Available stations include: {available_stations}...")
            
            # Filter data for the specific station
            orig_len = len(df)
            df = df.filter(
                (pl.col('id_estacion_origen') == station_id) | 
                (pl.col('id_estacion_destino') == station_id)
            )
            print(f"Filtered data for station {station_id}: {len(df)} trips (from {orig_len} total trips)")
        
        # Convert to delta T format
        delta_df = self._convert_to_delta_t(df, station_id)
        print(f"Created delta T format: {len(delta_df)} time intervals")
        
        # Add all feature types (user features disabled as requested)
        delta_df = self._add_climate_features(delta_df)
        delta_df = self._add_station_features(delta_df)
        delta_df = self._add_time_features(delta_df)
        delta_df = self._add_user_features(delta_df)  # Currently disabled but included for completeness
        
        # Clean and finalize
        delta_df = self._clean_features(delta_df)
        
        self.feature_columns = [col for col in delta_df.columns if col not in 
                               ['datetime', 'station_id', 'arrivals', 'departures', 'lat', 'lon']]
        
        print(f"\n{'='*50}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*50}")
        print(f"✓ Created {len(self.feature_columns)} features")
        print(f"✓ Dataset shape: {delta_df.shape}")
        print(f"✓ Time range: {delta_df.select(pl.col('datetime').min()).item()} to {delta_df.select(pl.col('datetime').max()).item()}")
        print(f"✓ Stations: {delta_df.select(pl.col('station_id').n_unique()).item()}")
        print(f"✓ Total time intervals: {len(delta_df)}")
        print(f"✓ Features created:")
        
        # Group features by type for better readability
        climate_features = [f for f in self.feature_columns if 'weather_' in f]
        station_features = [f for f in self.feature_columns if any(x in f for x in ['arrivals_', 'departures_'])]
        time_features = [f for f in self.feature_columns if any(x in f for x in ['day_of_week', 'day_of_month', 'is_weekend'])]
        other_features = [f for f in self.feature_columns if f not in climate_features + station_features + time_features]
        
        if climate_features:
            print(f"   - Climate features: {len(climate_features)}")
        if station_features:
            print(f"   - Station features: {len(station_features)}")
        if time_features:
            print(f"   - Time features: {len(time_features)}")
        if other_features:
            print(f"   - Other features: {len(other_features)}")
        
        print("Feature engineering complete!")
        return delta_df
    
    def _convert_to_delta_t(self, df, station_id=None):
        """
        Convert trip data to 30-minute time intervals with arrival/departure counts.
        """
        # Extract station coordinates before any filtering
        # This ensures we have coordinate data for all possible stations
        print("  Extracting station coordinates...")
        # Assuming coordinate column names, will raise error if not found
        try:
            coords_origen = df.select([
                pl.col('id_estacion_origen').alias('station_id'),
                pl.col('lat_estacion_origen').alias('lat'),
                pl.col('long_estacion_origen').alias('lon')
            ]).drop_nulls()
            
            coords_destino = df.select([
                pl.col('id_estacion_destino').alias('station_id'),
                pl.col('lat_estacion_destino').alias('lat'),
                pl.col('long_estacion_destino').alias('lon')
            ]).drop_nulls()

            station_coords = (
                pl.concat([coords_origen, coords_destino])
                .unique(subset='station_id', keep='first')
                .with_columns(pl.col('station_id').cast(pl.Int64))
            )
            print(f"  Found coordinates for {len(station_coords)} unique stations")
        except Exception as e:
            print(f"  ERROR: Could not find coordinate columns (e.g., 'lat_estacion_origen').")
            print(f"  Please ensure coordinate columns are in the input data. Details: {e}")
            # Create an empty dataframe so the script can continue without coords
            station_coords = pl.DataFrame({
                'station_id': pl.Series([], dtype=pl.Int64),
                'lat': pl.Series([], dtype=pl.Float64),
                'lon': pl.Series([], dtype=pl.Float64)
            })
            
        # Ensure datetime columns and station_id consistency
        df = df.with_columns([
            pl.col('id_estacion_origen').cast(pl.Int64),
            pl.col('id_estacion_destino').cast(pl.Int64)
        ])
        
        # Create time range
        start_time = df.select(pl.col('fecha_origen_recorrido').min()).item()
        end_time = df.select(pl.col('fecha_destino_recorrido').max()).item()
        
        # Round to nearest delta_t interval using proper datetime arithmetic
        from datetime import datetime, timedelta
        
        # Convert to total minutes since epoch for easier calculation
        epoch = datetime(1970, 1, 1)
        start_minutes = int((start_time - epoch).total_seconds() / 60)
        end_minutes = int((end_time - epoch).total_seconds() / 60)
        
        # Floor start_time to nearest delta_t interval
        start_minutes_floored = (start_minutes // self.delta_t_minutes) * self.delta_t_minutes
        start_time = epoch + timedelta(minutes=start_minutes_floored)
        
        # Ceiling end_time to nearest delta_t interval  
        end_minutes_ceiled = ((end_minutes + self.delta_t_minutes - 1) // self.delta_t_minutes) * self.delta_t_minutes
        end_time = epoch + timedelta(minutes=end_minutes_ceiled)
        
        # Create time index as polars DataFrame
        time_index = pl.datetime_range(start_time, end_time, f'{self.delta_t_minutes}m', eager=True)
        
        if station_id is not None:
            # Single station case
            print(f"  Processing single station: {station_id}")
            delta_df = pl.DataFrame({
                'datetime': time_index,
                'station_id': [int(station_id)] * len(time_index)
            }).with_columns(pl.col('station_id').cast(pl.Int64))
            print(f"  Created base time grid: {len(delta_df)} intervals")
            
            # Count departures (trips starting from this station)
            departures_df = df.filter(pl.col('id_estacion_origen') == station_id)
            print(f"  Found {len(departures_df)} departure trips for station {station_id}")
            departures_count = (departures_df
                              .with_columns(
                                  pl.col('fecha_origen_recorrido')
                                  .dt.truncate(f'{self.delta_t_minutes}m')
                                  .alias('datetime')
                              )
                              .group_by('datetime')
                              .agg(pl.len().alias('departures'))
                              .with_columns(pl.lit(int(station_id)).cast(pl.Int64).alias('station_id')))
            print(f"  Aggregated departures into {len(departures_count)} time intervals")
            
            # Count arrivals (trips ending at this station)
            arrivals_df = df.filter(pl.col('id_estacion_destino') == station_id)
            print(f"  Found {len(arrivals_df)} arrival trips for station {station_id}")
            arrivals_count = (arrivals_df
                            .with_columns(
                                pl.col('fecha_destino_recorrido')
                                .dt.truncate(f'{self.delta_t_minutes}m')
                                .alias('datetime')
                            )
                            .group_by('datetime')
                            .agg(pl.len().alias('arrivals'))
                            .with_columns(pl.lit(int(station_id)).cast(pl.Int64).alias('station_id')))
            print(f"  Aggregated arrivals into {len(arrivals_count)} time intervals")
            
        else:
            # Multi-station case - get all unique stations
            origin_stations = df.select(pl.col('id_estacion_origen').unique()).to_series().to_list()
            dest_stations = df.select(pl.col('id_estacion_destino').unique()).to_series().to_list()
            stations = list(set(origin_stations + dest_stations))
            
            # Create base DataFrame using cross join with integer station_id
            time_df = pl.DataFrame({'datetime': time_index})
            stations_df = pl.DataFrame({'station_id': stations}).with_columns(
                pl.col('station_id').cast(pl.Int64)
            )
            delta_df = time_df.join(stations_df, how='cross')
            
            # Count departures for all stations
            departures_count = (df
                              .with_columns(
                                  pl.col('fecha_origen_recorrido')
                                  .dt.truncate(f'{self.delta_t_minutes}m')
                                  .alias('datetime')
                              )
                              .group_by(['datetime', 'id_estacion_origen'])
                              .agg(pl.len().alias('departures'))
                              .rename({'id_estacion_origen': 'station_id'})
                              .with_columns(pl.col('station_id').cast(pl.Int64)))
            
            # Count arrivals for all stations
            arrivals_count = (df
                            .with_columns(
                                pl.col('fecha_destino_recorrido')
                                .dt.truncate(f'{self.delta_t_minutes}m')
                                .alias('datetime')
                            )
                            .group_by(['datetime', 'id_estacion_destino'])
                            .agg(pl.len().alias('arrivals'))
                            .rename({'id_estacion_destino': 'station_id'})
                            .with_columns(pl.col('station_id').cast(pl.Int64)))
        
        # Ensure all datetime columns are the same dtype (ns)
        delta_df = delta_df.with_columns(pl.col('datetime').cast(pl.Datetime('ns')))
        departures_count = departures_count.with_columns(pl.col('datetime').cast(pl.Datetime('ns')))
        arrivals_count = arrivals_count.with_columns(pl.col('datetime').cast(pl.Datetime('ns')))
        
        # Merge counts (now both single and multi-station cases have station_id column)
        print(f"  Merging departure counts...")
        delta_df = delta_df.join(departures_count, on=['datetime', 'station_id'], how='left')
        print(f"  Merging arrival counts...")
        delta_df = delta_df.join(arrivals_count, on=['datetime', 'station_id'], how='left')
        
        # Merge coordinates
        if not station_coords.is_empty():
            print(f"  Merging coordinates...")
            delta_df = delta_df.join(station_coords, on='station_id', how='left')
            # Check for nulls after coordinate merge
            lat_null_count = delta_df.select(pl.col('lat').is_null().sum()).item()
            if lat_null_count > 0:
                print(f"    WARNING: {lat_null_count} rows are missing coordinate data after merge.")
        
        print(f"  After merging: {len(delta_df)} rows, columns: {delta_df.columns}")
        
        # Fill missing values
        departures_nulls = delta_df.select(pl.col('departures').is_null().sum()).item()
        arrivals_nulls = delta_df.select(pl.col('arrivals').is_null().sum()).item()
        print(f"  Nulls before filling: departures={departures_nulls}, arrivals={arrivals_nulls}")
        
        delta_df = delta_df.with_columns([
            pl.col('departures').fill_null(0),
            pl.col('arrivals').fill_null(0)
        ])
        
        print(f"  Final delta_df shape: {delta_df.shape}")
        
        # Add only the three key weather features mentioned in approach document
        key_weather_features = [
            'weather_temperature_2m',
            'weather_relative_humidity_2m', 
            'weather_apparent_temperature'
        ]
        
        # Check which of the key weather features are available
        available_weather_cols = [col for col in key_weather_features if col in df.columns]
        
        if available_weather_cols:
            print(f"  Found {len(available_weather_cols)} key weather columns: {available_weather_cols}")
            # Get weather data by taking the mean for each time interval
            weather_agg = (df
                          .select(['fecha_origen_recorrido'] + available_weather_cols)
                          .with_columns(
                              pl.col('fecha_origen_recorrido')
                              .dt.truncate(f'{self.delta_t_minutes}m')
                              .alias('datetime')
                          )
                          .group_by('datetime')
                          .agg([pl.col(col).mean() for col in available_weather_cols]))
            
            delta_df = delta_df.join(weather_agg, on='datetime', how='left')
            
            # Check for nulls after weather merge
            for col in available_weather_cols:
                null_count = delta_df.select(pl.col(col).is_null().sum()).item()
                if null_count > 0:
                    print(f"    {col}: {null_count} nulls after merge ({null_count/len(delta_df)*100:.1f}%)")
            
            # Forward fill weather data
            print("  Forward filling weather data...")
            delta_df = delta_df.with_columns([
                pl.col(col).fill_null(strategy='forward') for col in available_weather_cols
            ])
            
            # Check remaining nulls after forward fill
            remaining_nulls = sum([
                delta_df.select(pl.col(col).is_null().sum()).item() 
                for col in available_weather_cols
            ])
            if remaining_nulls > 0:
                print(f"    {remaining_nulls} weather nulls remain after forward fill")
            else:
                print("    All weather nulls filled by forward fill")
        else:
            print(f"  None of the key weather features found in data: {key_weather_features}")
            weather_cols = [col for col in df.columns if 'weather' in col.lower()]
            print("  Available columns:", weather_cols)
        
        return delta_df
    
    def _add_climate_features(self, df):
        """
        Add climate-related features.
        Focus on only the three key weather features as specified in approach.
        """
        print("Adding climate features...")
        
        # Use only the three key weather features as specified in the approach
        key_weather_features = [
            'weather_temperature_2m',
            'weather_relative_humidity_2m', 
            'weather_apparent_temperature'
        ]
        
        # Check which features are actually available
        available_features = [f for f in key_weather_features if f in df.columns]
        missing_features = [f for f in key_weather_features if f not in df.columns]
        
        if missing_features:
            print(f"  Warning: Missing weather features: {missing_features}")
        
        if available_features:
            print(f"  Using available weather features: {available_features}")
            
            # Add lagged weather features (1 week lag)
            lag_periods = 7 * 24 * (60 // self.delta_t_minutes)  # 1 week in delta_t periods
            lag_expressions = []
            
            for feature in available_features:
                lag_feature_name = f'{feature}_1w_lag'
                lag_expressions.append(
                    pl.col(feature).shift(lag_periods).over('station_id').alias(lag_feature_name)
                )
            
            df = df.with_columns(lag_expressions)
            
            # Print null counts for lagged weather features
            for feature in available_features:
                lag_feature_name = f'{feature}_1w_lag'
                null_count = df.select(pl.col(lag_feature_name).is_null().sum()).item()
                print(f"    {lag_feature_name}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
        else:
            print("  No key weather features available - skipping climate features")
        
        return df
    
    def _add_station_features(self, df):
        """
        Add station-wise features focusing on past vs future state.
        """
        print("Adding station features...")
        
        # Last delta T features and 1 week lag features
        lag_periods = 7 * 24 * (60 // self.delta_t_minutes)  # 1 week in delta_t periods
        
        df = df.with_columns([
            pl.col('arrivals').shift(1).over('station_id').alias('arrivals_last_dt'),
            pl.col('departures').shift(1).over('station_id').alias('departures_last_dt'),
            pl.col('arrivals').shift(lag_periods).over('station_id').alias('arrivals_1w_lag'),
            pl.col('departures').shift(lag_periods).over('station_id').alias('departures_1w_lag')
        ])
        
        # Print null counts for last delta T features
        arrivals_last_nulls = df.select(pl.col('arrivals_last_dt').is_null().sum()).item()
        departures_last_nulls = df.select(pl.col('departures_last_dt').is_null().sum()).item()
        print(f"  arrivals_last_dt: {arrivals_last_nulls} nulls ({arrivals_last_nulls/len(df)*100:.1f}%)")
        print(f"  departures_last_dt: {departures_last_nulls} nulls ({departures_last_nulls/len(df)*100:.1f}%)")
        
        # Print null counts for 1 week lag features
        arrivals_1w_nulls = df.select(pl.col('arrivals_1w_lag').is_null().sum()).item()
        departures_1w_nulls = df.select(pl.col('departures_1w_lag').is_null().sum()).item()
        print(f"  arrivals_1w_lag: {arrivals_1w_nulls} nulls ({arrivals_1w_nulls/len(df)*100:.1f}%)")
        print(f"  departures_1w_lag: {departures_1w_nulls} nulls ({departures_1w_nulls/len(df)*100:.1f}%)")
        
        # Create time slot identifier for same-time-across-weeks moving averages
        # Combine hour and day_of_week to create unique time slots
        dow_exists = 'day_of_week' in df.columns
        
        time_exprs = []
        # Extract hour temporarily for time_slot calculation
        time_exprs.append(pl.col('datetime').dt.hour().alias('hour_temp'))
        if not dow_exists:
            time_exprs.append(pl.col('datetime').dt.weekday().alias('day_of_week'))
        
        if time_exprs:
            df = df.with_columns(time_exprs)
        
        df = df.with_columns(
            (pl.col('hour_temp') * 7 + pl.col('day_of_week')).alias('time_slot')
        )
        
        print("  Computing moving averages for same time slots across weeks...")
        
        # Sort by station_id, time_slot, and datetime for proper window operations
        df = df.sort(['station_id', 'time_slot', 'datetime'])
        
        # 4-week and 12-week moving averages for same time slots
        print("    Computing 4-week and 12-week moving averages (excluding current interval)...")
        df = df.with_columns([
            # Shift 1 interval back to avoid using value at t (prevents leakage)
            pl.col('arrivals').shift(1).rolling_mean(window_size=4, min_periods=1)
              .over(['station_id', 'time_slot']).alias('arrivals_ma4_weekly'),
            pl.col('departures').shift(1).rolling_mean(window_size=4, min_periods=1)
              .over(['station_id', 'time_slot']).alias('departures_ma4_weekly'),
            pl.col('arrivals').shift(1).rolling_mean(window_size=12, min_periods=1)
              .over(['station_id', 'time_slot']).alias('arrivals_ma12_weekly'),
            pl.col('departures').shift(1).rolling_mean(window_size=12, min_periods=1)
              .over(['station_id', 'time_slot']).alias('departures_ma12_weekly')
        ])
        
        # Print null counts for moving averages
        ma_cols = ['arrivals_ma4_weekly', 'departures_ma4_weekly', 'arrivals_ma12_weekly', 'departures_ma12_weekly']
        for col in ma_cols:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            print(f"  {col}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
        
        # Conditionally fill missing values in station features
        station_features = [
            'arrivals_last_dt', 'departures_last_dt',
            'arrivals_1w_lag', 'departures_1w_lag',
            'arrivals_ma4_weekly', 'departures_ma4_weekly',
            'arrivals_ma12_weekly', 'departures_ma12_weekly'
        ]
        
        if self.fill_nulls:
            print(f"  Filling station feature nulls with strategy: {self.fill_strategy}...")
            fill_exprs = []
            
            for feature in station_features:
                null_count_before = df.select(pl.col(feature).is_null().sum()).item()
                if null_count_before > 0:
                    if self.fill_strategy == 'zero':
                        fill_exprs.append(pl.col(feature).fill_null(0))
                    elif self.fill_strategy == 'mean':
                        mean_val = df.select(pl.col(feature).mean()).item()
                        fill_exprs.append(pl.col(feature).fill_null(mean_val))
                    elif self.fill_strategy == 'median':
                        median_val = df.select(pl.col(feature).median()).item()
                        fill_exprs.append(pl.col(feature).fill_null(median_val))
                    elif self.fill_strategy == 'forward_fill':
                        fill_exprs.append(
                            pl.col(feature).fill_null(strategy='forward').over('station_id').fill_null(0)
                        )
                    else:
                        # Keep as is if no matching strategy
                        fill_exprs.append(pl.col(feature))
                    
                    print(f"    {feature}: filled {null_count_before} nulls")
                else:
                    fill_exprs.append(pl.col(feature))
            
            if fill_exprs:
                df = df.with_columns(fill_exprs)
        else:
            print("  Keeping station feature nulls as requested...")
            for feature in station_features:
                null_count = df.select(pl.col(feature).is_null().sum()).item()
                if null_count > 0:
                    print(f"    {feature}: {null_count} nulls preserved")
        
        # Drop temporary columns
        df = df.drop(['time_slot', 'hour_temp'])
        
        return df
    
    def _add_time_features(self, df):
        """
        Add time-wise features using cyclical encoding.
        """
        print("Adding time features...")
        
        # Extract basic time components (check if already exist to avoid duplication)
        time_exprs = []
        
        if 'day_of_week' not in df.columns:
            time_exprs.append(pl.col('datetime').dt.weekday().alias('day_of_week'))
        
        time_exprs.extend([
            pl.col('datetime').dt.day().alias('day_of_month'),
        ])
        
        if time_exprs:
            df = df.with_columns(time_exprs)
        
        # Boolean and cyclical features
        df = df.with_columns([
            # Boolean features
            (pl.col('day_of_week') >= 5).cast(pl.Int32).alias('is_weekend'),
            
            # Cyclical encoding for periodic features
            # Day of week (7-day cycle)
            (2 * np.pi * pl.col('day_of_week') / 7).sin().alias('dow_sin'),
            (2 * np.pi * pl.col('day_of_week') / 7).cos().alias('dow_cos'),
            
            # Day of month (30-day cycle approximation)
            (2 * np.pi * pl.col('day_of_month') / 30).sin().alias('dom_sin'),
            (2 * np.pi * pl.col('day_of_month') / 30).cos().alias('dom_cos'),
        ])

        return df
    
    def _add_user_features(self, df):
        """
        Add user-wise features (simplified version for now).
        Note: This method is currently disabled per user request.
        """
        print("User features disabled - skipping...")
        
        # DISABLED: User features removed per user request
        # Keeping method for potential future use
        
        # # Station popularity (total trips per station)  
        # station_popularity = df.groupby('station_id')[['arrivals', 'departures']].sum()
        # station_popularity['total_trips'] = station_popularity['arrivals'] + station_popularity['departures']
        # station_popularity['popularity_score'] = (station_popularity['total_trips'] / 
        #                                          station_popularity['total_trips'].max())
        # 
        # # Merge back to main dataframe
        # df = df.merge(station_popularity[['popularity_score']].reset_index(), 
        #              on='station_id', how='left')
        
        return df
    
    def _clean_features(self, df):
        """
        Clean and finalize features.
        """
        print("Cleaning features...")
        
        # Check for infinite values
        numeric_columns = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        inf_counts = {}
        
        for col in numeric_columns:
            inf_count = df.select(pl.col(col).is_infinite().sum()).item()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"  Found infinite values in columns: {inf_counts}")
            print("  Replacing infinite values with NaN...")
            # Replace infinite values with null
            for col in inf_counts.keys():
                df = df.with_columns(
                    pl.when(pl.col(col).is_infinite())
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        else:
            print("  No infinite values found")
        
        # Check remaining NaN values before final fill
        print("  Checking remaining null values before final cleanup...")
        total_nulls = 0
        null_cols = []
        
        for col in numeric_columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            if null_count > 0:
                print(f"    {col}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
                total_nulls += null_count
                null_cols.append(col)
        
        if total_nulls > 0:
            print(f"  Total nulls found: {total_nulls}")
            
            if self.fill_nulls:
                print(f"  Filling remaining NaN values using strategy: {self.fill_strategy}")
                
                if self.fill_strategy == 'zero':
                    fill_exprs = [pl.col(col).fill_null(0) for col in null_cols]
                    df = df.with_columns(fill_exprs)
                    
                elif self.fill_strategy == 'mean':
                    for col in null_cols:
                        mean_val = df.select(pl.col(col).mean()).item()
                        df = df.with_columns(pl.col(col).fill_null(mean_val))
                        print(f"    {col}: filled with mean={mean_val:.3f}")
                        
                elif self.fill_strategy == 'median':
                    for col in null_cols:
                        median_val = df.select(pl.col(col).median()).item()
                        df = df.with_columns(pl.col(col).fill_null(median_val))
                        print(f"    {col}: filled with median={median_val:.3f}")
                        
                elif self.fill_strategy == 'forward_fill':
                    # Sort by station_id and datetime first for proper forward fill
                    df = df.sort(['station_id', 'datetime'])
                    
                    fill_exprs = []
                    for col in null_cols:
                        # Forward fill within each station, then fill remaining with 0
                        fill_exprs.append(
                            pl.col(col).fill_null(strategy='forward').over('station_id').fill_null(0).alias(col)
                        )
                    
                    df = df.with_columns(fill_exprs)
                    
                    for col in null_cols:
                        print(f"    {col}: forward filled successfully")
                
                print("  All nulls filled")
            else:
                print("  Null filling disabled - keeping null values for model to handle")
                print("  WARNING: Some models may not handle null values well!")
        else:
            print("  No remaining null values found")
        
        # Sort by datetime and station
        print("  Sorting by station_id and datetime...")
        df = df.sort(['station_id', 'datetime'])
        
        # Final validation - check for any remaining nulls
        null_counts = df.null_count()
        final_nulls = null_counts.sum_horizontal().item()
        if final_nulls > 0:
            if self.fill_nulls:
                print(f"  WARNING: {final_nulls} null values remain after cleaning!")
            else:
                print(f"  ✓ {final_nulls} null values preserved as requested")
        else:
            print("  ✓ No null values remain - dataset clean")
        
        return df
    
    def get_feature_names(self):
        """
        Get list of feature column names.
        """
        return self.feature_columns
    
    def prepare_for_training(self, df, target_cols=['arrivals', 'departures']):
        """
        Prepare data for training by separating features and targets.
        
        Args:
            df (pl.DataFrame): Processed DataFrame
            target_cols (list): Target column names
            
        Returns:
            tuple: (X, y, metadata) where X=features, y=targets, metadata=datetime/station info
        """
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in 
                       ['datetime', 'station_id', 'lat', 'lon'] + target_cols]
        
        X = df.select(feature_cols)
        y = df.select(target_cols)
        metadata = df.select(['datetime', 'station_id'])
        
        print(f"Prepared for training: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, metadata


def load_and_process_data(data_path, station_id=None, delta_t_minutes=30, 
                         fill_nulls=True, fill_strategy='zero', create_splits=False,
                         train_split=0.7, val_split=0.15, output_dir=None):
    """
    Convenience function to load and process data.
    
    Args:
        data_path (str): Path to the input data file (CSV or Parquet)
        station_id (int, optional): Specific station ID to process
        delta_t_minutes (int): Time interval in minutes
        fill_nulls (bool): Whether to fill null values (default: True)
        fill_strategy (str): Strategy for filling nulls ('zero', 'mean', 'median', 'forward_fill')
        create_splits (bool): Whether to create train/val/test splits (default: False)
        train_split (float): Fraction of data for training (default: 0.7)
        val_split (float): Fraction of data for validation (default: 0.15)
        output_dir (str, optional): Directory to save split files. If None, returns splits in memory
        
    Returns:
        tuple: If create_splits=False: (processed_df, feature_engineer)
               If create_splits=True: (splits_dict, feature_engineer) where splits_dict contains train/val/test
    """
    print(f"Loading data from {data_path}...")
    
    # Determine file format and load accordingly
    if data_path.lower().endswith('.parquet'):
        print("  Detected parquet format")
        df = pl.read_parquet(data_path)
    elif data_path.lower().endswith('.csv'):
        print("  Detected CSV format")
        df = pl.read_csv(data_path, try_parse_dates=True)
    else:
        # Try parquet first, then CSV as fallback
        try:
            print("  File extension not recognized, trying parquet format...")
            df = pl.read_parquet(data_path)
            print("  Successfully loaded as parquet")
        except:
            try:
                print("  Parquet failed, trying CSV format...")
                df = pl.read_csv(data_path, try_parse_dates=True)
                print("  Successfully loaded as CSV")
            except Exception as e:
                raise ValueError(f"Could not load file {data_path}. Tried both parquet and CSV formats. Error: {str(e)}")
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns}")
    
    # Initialize feature engineer
    fe = FeatureEngineer(delta_t_minutes=delta_t_minutes, 
                        fill_nulls=fill_nulls, 
                        fill_strategy=fill_strategy)
    
    # Process data
    processed_df = fe.process_data(df, station_id=station_id)
    
    if not create_splits:
        return processed_df, fe
    
    # Create train/val/test splits
    print(f"\n{'='*50}")
    print("CREATING TEMPORAL SPLITS")
    print(f"{'='*50}")
    
    splits = create_temporal_splits(
        processed_df, 
        train_split=train_split, 
        val_split=val_split
    )
    
    # Save splits if output directory is provided
    if output_dir is not None:
        save_splits_to_files(splits, output_dir, fe)
    
    return splits, fe


def create_temporal_splits(df, train_split=0.7, val_split=0.15):
    """
    Create temporal train/val/test splits from processed data.
    
    Args:
        df (pl.DataFrame): Processed DataFrame with datetime column
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' splits
    """
    print("Creating temporal data splits...")
    
    # Sort by datetime for temporal consistency
    df_sorted = df.sort('datetime')
    n_total = len(df_sorted)
    
    # Calculate split points
    train_end = int(n_total * train_split)
    val_end = int(n_total * (train_split + val_split))
    test_split = 1.0 - train_split - val_split
    
    # Create splits
    splits = {
        'train': df_sorted[:train_end],
        'val': df_sorted[train_end:val_end],
        'test': df_sorted[val_end:]
    }
    
    # Print split information
    print(f"  Total samples: {n_total}")
    print(f"  Split ratios: train={train_split:.1%}, val={val_split:.1%}, test={test_split:.1%}")
    print(f"  Split sizes:")
    
    for split_name, split_df in splits.items():
        start_date = split_df.select(pl.col('datetime').min()).item()
        end_date = split_df.select(pl.col('datetime').max()).item()
        print(f"    {split_name}: {len(split_df)} samples ({start_date} to {end_date})")
    
    return splits


def save_splits_to_files(splits, output_dir, feature_engineer):
    """
    Save train/val/test splits to separate files.
    
    Args:
        splits (dict): Dictionary with train/val/test DataFrames
        output_dir (str): Output directory path
        feature_engineer (FeatureEngineer): Feature engineer instance for metadata
    """
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}...")
    
    # Save each split
    for split_name, split_df in splits.items():
        # Save data
        file_path = output_path / f'{split_name}_features.csv'
        split_df.write_csv(file_path)
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {split_name}: {file_path} ({file_size_mb:.1f} MB)")
    
    # Save metadata
    feature_names = feature_engineer.get_feature_names()
    target_cols = ['arrivals', 'departures']
    
    # Get unique stations
    unique_stations = splits['train'].select(pl.col('station_id').unique()).to_series().to_list()
    
    # Create metadata
    metadata = {
        'feature_names': feature_names,
        'target_columns': target_cols,
        'n_features': len(feature_names),
        'n_targets': len(target_cols),
        'has_coordinates': 'lat' in splits['train'].columns,
        'unique_stations': sorted(unique_stations),
        'n_stations': len(unique_stations),
        'delta_t_minutes': feature_engineer.delta_t_minutes,
        'splits': {
            'train': {
                'n_samples': len(splits['train']),
                'date_range': [
                    str(splits['train'].select(pl.col('datetime').min()).item()),
                    str(splits['train'].select(pl.col('datetime').max()).item())
                ]
            },
            'val': {
                'n_samples': len(splits['val']),
                'date_range': [
                    str(splits['val'].select(pl.col('datetime').min()).item()),
                    str(splits['val'].select(pl.col('datetime').max()).item())
                ]
            },
            'test': {
                'n_samples': len(splits['test']),
                'date_range': [
                    str(splits['test'].select(pl.col('datetime').min()).item()),
                    str(splits['test'].select(pl.col('datetime').max()).item())
                ]
            }
        },
        'processing_config': {
            'fill_nulls': feature_engineer.fill_nulls,
            'fill_strategy': feature_engineer.fill_strategy,
            'created_features': {
                'climate': bool([f for f in feature_names if 'weather_' in f]),
                'station': bool([f for f in feature_names if any(x in f for x in ['arrivals_', 'departures_'])]),
                'time': bool([f for f in feature_names if any(x in f for x in ['day_of_week', 'is_weekend'])]),
            }
        }
    }
    
    # Save metadata
    metadata_path = output_path / 'feature_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata: {metadata_path}")
    
    # Save summary
    print(f"\n{'='*50}")
    print("FEATURE ENGINEERING WITH SPLITS COMPLETE!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    
    total_size_mb = 0
    for file in sorted(output_path.glob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  {file.name} ({size_mb:.1f} MB)")
    
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"\nReady for GNN processing:")
    print(f"  python scripts/prepare_gnn_dataset.py --input_file {output_dir}/train_features.csv")


def main():
    """Main function for command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Engineering for EcoBici bike prediction')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input data file (CSV or Parquet)')
    parser.add_argument('--output_dir', type=str, default='data/processed_features',
                        help='Output directory for processed features and splits')
    parser.add_argument('--station_id', type=int, default=None,
                        help='Specific station ID to process (optional)')
    parser.add_argument('--delta_t_minutes', type=int, default=30,
                        help='Time interval in minutes (default: 30)')
    parser.add_argument('--fill_nulls', action='store_true', default=True,
                        help='Whether to fill null values (default: True)')
    parser.add_argument('--no_fill_nulls', action='store_false', dest='fill_nulls',
                        help='Disable null filling')
    parser.add_argument('--fill_strategy', type=str, default='zero',
                        choices=['zero', 'mean', 'median', 'forward_fill'],
                        help='Strategy for filling nulls (default: zero)')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction of data for training (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15)')
    parser.add_argument('--no_splits', action='store_true',
                        help='Do not create train/val/test splits, just process features')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if not args.no_splits:
        test_split = 1.0 - args.train_split - args.val_split
        if test_split <= 0:
            raise ValueError(f"Invalid split ratios: train={args.train_split}, val={args.val_split}. "
                           f"Sum must be < 1.0 to leave room for test split.")
        print(f"Split ratios: train={args.train_split:.1%}, val={args.val_split:.1%}, test={test_split:.1%}")
    
    # Process data
    print(f"{'='*80}")
    print("ECOBICI FEATURE ENGINEERING")
    print(f"{'='*80}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Station filter: {args.station_id if args.station_id else 'All stations'}")
    print(f"Time interval: {args.delta_t_minutes} minutes")
    print(f"Null handling: {'Enabled' if args.fill_nulls else 'Disabled'}" + 
          (f" (strategy: {args.fill_strategy})" if args.fill_nulls else ""))
    print(f"Create splits: {'No' if args.no_splits else 'Yes'}")
    
    if args.no_splits:
        # Just process features without splits
        processed_df, fe = load_and_process_data(
            data_path=args.input_file,
            station_id=args.station_id,
            delta_t_minutes=args.delta_t_minutes,
            fill_nulls=args.fill_nulls,
            fill_strategy=args.fill_strategy,
            create_splits=False
        )
        
        # Save single processed file
        from pathlib import Path
        import json
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        output_file = output_path / 'processed_features.csv'
        processed_df.write_csv(output_file)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        # Save metadata
        feature_names = fe.get_feature_names()
        unique_stations = processed_df.select(pl.col('station_id').unique()).to_series().to_list()
        
        metadata = {
            'feature_names': feature_names,
            'target_columns': ['arrivals', 'departures'],
            'n_features': len(feature_names),
            'n_targets': 2,
            'has_coordinates': 'lat' in processed_df.columns,
            'unique_stations': sorted(unique_stations),
            'n_stations': len(unique_stations),
            'delta_t_minutes': fe.delta_t_minutes,
            'n_samples': len(processed_df),
            'date_range': [
                str(processed_df.select(pl.col('datetime').min()).item()),
                str(processed_df.select(pl.col('datetime').max()).item())
            ],
            'processing_config': {
                'fill_nulls': fe.fill_nulls,
                'fill_strategy': fe.fill_strategy,
                'station_filter': args.station_id
            }
        }
        
        metadata_path = output_path / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Output file: {output_file} ({file_size_mb:.1f} MB)")
        print(f"Metadata: {metadata_path}")
        print(f"Features created: {len(feature_names)}")
        print(f"Samples: {len(processed_df)}")
        
    else:
        # Process with splits
        splits, fe = load_and_process_data(
            data_path=args.input_file,
            station_id=args.station_id,
            delta_t_minutes=args.delta_t_minutes,
            fill_nulls=args.fill_nulls,
            fill_strategy=args.fill_strategy,
            create_splits=True,
            train_split=args.train_split,
            val_split=args.val_split,
            output_dir=args.output_dir
        )
        
        print(f"\nReady for next steps:")
        print(f"  GNN training: python scripts/prepare_gnn_dataset.py --input_dir {args.output_dir}")
        print(f"  Traditional ML: Use files in {args.output_dir}")


if __name__ == "__main__":
    main() 