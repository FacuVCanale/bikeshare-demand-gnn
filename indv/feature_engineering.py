"""
Feature Engineering for Individual Station Bike Prediction

Converts trip-level data to 30-minute delta T format and creates features for:
- Climate features
- Station-wise features  
- Time-wise features
- User-wise features

Based on the approach document requirements.
"""

import pandas as pd
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
        print("  ✓ Time: cyclical encoding for hour/day/month/year")
        print("  ✗ User: excluded as requested")
        print(f"Null handling: {'Enabled' if self.fill_nulls else 'Disabled'} " +
              f"(strategy: {self.fill_strategy})" if self.fill_nulls else "")
        
        # Filter for specific station if provided
        if station_id is not None:
            # First check if station exists in the data
            origin_stations = set(df['id_estacion_origen'].unique())
            dest_stations = set(df['id_estacion_destino'].unique())
            all_stations = origin_stations | dest_stations
            
            if station_id not in all_stations:
                available_stations = sorted(list(all_stations))[:10]  # Show first 10
                raise ValueError(f"Station {station_id} not found in data. "
                               f"Available stations include: {available_stations}...")
            
            # Filter data for the specific station
            orig_len = len(df)
            df = df[
                (df['id_estacion_origen'] == station_id) | 
                (df['id_estacion_destino'] == station_id)
            ].copy()
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
                               ['datetime', 'station_id', 'arrivals', 'departures']]
        
        print(f"\n{'='*50}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*50}")
        print(f"✓ Created {len(self.feature_columns)} features")
        print(f"✓ Dataset shape: {delta_df.shape}")
        print(f"✓ Time range: {delta_df['datetime'].min()} to {delta_df['datetime'].max()}")
        print(f"✓ Stations: {delta_df['station_id'].nunique()}")
        print(f"✓ Total time intervals: {len(delta_df)}")
        print(f"✓ Features created:")
        
        # Group features by type for better readability
        climate_features = [f for f in self.feature_columns if 'weather_' in f]
        station_features = [f for f in self.feature_columns if any(x in f for x in ['arrivals_', 'departures_'])]
        time_features = [f for f in self.feature_columns if any(x in f for x in ['hour', 'day_', 'month', 'year', 'weekend'])]
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
        # Ensure datetime columns
        df['fecha_origen_recorrido'] = pd.to_datetime(df['fecha_origen_recorrido'])
        df['fecha_destino_recorrido'] = pd.to_datetime(df['fecha_destino_recorrido'])
        
        # Create time range
        start_time = df['fecha_origen_recorrido'].min()
        end_time = df['fecha_destino_recorrido'].max()
        
        # Round to nearest delta_t interval
        start_time = start_time.floor(f'{self.delta_t_minutes}min')
        end_time = end_time.ceil(f'{self.delta_t_minutes}min')
        
        # Create time index
        time_index = pd.date_range(start_time, end_time, freq=f'{self.delta_t_minutes}min')
        
        if station_id is not None:
            # Single station case
            print(f"  Processing single station: {station_id}")
            delta_df = pd.DataFrame({'datetime': time_index, 'station_id': station_id})
            print(f"  Created base time grid: {len(delta_df)} intervals")
            
            # Count departures (trips starting from this station)
            departures_df = df[df['id_estacion_origen'] == station_id].copy()
            print(f"  Found {len(departures_df)} departure trips for station {station_id}")
            departures_df['datetime'] = departures_df['fecha_origen_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            departures_count = departures_df.groupby('datetime').size().reset_index(name='departures')
            # Add station_id to match delta_df structure
            departures_count['station_id'] = station_id
            print(f"  Aggregated departures into {len(departures_count)} time intervals")
            
            # Count arrivals (trips ending at this station)
            arrivals_df = df[df['id_estacion_destino'] == station_id].copy()
            print(f"  Found {len(arrivals_df)} arrival trips for station {station_id}")
            arrivals_df['datetime'] = arrivals_df['fecha_destino_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            arrivals_count = arrivals_df.groupby('datetime').size().reset_index(name='arrivals')
            # Add station_id to match delta_df structure
            arrivals_count['station_id'] = station_id
            print(f"  Aggregated arrivals into {len(arrivals_count)} time intervals")
            
        else:
            # Multi-station case - get all unique stations
            stations = set(df['id_estacion_origen'].unique()) | set(df['id_estacion_destino'].unique())
            
            # Create base DataFrame
            delta_df = pd.DataFrame([
                {'datetime': dt, 'station_id': station}
                for dt in time_index
                for station in stations
            ])
            
            # Count departures for all stations
            departures_df = df.copy()
            departures_df['datetime'] = departures_df['fecha_origen_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            departures_count = (departures_df.groupby(['datetime', 'id_estacion_origen'])
                              .size().reset_index(name='departures'))
            departures_count.rename(columns={'id_estacion_origen': 'station_id'}, inplace=True)
            
            # Count arrivals for all stations
            arrivals_df = df.copy()
            arrivals_df['datetime'] = arrivals_df['fecha_destino_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            arrivals_count = (arrivals_df.groupby(['datetime', 'id_estacion_destino'])
                            .size().reset_index(name='arrivals'))
            arrivals_count.rename(columns={'id_estacion_destino': 'station_id'}, inplace=True)
        
        # Merge counts (now both single and multi-station cases have station_id column)
        print(f"  Merging departure counts...")
        delta_df = delta_df.merge(departures_count, on=['datetime', 'station_id'], how='left')
        print(f"  Merging arrival counts...")
        delta_df = delta_df.merge(arrivals_count, on=['datetime', 'station_id'], how='left')
        
        print(f"  After merging: {len(delta_df)} rows, columns: {list(delta_df.columns)}")
        
        # Fill missing values
        departures_nulls = delta_df['departures'].isnull().sum()
        arrivals_nulls = delta_df['arrivals'].isnull().sum()
        print(f"  Nulls before filling: departures={departures_nulls}, arrivals={arrivals_nulls}")
        
        delta_df['departures'] = delta_df['departures'].fillna(0)
        delta_df['arrivals'] = delta_df['arrivals'].fillna(0)
        
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
            weather_df = df[['fecha_origen_recorrido'] + available_weather_cols].copy()
            weather_df['datetime'] = weather_df['fecha_origen_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            weather_agg = weather_df.groupby('datetime')[available_weather_cols].mean().reset_index()
            delta_df = delta_df.merge(weather_agg, on='datetime', how='left')
            
            # Check for nulls after weather merge
            for col in available_weather_cols:
                null_count = delta_df[col].isnull().sum()
                if null_count > 0:
                    print(f"    {col}: {null_count} nulls after merge ({null_count/len(delta_df)*100:.1f}%)")
            
            # Forward fill weather data
            print("  Forward filling weather data...")
            delta_df[available_weather_cols] = delta_df[available_weather_cols].fillna(method='ffill')
            
            # Check remaining nulls after forward fill
            remaining_nulls = delta_df[available_weather_cols].isnull().sum().sum()
            if remaining_nulls > 0:
                print(f"    {remaining_nulls} weather nulls remain after forward fill")
            else:
                print("    All weather nulls filled by forward fill")
        else:
            print(f"  None of the key weather features found in data: {key_weather_features}")
            print("  Available columns:", [col for col in df.columns if 'weather' in col.lower()])
        
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
            for feature in available_features:
                lag_periods = 7 * 24 * (60 // self.delta_t_minutes)  # 1 week in delta_t periods
                lag_feature_name = f'{feature}_1w_lag'
                df[lag_feature_name] = df.groupby('station_id')[feature].shift(lag_periods)
                
                # Print null count for lagged weather features
                null_count = df[lag_feature_name].isnull().sum()
                print(f"    {lag_feature_name}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
        else:
            print("  No key weather features available - skipping climate features")
        
        return df
    
    def _add_station_features(self, df):
        """
        Add station-wise features focusing on past vs future state.
        """
        print("Adding station features...")
        
        # Last delta T features
        df['arrivals_last_dt'] = df.groupby('station_id')['arrivals'].shift(1)
        df['departures_last_dt'] = df.groupby('station_id')['departures'].shift(1)
        
        # Print null counts for last delta T features
        print(f"  arrivals_last_dt: {df['arrivals_last_dt'].isnull().sum()} nulls ({df['arrivals_last_dt'].isnull().sum()/len(df)*100:.1f}%)")
        print(f"  departures_last_dt: {df['departures_last_dt'].isnull().sum()} nulls ({df['departures_last_dt'].isnull().sum()/len(df)*100:.1f}%)")
        
        # 1 week lag predictions (same time last week)
        lag_periods = 7 * 24 * (60 // self.delta_t_minutes)  # 1 week in delta_t periods
        df['arrivals_1w_lag'] = df.groupby('station_id')['arrivals'].shift(lag_periods)
        df['departures_1w_lag'] = df.groupby('station_id')['departures'].shift(lag_periods)
        
        # Print null counts for 1 week lag features
        print(f"  arrivals_1w_lag: {df['arrivals_1w_lag'].isnull().sum()} nulls ({df['arrivals_1w_lag'].isnull().sum()/len(df)*100:.1f}%)")
        print(f"  departures_1w_lag: {df['departures_1w_lag'].isnull().sum()} nulls ({df['departures_1w_lag'].isnull().sum()/len(df)*100:.1f}%)")
        
        # Create time slot identifier for same-time-across-weeks moving averages
        # Combine hour and day_of_week to create unique time slots
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        df['time_slot'] = df['hour'] * 7 + df['day_of_week']  # Unique identifier for each hour-day combination
        
        print("  Computing moving averages for same time slots across weeks...")
        
        # Moving averages computed over same time slots across different weeks
        # For today 11am Tuesday, this will average [last Tuesday 11am, 2 weeks ago Tuesday 11am, ...]
        
        def compute_weekly_moving_average(group, window, col_name):
            """Compute moving average for same time slots across weeks"""
            # Sort by datetime to ensure proper chronological order
            group_sorted = group.sort_values('datetime')
            # Compute rolling mean
            ma = group_sorted[col_name].rolling(window=window, min_periods=1).mean()
            return ma
        
        # 4-week moving average for same time slots
        print("    Computing 4-week moving averages...")
        ma4_arrivals = df.groupby(['station_id', 'time_slot']).apply(
            lambda x: compute_weekly_moving_average(x, 4, 'arrivals')
        ).reset_index(level=[0,1], drop=True)
        ma4_departures = df.groupby(['station_id', 'time_slot']).apply(
            lambda x: compute_weekly_moving_average(x, 4, 'departures')
        ).reset_index(level=[0,1], drop=True)
        
        # Align indices and assign
        df['arrivals_ma4_weekly'] = ma4_arrivals.reindex(df.index)
        df['departures_ma4_weekly'] = ma4_departures.reindex(df.index)
        
        # 12-week moving average for same time slots
        print("    Computing 12-week moving averages...")
        ma12_arrivals = df.groupby(['station_id', 'time_slot']).apply(
            lambda x: compute_weekly_moving_average(x, 12, 'arrivals')
        ).reset_index(level=[0,1], drop=True)
        ma12_departures = df.groupby(['station_id', 'time_slot']).apply(
            lambda x: compute_weekly_moving_average(x, 12, 'departures')
        ).reset_index(level=[0,1], drop=True)
        
        # Align indices and assign
        df['arrivals_ma12_weekly'] = ma12_arrivals.reindex(df.index)
        df['departures_ma12_weekly'] = ma12_departures.reindex(df.index)
        
        # Print null counts for moving averages
        print(f"  arrivals_ma4_weekly: {df['arrivals_ma4_weekly'].isnull().sum()} nulls ({df['arrivals_ma4_weekly'].isnull().sum()/len(df)*100:.1f}%)")
        print(f"  departures_ma4_weekly: {df['departures_ma4_weekly'].isnull().sum()} nulls ({df['departures_ma4_weekly'].isnull().sum()/len(df)*100:.1f}%)")
        print(f"  arrivals_ma12_weekly: {df['arrivals_ma12_weekly'].isnull().sum()} nulls ({df['arrivals_ma12_weekly'].isnull().sum()/len(df)*100:.1f}%)")
        print(f"  departures_ma12_weekly: {df['departures_ma12_weekly'].isnull().sum()} nulls ({df['departures_ma12_weekly'].isnull().sum()/len(df)*100:.1f}%)")
        
        # Conditionally fill missing values in station features
        station_features = [
            'arrivals_last_dt', 'departures_last_dt',
            'arrivals_1w_lag', 'departures_1w_lag',
            'arrivals_ma4_weekly', 'departures_ma4_weekly',
            'arrivals_ma12_weekly', 'departures_ma12_weekly'
        ]
        
        if self.fill_nulls:
            print(f"  Filling station feature nulls with strategy: {self.fill_strategy}...")
            for feature in station_features:
                null_count_before = df[feature].isnull().sum()
                if null_count_before > 0:
                    if self.fill_strategy == 'zero':
                        df[feature] = df[feature].fillna(0)
                    elif self.fill_strategy == 'mean':
                        mean_val = df[feature].mean()
                        df[feature] = df[feature].fillna(mean_val)
                    elif self.fill_strategy == 'median':
                        median_val = df[feature].median()
                        df[feature] = df[feature].fillna(median_val)
                    elif self.fill_strategy == 'forward_fill':
                        df[feature] = df.groupby('station_id')[feature].fillna(method='ffill').fillna(0)
                    
                    print(f"    {feature}: filled {null_count_before} nulls")
        else:
            print("  Keeping station feature nulls as requested...")
            for feature in station_features:
                null_count = df[feature].isnull().sum()
                if null_count > 0:
                    print(f"    {feature}: {null_count} nulls preserved")
        
        # Drop temporary columns
        df = df.drop(['time_slot'], axis=1)
        
        return df
    
    def _add_time_features(self, df):
        """
        Add time-wise features using cyclical encoding.
        """
        print("Adding time features...")
        
        # Extract basic time components (check if already exist to avoid duplication)
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        # Boolean features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for periodic features
        # Hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month (30-day cycle approximation)
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        

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
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"  Found infinite values in columns: {inf_counts}")
            print("  Replacing infinite values with NaN...")
            df = df.replace([np.inf, -np.inf], np.nan)
        else:
            print("  No infinite values found")
        
        # Check remaining NaN values before final fill
        print("  Checking remaining null values before final cleanup...")
        total_nulls = 0
        for col in numeric_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"    {col}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
                total_nulls += null_count
        
        if total_nulls > 0:
            print(f"  Total nulls found: {total_nulls}")
            
            if self.fill_nulls:
                print(f"  Filling remaining NaN values using strategy: {self.fill_strategy}")
                
                if self.fill_strategy == 'zero':
                    df[numeric_columns] = df[numeric_columns].fillna(0)
                elif self.fill_strategy == 'mean':
                    for col in numeric_columns:
                        if df[col].isnull().sum() > 0:
                            mean_val = df[col].mean()
                            df[col] = df[col].fillna(mean_val)
                            print(f"    {col}: filled with mean={mean_val:.3f}")
                elif self.fill_strategy == 'median':
                    for col in numeric_columns:
                        if df[col].isnull().sum() > 0:
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)
                            print(f"    {col}: filled with median={median_val:.3f}")
                elif self.fill_strategy == 'forward_fill':
                    # Sort by station_id and datetime first for proper forward fill
                    df_sorted = df.sort_values(['station_id', 'datetime'])
                    for col in numeric_columns:
                        if df[col].isnull().sum() > 0:
                            df_sorted[col] = df_sorted.groupby('station_id')[col].fillna(method='ffill')
                            # Fill any remaining nulls with 0 (start of each station's time series)
                            remaining_nulls = df_sorted[col].isnull().sum()
                            if remaining_nulls > 0:
                                df_sorted[col] = df_sorted[col].fillna(0)
                                print(f"    {col}: forward filled + {remaining_nulls} remaining filled with 0")
                            else:
                                print(f"    {col}: forward filled successfully")
                    df = df_sorted.reindex(df.index)  # Restore original order
                
                print("  All nulls filled")
            else:
                print("  Null filling disabled - keeping null values for model to handle")
                print("  WARNING: Some models may not handle null values well!")
        else:
            print("  No remaining null values found")
        
        # Sort by datetime and station
        print("  Sorting by station_id and datetime...")
        df = df.sort_values(['station_id', 'datetime']).reset_index(drop=True)
        
        # Final validation - check for any remaining nulls
        final_nulls = df.isnull().sum().sum()
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
            df (pd.DataFrame): Processed DataFrame
            target_cols (list): Target column names
            
        Returns:
            tuple: (X, y, metadata) where X=features, y=targets, metadata=datetime/station info
        """
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in 
                       ['datetime', 'station_id'] + target_cols]
        
        X = df[feature_cols].copy()
        y = df[target_cols].copy()
        metadata = df[['datetime', 'station_id']].copy()
        
        print(f"Prepared for training: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, metadata


def load_and_process_data(data_path, station_id=None, delta_t_minutes=30, 
                         fill_nulls=True, fill_strategy='zero'):
    """
    Convenience function to load and process data.
    
    Args:
        data_path (str): Path to the input data file (CSV or Parquet)
        station_id (int, optional): Specific station ID to process
        delta_t_minutes (int): Time interval in minutes
        fill_nulls (bool): Whether to fill null values (default: True)
        fill_strategy (str): Strategy for filling nulls ('zero', 'mean', 'median', 'forward_fill')
        
    Returns:
        tuple: (processed_df, feature_engineer)
    """
    print(f"Loading data from {data_path}...")
    
    # Determine file format and load accordingly
    if data_path.lower().endswith('.parquet'):
        print("  Detected parquet format")
        df = pd.read_parquet(data_path)
    elif data_path.lower().endswith('.csv'):
        print("  Detected CSV format")
        df = pd.read_csv(data_path)
    else:
        # Try parquet first, then CSV as fallback
        try:
            print("  File extension not recognized, trying parquet format...")
            df = pd.read_parquet(data_path)
            print("  Successfully loaded as parquet")
        except:
            try:
                print("  Parquet failed, trying CSV format...")
                df = pd.read_csv(data_path)
                print("  Successfully loaded as CSV")
            except Exception as e:
                raise ValueError(f"Could not load file {data_path}. Tried both parquet and CSV formats. Error: {str(e)}")
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize feature engineer
    fe = FeatureEngineer(delta_t_minutes=delta_t_minutes, 
                        fill_nulls=fill_nulls, 
                        fill_strategy=fill_strategy)
    
    # Process data
    processed_df = fe.process_data(df, station_id=station_id)
    
    return processed_df, fe


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Use load_and_process_data() to process your trip data") 