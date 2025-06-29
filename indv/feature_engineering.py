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
    
    def __init__(self, delta_t_minutes=30):
        """
        Initialize the feature engineer.
        
        Args:
            delta_t_minutes (int): Time interval in minutes (default: 30)
        """
        self.delta_t_minutes = delta_t_minutes
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
        
        # Filter for specific station if provided
        if station_id is not None:
            df = df[
                (df['id_estacion_origen'] == station_id) | 
                (df['id_estacion_destino'] == station_id)
            ].copy()
            print(f"Filtered data for station {station_id}: {len(df)} trips")
        
        # Convert to delta T format
        delta_df = self._convert_to_delta_t(df, station_id)
        print(f"Created delta T format: {len(delta_df)} time intervals")
        
        # Add all feature types
        delta_df = self._add_climate_features(delta_df)
        delta_df = self._add_station_features(delta_df)
        delta_df = self._add_time_features(delta_df)
        delta_df = self._add_user_features(delta_df)
        
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
        station_features = [f for f in self.feature_columns if any(x in f for x in ['arrivals_', 'departures_', 'popularity'])]
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
            delta_df = pd.DataFrame({'datetime': time_index, 'station_id': station_id})
            
            # Count departures (trips starting from this station)
            departures_df = df[df['id_estacion_origen'] == station_id].copy()
            departures_df['datetime'] = departures_df['fecha_origen_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            departures_count = departures_df.groupby('datetime').size().reset_index(name='departures')
            
            # Count arrivals (trips ending at this station)
            arrivals_df = df[df['id_estacion_destino'] == station_id].copy()
            arrivals_df['datetime'] = arrivals_df['fecha_destino_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            arrivals_count = arrivals_df.groupby('datetime').size().reset_index(name='arrivals')
            
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
        
        # Merge counts
        delta_df = delta_df.merge(departures_count, on=['datetime', 'station_id'], how='left')
        delta_df = delta_df.merge(arrivals_count, on=['datetime', 'station_id'], how='left')
        
        # Fill missing values
        delta_df['departures'] = delta_df['departures'].fillna(0)
        delta_df['arrivals'] = delta_df['arrivals'].fillna(0)
        
        # Add weather data (assuming it's already in the input df)
        weather_cols = [col for col in df.columns if col.startswith('weather_')]
        if weather_cols:
            print(f"  Found {len(weather_cols)} weather columns, merging weather data...")
            # Get weather data by taking the mean for each time interval
            weather_df = df[['fecha_origen_recorrido'] + weather_cols].copy()
            weather_df['datetime'] = weather_df['fecha_origen_recorrido'].dt.floor(f'{self.delta_t_minutes}min')
            weather_agg = weather_df.groupby('datetime')[weather_cols].mean().reset_index()
            delta_df = delta_df.merge(weather_agg, on='datetime', how='left')
            
            # Check for nulls after weather merge
            for col in weather_cols:
                null_count = delta_df[col].isnull().sum()
                if null_count > 0:
                    print(f"    {col}: {null_count} nulls after merge ({null_count/len(delta_df)*100:.1f}%)")
            
            # Forward fill weather data
            print("  Forward filling weather data...")
            delta_df[weather_cols] = delta_df[weather_cols].fillna(method='ffill')
            
            # Check remaining nulls after forward fill
            remaining_nulls = delta_df[weather_cols].isnull().sum().sum()
            if remaining_nulls > 0:
                print(f"    {remaining_nulls} weather nulls remain after forward fill")
            else:
                print("    All weather nulls filled by forward fill")
        else:
            print("  No weather columns found in input data")
        
        return delta_df
    
    def _add_climate_features(self, df):
        """
        Add climate-related features.
        Focus on temperature, humidity, and apparent temperature as specified.
        """
        print("Adding climate features...")
        
        # Use key weather features as specified in the approach
        key_weather_features = [
            'weather_temperature_2m',
            'weather_relative_humidity_2m', 
            'weather_apparent_temperature'
        ]
        
        # Ensure these features exist
        for feature in key_weather_features:
            if feature not in df.columns:
                df[feature] = 0  # Default value if missing
        
        # Add lagged weather features (1 week lag)
        for feature in key_weather_features:
            lag_periods = 7 * 24 * (60 // self.delta_t_minutes)  # 1 week in delta_t periods
            lag_feature_name = f'{feature}_1w_lag'
            df[lag_feature_name] = df.groupby('station_id')[feature].shift(lag_periods)
            
            # Print null count for lagged weather features
            null_count = df[lag_feature_name].isnull().sum()
            print(f"  {lag_feature_name}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
        
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
        
        # Fill missing values with 0 (after printing null counts)
        station_features = [
            'arrivals_last_dt', 'departures_last_dt',
            'arrivals_1w_lag', 'departures_1w_lag',
            'arrivals_ma4_weekly', 'departures_ma4_weekly',
            'arrivals_ma12_weekly', 'departures_ma12_weekly'
        ]
        
        print("  Filling null values with 0...")
        for feature in station_features:
            null_count_before = df[feature].isnull().sum()
            df[feature] = df[feature].fillna(0)
            print(f"    {feature}: filled {null_count_before} nulls")
        
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
        
        # Year (relative to start year for trend)
        start_year = df['year'].min()
        df['year_normalized'] = (df['year'] - start_year) / 10  # Scale by decade
        
        return df
    
    def _add_user_features(self, df):
        """
        Add user-wise features (simplified version for now).
        Note: This requires joining back with original trip data.
        """
        print("Adding user features (placeholder)...")
        
        # Placeholder features - these would require more complex joins with trip data
        # For now, adding basic aggregated features that can be computed from the delta_t format
        
        # Station popularity (total trips per station)
        station_popularity = df.groupby('station_id')[['arrivals', 'departures']].sum()
        station_popularity['total_trips'] = station_popularity['arrivals'] + station_popularity['departures']
        station_popularity['popularity_score'] = (station_popularity['total_trips'] / 
                                                 station_popularity['total_trips'].max())
        
        # Merge back to main dataframe
        df = df.merge(station_popularity[['popularity_score']].reset_index(), 
                     on='station_id', how='left')
        
        # Print null counts for user features
        print(f"  popularity_score: {df['popularity_score'].isnull().sum()} nulls ({df['popularity_score'].isnull().sum()/len(df)*100:.1f}%)")
        
        # Fill any nulls in popularity score
        if df['popularity_score'].isnull().sum() > 0:
            print("  Filling popularity_score nulls with 0...")
            df['popularity_score'] = df['popularity_score'].fillna(0)
        
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
            print("  Filling remaining NaN values with 0...")
            df[numeric_columns] = df[numeric_columns].fillna(0)
            print("  All nulls filled")
        else:
            print("  No remaining null values found")
        
        # Sort by datetime and station
        print("  Sorting by station_id and datetime...")
        df = df.sort_values(['station_id', 'datetime']).reset_index(drop=True)
        
        # Final validation - check for any remaining nulls
        final_nulls = df.isnull().sum().sum()
        if final_nulls > 0:
            print(f"  WARNING: {final_nulls} null values remain after cleaning!")
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


def load_and_process_data(data_path, station_id=None, delta_t_minutes=30):
    """
    Convenience function to load and process data.
    
    Args:
        data_path (str): Path to the input data file (CSV or Parquet)
        station_id (int, optional): Specific station ID to process
        delta_t_minutes (int): Time interval in minutes
        
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
    fe = FeatureEngineer(delta_t_minutes=delta_t_minutes)
    
    # Process data
    processed_df = fe.process_data(df, station_id=station_id)
    
    return processed_df, fe


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Use load_and_process_data() to process your trip data") 