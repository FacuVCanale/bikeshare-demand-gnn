"""
Weather data collection and processing module for EcoBici-AI.

This module provides functionality to collect historical weather data from Open-Meteo API
and match it with trip data by finding the closest weather conditions.
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
from datetime import datetime, timedelta


class WeatherDataCollector:
    """
    Collects and processes weather data for EcoBici trip analysis.
    
    This class handles:
    - Historical weather data collection from Open-Meteo API
    - Caching of API responses to avoid repeated requests
    - Matching weather conditions with trip timestamps
    - Weather data preprocessing and cleaning
    """
    
    def __init__(self, cache_dir: str = '.cache'):
        """
        Initialize the weather data collector.
        
        Args:
            cache_dir: Directory to cache API responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # setup cached session for API requests
        cache_session = requests_cache.CachedSession(
            str(self.cache_dir / 'weather_cache'), 
            expire_after=-1
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
    def collect_historical_weather(
        self,
        latitude: float = -34.6131,
        longitude: float = -58.3772,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Collect historical weather data from Open-Meteo API.
        
        Args:
            latitude: Latitude for Buenos Aires
            longitude: Longitude for Buenos Aires
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_path: Optional path to save the data
            
        Returns:
            DataFrame with hourly weather data
        """
        print(f"collecting weather data from {start_date} to {end_date}...")
        
        # configure API request
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                "apparent_temperature", "precipitation", "weather_code",
                "pressure_msl", "cloud_cover", "cloud_cover_low", 
                "cloud_cover_mid", "cloud_cover_high", "vapour_pressure_deficit",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", 
                "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
                "sunshine_duration", "is_day", "direct_radiation"
            ],
            "models": "best_match",
            "timezone": "auto"
        }
        
        # make API request
        responses = self.openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        print(f"coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"elevation: {response.Elevation()} m asl")
        print(f"timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
        
        # process hourly data
        hourly = response.Hourly()
        
        # create date range
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }
        
        # extract all variables
        variable_names = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", 
            "apparent_temperature", "precipitation", "weather_code",
            "pressure_msl", "cloud_cover", "cloud_cover_low", 
            "cloud_cover_mid", "cloud_cover_high", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", 
            "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
            "sunshine_duration", "is_day", "direct_radiation"
        ]
        
        for i, var_name in enumerate(variable_names):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
        
        # create dataframe
        weather_df = pd.DataFrame(data=hourly_data)
        
        # remove constant columns (like snowfall in Buenos Aires)
        constant_columns = []
        for column in weather_df.columns:
            if column != 'date' and len(weather_df[column].unique()) == 1:
                constant_columns.append(column)
                print(f"removing constant column '{column}' with value: {weather_df[column].iloc[0]}")
        
        if constant_columns:
            weather_df = weather_df.drop(columns=constant_columns)
        
        # save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            weather_df.to_csv(output_path, index=False)
            print(f"weather data saved to: {output_path}")
        
        return weather_df
    
    def load_weather_data(self, file_path: str) -> pd.DataFrame:
        """
        Load weather data from CSV file.
        
        Args:
            file_path: Path to the weather CSV file
            
        Returns:
            DataFrame with weather data
        """
        weather_df = pd.read_csv(file_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        return weather_df
    
    def get_weather_data(
        self,
        file_path: str = "../../data/raw/meteo/hourly_open_meteo.csv",
        latitude: float = -34.6131,
        longitude: float = -58.3772,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Smart weather data loader: uses existing CSV or fetches from API if needed.
        
        Args:
            file_path: Path to the weather CSV file
            latitude: Latitude for Buenos Aires
            longitude: Longitude for Buenos Aires  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, always fetch from API regardless of existing file
            
        Returns:
            DataFrame with weather data
        """
        file_path = Path(file_path)
        
        # check if file exists and force_refresh is False
        if file_path.exists() and not force_refresh:
            print(f"loading existing weather data from: {file_path}")
            try:
                weather_df = self.load_weather_data(str(file_path))
                
                # validate data completeness
                expected_start = pd.to_datetime(start_date)
                expected_end = pd.to_datetime(end_date)
                actual_start = weather_df['date'].min()
                actual_end = weather_df['date'].max()
                
                # check if existing data covers the requested period
                if actual_start <= expected_start and actual_end >= expected_end:
                    print(f"existing data covers requested period ({start_date} to {end_date})")
                    # filter to requested period
                    mask = (weather_df['date'] >= expected_start) & (weather_df['date'] <= expected_end)
                    return weather_df[mask].copy()
                else:
                    print(f"existing data incomplete: {actual_start.date()} to {actual_end.date()}")
                    print(f"requested period: {start_date} to {end_date}")
                    print("fetching fresh data from API...")
            except Exception as e:
                print(f"error loading existing file: {e}")
                print("fetching fresh data from API...")
        
        # fetch from API (file doesn't exist, is incomplete, or force_refresh=True)
        if force_refresh:
            print("force refresh enabled - fetching fresh data from API...")
        elif not file_path.exists():
            print(f"file not found: {file_path}")
            print("fetching data from API...")
            
        return self.collect_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            output_path=str(file_path)
        )
    
    def match_weather_to_trips(
        self, 
        trips_df: pd.DataFrame, 
        weather_df: pd.DataFrame,
        date_column: str = 'fecha_origen_recorrido'
    ) -> pd.DataFrame:
        """
        Match weather conditions to trip data by finding the closest hour.
        
        Args:
            trips_df: DataFrame with trip data
            weather_df: DataFrame with weather data
            date_column: Column name with trip start datetime
            
        Returns:
            DataFrame with trips and matching weather conditions
        """
        print("matching weather conditions to trips...")
        
        # ensure datetime columns are properly formatted
        if not pd.api.types.is_datetime64_any_dtype(trips_df[date_column]):
            trips_df[date_column] = pd.to_datetime(trips_df[date_column])
        
        if not pd.api.types.is_datetime64_any_dtype(weather_df['date']):
            weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # create a copy to avoid modifying original data
        trips_with_weather = trips_df.copy()
        
        # convert to timezone-naive for matching (assuming both are in same timezone)
        trips_dates = trips_with_weather[date_column].dt.tz_localize(None) if trips_with_weather[date_column].dt.tz is not None else trips_with_weather[date_column]
        weather_dates = weather_df['date'].dt.tz_localize(None) if weather_df['date'].dt.tz is not None else weather_df['date']
        
        # round trip times to nearest hour for matching
        trips_with_weather['weather_match_hour'] = trips_dates.dt.round('H')
        
        # create weather lookup dictionary for faster matching
        weather_lookup = weather_df.set_index(weather_dates.dt.round('H')).to_dict('index')
        
        # function to get weather data for a given hour
        def get_weather_for_hour(hour):
            if pd.isna(hour):
                return {col: np.nan for col in weather_df.columns if col != 'date'}
            
            weather_data = weather_lookup.get(hour, {})
            if not weather_data:
                # if exact hour not found, find closest
                time_diffs = np.abs((weather_dates - hour).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                weather_data = weather_df.iloc[closest_idx].to_dict()
                weather_data.pop('date', None)
            else:
                weather_data.pop('date', None)
            
            return weather_data
        
        # match weather data to trips
        print("finding weather conditions for each trip...")
        weather_matches = trips_with_weather['weather_match_hour'].apply(get_weather_for_hour)
        
        # convert to dataframe and add to trips
        weather_features_df = pd.DataFrame(weather_matches.tolist(), index=trips_with_weather.index)
        
        # add weather prefix to column names to avoid conflicts
        weather_features_df.columns = [f'weather_{col}' for col in weather_features_df.columns]
        
        # combine trips with weather data
        trips_with_weather = pd.concat([trips_with_weather, weather_features_df], axis=1)
        
        # remove temporary column
        trips_with_weather = trips_with_weather.drop('weather_match_hour', axis=1)
        
        # report matching statistics
        matched_trips = trips_with_weather['weather_temperature_2m'].notna().sum()
        total_trips = len(trips_with_weather)
        print(f"matched weather data for {matched_trips:,} out of {total_trips:,} trips ({matched_trips/total_trips*100:.1f}%)")
        
        return trips_with_weather
    
    def get_weather_summary(self, weather_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of weather data.
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            Dictionary with weather summary statistics
        """
        summary = {
            'total_records': len(weather_df),
            'date_range': {
                'start': weather_df['date'].min(),
                'end': weather_df['date'].max()
            },
            'temperature_stats': {
                'min': weather_df['temperature_2m'].min(),
                'max': weather_df['temperature_2m'].max(),
                'mean': weather_df['temperature_2m'].mean(),
                'std': weather_df['temperature_2m'].std()
            },
            'precipitation_stats': {
                'total_mm': weather_df['precipitation'].sum(),
                'rainy_hours': (weather_df['precipitation'] > 0).sum(),
                'max_hourly_mm': weather_df['precipitation'].max()
            },
            'wind_stats': {
                'mean_speed_10m': weather_df['wind_speed_10m'].mean(),
                'max_speed_10m': weather_df['wind_speed_10m'].max(),
                'mean_gusts_10m': weather_df['wind_gusts_10m'].mean()
            },
            'humidity_stats': {
                'mean': weather_df['relative_humidity_2m'].mean(),
                'min': weather_df['relative_humidity_2m'].min(),
                'max': weather_df['relative_humidity_2m'].max()
            }
        }
        
        return summary
    
    def create_weather_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional weather features for machine learning.
        
        Args:
            weather_df: DataFrame with raw weather data
            
        Returns:
            DataFrame with additional weather features
        """
        df = weather_df.copy()
        
        # time-based features
        if 'date' in df.columns:
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['season'] = df['month'].map({12: 'summer', 1: 'summer', 2: 'summer',
                                          3: 'autumn', 4: 'autumn', 5: 'autumn',
                                          6: 'winter', 7: 'winter', 8: 'winter',
                                          9: 'spring', 10: 'spring', 11: 'spring'})
        
        # comfort indices
        if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
            # simple wind chill approximation (for temps below 10C)
            df['feels_like_wind_chill'] = np.where(
                df['temperature_2m'] < 10,
                13.12 + 0.6215 * df['temperature_2m'] - 11.37 * (df['wind_speed_10m'] ** 0.16) + 0.3965 * df['temperature_2m'] * (df['wind_speed_10m'] ** 0.16),
                df['temperature_2m']
            )
        
        # cloud cover ratio: low vs high impact on cycling
        # high clouds trap heat (warmer), low clouds reflect sun (cooler)
        if all(col in df.columns for col in ['cloud_cover_low', 'cloud_cover_high']):
            df['cloud_cooling_warming_ratio'] = df['cloud_cover_low'] / (df['cloud_cover_high'] + 0.01)
        
        # weather categories (WMO codes)
        if 'weather_code' in df.columns:
            df['weather_category'] = df['weather_code'].map({
                0: 'clear', 1: 'mainly_clear', 2: 'partly_cloudy', 3: 'overcast',
                45: 'fog', 48: 'fog', 51: 'drizzle', 53: 'drizzle', 55: 'drizzle',
                56: 'freezing_drizzle', 57: 'freezing_drizzle', 61: 'rain', 63: 'rain', 
                65: 'rain', 66: 'freezing_rain', 67: 'freezing_rain', 71: 'snow', 
                73: 'snow', 75: 'snow', 77: 'snow', 80: 'rain_showers', 81: 'rain_showers', 
                82: 'rain_showers', 85: 'snow_showers', 86: 'snow_showers', 95: 'thunderstorm',
                96: 'thunderstorm', 99: 'thunderstorm'
            })
        
        # binary weather conditions
        if 'precipitation' in df.columns:
            df['is_raining'] = (df['precipitation'] > 0).astype(int)
            df['is_heavy_rain'] = (df['precipitation'] > 2.5).astype(int)  # >2.5mm/h is moderate rain
        
        if 'wind_speed_10m' in df.columns:
            df['is_windy'] = (df['wind_speed_10m'] > 20).astype(int)  # >20 km/h is considered windy
        
        if 'temperature_2m' in df.columns:
            df['is_cold'] = (df['temperature_2m'] < 10).astype(int)
            df['is_hot'] = (df['temperature_2m'] > 25).astype(int)
            df['is_comfortable'] = ((df['temperature_2m'] >= 15) & (df['temperature_2m'] <= 25)).astype(int)
        
        return df 
    
def analyze_weather_impact(trips_df):
    """analyze impact of weather conditions on bike usage and trip duration"""
    results = {}
    
    # temperature analysis
    if 'weather_temperature_2m' in trips_df.columns:
        temp_ranges = pd.cut(trips_df['weather_temperature_2m'],
                           bins=[-float('inf'), 10, 15, 20, 25, 30, float('inf')],
                           labels=['<10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C'])
        results['temperature'] = {
            'usage': temp_ranges.value_counts().sort_index(),
            'duration': trips_df.groupby(temp_ranges)['duracion_recorrido'].mean() if 'duracion_recorrido' in trips_df.columns else None
        }
    
    # rain analysis
    if 'weather_is_raining' in trips_df.columns:
        results['rain'] = {
            'usage': trips_df['weather_is_raining'].value_counts(),
            'duration': trips_df.groupby('weather_is_raining')['duracion_recorrido'].mean() if 'duracion_recorrido' in trips_df.columns else None
        }
    
    # wind analysis
    if 'weather_wind_speed_10m' in trips_df.columns:
        wind_ranges = pd.cut(trips_df['weather_wind_speed_10m'],
                           bins=[0, 10, 20, 30, float('inf')],
                           labels=['calm (0-10)', 'light (10-20)', 'moderate (20-30)', 'strong (>30)'])
        results['wind'] = {
            'usage': wind_ranges.value_counts().sort_index()
        }
    
    return results

def print_weather_analysis(results, total_trips):
    """print formatted weather analysis results"""
    # temperature impact
    if 'temperature' in results:
        print("temperature impact on bike usage:")
        for temp_range, count in results['temperature']['usage'].items():
            percentage = count / total_trips * 100
            print(f"  {temp_range}: {count:,} trips ({percentage:.1f}%)")
        
        if results['temperature']['duration'] is not None:
            print("\naverage trip duration by temperature:")
            for temp_range, avg_duration in results['temperature']['duration'].items():
                print(f"  {temp_range}: {avg_duration:.0f} seconds")
    
    # rain impact
    if 'rain' in results:
        print("\nrain impact on bike usage:")
        for is_raining, count in results['rain']['usage'].items():
            condition = "rainy" if is_raining else "dry"
            percentage = count / total_trips * 100
            print(f"  {condition} conditions: {count:,} trips ({percentage:.1f}%)")
        
        if results['rain']['duration'] is not None:
            print("\naverage trip duration by rain conditions:")
            for is_raining, avg_duration in results['rain']['duration'].items():
                condition = "rainy" if is_raining else "dry"
                print(f"  {condition}: {avg_duration:.0f} seconds")
    
    # wind impact
    if 'wind' in results:
        print("\nwind impact on bike usage:")
        for wind_range, count in results['wind']['usage'].items():
            percentage = count / total_trips * 100
            print(f"  {wind_range} km/h: {count:,} trips ({percentage:.1f}%)")


def load_and_summarize_weather_data(weather_collector: WeatherDataCollector, weather_file_path: str = "../../data/raw/meteo/hourly_open_meteo.csv"):
    """
    load weather data and print summary statistics
    
    parameters:
        weather_collector: weather data collector
        weather_file_path: path to weather data file
    
    returns:
        weather_df: dataframe with weather data
        weather_summary: dictionary with weather statistics
    """
    # initialize collector and load data
    weather_df = weather_collector.load_weather_data(weather_file_path)
    
    # print basic info
    print(f"\nweather dataframe shape: {weather_df.shape}")
    print(f"weather columns: {list(weather_df.columns)}")
    print(f"date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
    
    # get and print summary stats
    weather_summary = weather_collector.get_weather_summary(weather_df)
    print(f"\nweather summary:")
    print(f"total records: {weather_summary['total_records']:,}")
    print(f"temperature range: {weather_summary['temperature_stats']['min']:.1f}°C to {weather_summary['temperature_stats']['max']:.1f}°C")
    print(f"mean temperature: {weather_summary['temperature_stats']['mean']:.1f}°C")
    print(f"total precipitation: {weather_summary['precipitation_stats']['total_mm']:.1f}mm")
    print(f"rainy hours: {weather_summary['precipitation_stats']['rainy_hours']:,}")
    print(f"mean wind speed: {weather_summary['wind_stats']['mean_speed_10m']:.1f} km/h")
    
    return weather_df, weather_summary
