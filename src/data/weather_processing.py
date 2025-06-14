"""
Weather data processing functions for EcoBici-AI.

This module provides functionality to:
- Collect historical weather data from Open-Meteo API
- Match weather conditions with trip data
- Create weather features for modeling
- Analyze weather impact on bike usage
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
                    return weather_df[
                        (weather_df['date'] >= expected_start) & 
                        (weather_df['date'] <= expected_end)
                    ].copy()
                else:
                    print(f"existing data doesn't cover full period. Fetching from API...")
                    
            except Exception as e:
                print(f"error loading existing file: {e}. Fetching from API...")
        
        # fetch from API if no valid cached data
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
        Match weather conditions to trip records by finding closest weather data.
        
        Args:
            trips_df: DataFrame with trip data
            weather_df: DataFrame with weather data  
            date_column: Column name containing trip timestamps
            
        Returns:
            DataFrame with trip data and matched weather conditions
        """
        print("matching weather data to trips...")
        print(f"trips: {len(trips_df):,}, weather records: {len(weather_df):,}")
        
        # ensure date columns are datetime
        if not pd.api.types.is_datetime64_any_dtype(trips_df[date_column]):
            trips_df[date_column] = pd.to_datetime(trips_df[date_column])
        
        if not pd.api.types.is_datetime64_any_dtype(weather_df['date']):
            weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # create hourly weather lookup
        weather_hourly = weather_df.set_index('date')
        
        # function to get weather for each trip hour
        def get_weather_for_hour(hour):
            try:
                # find the closest weather record (within the same hour)
                hour_rounded = hour.round('H')
                if hour_rounded in weather_hourly.index:
                    return weather_hourly.loc[hour_rounded]
                else:
                    # find nearest available weather record
                    nearest_idx = weather_hourly.index.get_loc(hour_rounded, method='nearest')
                    return weather_hourly.iloc[nearest_idx]
            except:
                # return nulls if no match found
                return pd.Series([np.nan] * len(weather_hourly.columns), index=weather_hourly.columns)
        
        # match weather to trips
        print("matching weather conditions...")
        weather_matched = trips_df[date_column].apply(get_weather_for_hour)
        
        # combine trip data with weather
        trips_with_weather = pd.concat([trips_df.reset_index(drop=True), weather_matched.reset_index(drop=True)], axis=1)
        
        # add weather matching quality indicator
        trips_with_weather['weather_matched'] = ~trips_with_weather['temperature_2m'].isna()
        
        matched_count = trips_with_weather['weather_matched'].sum()
        print(f"successfully matched weather for {matched_count:,}/{len(trips_df):,} trips ({matched_count/len(trips_df)*100:.1f}%)")
        
        return trips_with_weather
    
    def get_weather_summary(self, weather_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for weather data.
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            Dictionary with weather statistics
        """
        numeric_columns = weather_df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'period': {
                'start': weather_df['date'].min(),
                'end': weather_df['date'].max(),
                'duration_days': (weather_df['date'].max() - weather_df['date'].min()).days,
                'total_records': len(weather_df)
            },
            'temperature': {
                'mean': weather_df['temperature_2m'].mean(),
                'min': weather_df['temperature_2m'].min(),
                'max': weather_df['temperature_2m'].max(),
                'std': weather_df['temperature_2m'].std()
            },
            'precipitation': {
                'total': weather_df['precipitation'].sum(),
                'mean': weather_df['precipitation'].mean(),
                'max': weather_df['precipitation'].max(),
                'rainy_hours': (weather_df['precipitation'] > 0).sum(),
                'rainy_percentage': (weather_df['precipitation'] > 0).mean() * 100
            },
            'wind': {
                'mean_speed': weather_df['wind_speed_10m'].mean(),
                'max_speed': weather_df['wind_speed_10m'].max(),
                'max_gusts': weather_df['wind_gusts_10m'].max()
            }
        }
        
        return summary
    
    def create_weather_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional weather features for modeling.
        
        Args:
            weather_df: DataFrame with basic weather data
            
        Returns:
            DataFrame with additional weather features
        """
        df = weather_df.copy()
        
        # temperature features
        df['temp_feel_diff'] = df['apparent_temperature'] - df['temperature_2m']
        df['temp_comfortable'] = ((df['temperature_2m'] >= 15) & (df['temperature_2m'] <= 25)).astype(int)
        df['temp_very_hot'] = (df['temperature_2m'] > 30).astype(int)
        df['temp_cold'] = (df['temperature_2m'] < 10).astype(int)
        
        # precipitation features
        df['is_raining'] = (df['precipitation'] > 0).astype(int)
        df['heavy_rain'] = (df['precipitation'] > 5).astype(int)
        df['light_rain'] = ((df['precipitation'] > 0) & (df['precipitation'] <= 1)).astype(int)
        
        # wind features
        df['strong_wind'] = (df['wind_speed_10m'] > 20).astype(int)
        df['moderate_wind'] = ((df['wind_speed_10m'] > 10) & (df['wind_speed_10m'] <= 20)).astype(int)
        
        # weather comfort index (combination of factors)
        df['weather_comfort'] = (
            df['temp_comfortable'] * 0.4 +
            (1 - df['is_raining']) * 0.3 +
            (1 - df['strong_wind']) * 0.2 +
            (df['cloud_cover'] < 70) * 0.1
        )
        
        # seasonal adjustments
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df['is_summer'] = df['month'].isin([12, 1, 2]).astype(int)
            df['is_winter'] = df['month'].isin([6, 7, 8]).astype(int)
            df['hour'] = df['date'].dt.hour
            df['is_daylight'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
        
        print(f"created {len(df.columns) - len(weather_df.columns)} additional weather features")
        
        return df


# Note: Weather analysis functions (analyze_weather_impact, print_weather_analysis) 
# have been moved to src/analysis/data_analysis.py for better organization


def load_and_summarize_weather_data(weather_collector: WeatherDataCollector, weather_file_path: str = "../../data/raw/meteo/hourly_open_meteo.csv"):
    """
    Load weather data and display summary information.
    
    Args:
        weather_collector: Instance of WeatherDataCollector
        weather_file_path: Path to weather data file
        
    Returns:
        DataFrame with weather data
    """
    # load weather data
    weather_df = weather_collector.get_weather_data(weather_file_path)
    
    # create enhanced features
    weather_df = weather_collector.create_weather_features(weather_df)
    
    # get summary
    summary = weather_collector.get_weather_summary(weather_df)
    
    # print summary
    print("🌤️  WEATHER DATA SUMMARY")
    print("=" * 50)
    print(f"📅 Period: {summary['period']['start'].strftime('%Y-%m-%d')} to {summary['period']['end'].strftime('%Y-%m-%d')}")
    print(f"   Duration: {summary['period']['duration_days']} days")
    print(f"   Records: {summary['period']['total_records']:,}")
    
    print(f"\n🌡️  Temperature:")
    print(f"   Mean: {summary['temperature']['mean']:.1f}°C")
    print(f"   Range: {summary['temperature']['min']:.1f}°C to {summary['temperature']['max']:.1f}°C")
    print(f"   Std: {summary['temperature']['std']:.1f}°C")
    
    print(f"\n🌧️  Precipitation:")
    print(f"   Total: {summary['precipitation']['total']:.1f}mm")
    print(f"   Rainy hours: {summary['precipitation']['rainy_hours']:,} ({summary['precipitation']['rainy_percentage']:.1f}%)")
    print(f"   Max hourly: {summary['precipitation']['max']:.1f}mm")
    
    print(f"\n💨 Wind:")
    print(f"   Mean speed: {summary['wind']['mean_speed']:.1f}km/h")
    print(f"   Max speed: {summary['wind']['max_speed']:.1f}km/h")
    print(f"   Max gusts: {summary['wind']['max_gusts']:.1f}km/h")
    
    return weather_df 