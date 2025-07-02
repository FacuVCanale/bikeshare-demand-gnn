"""
Prepare data for GNN model inference.

This script takes new trip data, combines it with historical processed data,
and generates the feature set required for the GNN model to make predictions.
It uses existing logic from FeatureEngineer to ensure consistency.
"""

import polars as pl
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Optional

# Add project root and scripts directory to path to import modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root / 'src'))

# Now we can import from our custom modules
try:
    from features.weather import WeatherDataCollector
    from feature_engineering import FeatureEngineer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that 'scripts/feature_engineering.py' and 'src/features/weather.py' exist.")
    sys.exit(1)

def prepare_inference_data(
    historical_data_path: str,
    inference_trips_path: str,
    weather_data_path: str,
    output_path: str,
    metadata_path: Optional[str] = None
):
    """
    Prepares a dataset for inference by applying feature engineering logic.

    Args:
        historical_data_path: Path to the processed training features CSV file.
        inference_trips_path: Path to the new raw trips CSV file for inference.
        weather_data_path: Path to the hourly weather data CSV file.
        output_path: Path to save the final inference-ready data.
        metadata_path: Optional path to the feature metadata JSON file.
    """
    print("--- Starting Inference Data Preparation ---")

    # 1. Load historical PROCESSED data
    print(f"Loading historical processed data from: {historical_data_path}")
    historical_df = pl.read_csv(historical_data_path, try_parse_dates=True)

    # 2. Load new RAW trip data for inference
    print(f"Loading inference trip data from: {inference_trips_path}")
    inference_trips_df = pl.read_csv(inference_trips_path, try_parse_dates=True)

    # 3. Load metadata and instantiate FeatureEngineer
    if metadata_path is None:
        metadata_path = Path(historical_data_path).parent / 'feature_metadata.json'
    
    if not Path(metadata_path).exists():
        print(f"Warning: Metadata file not found at {metadata_path}.")
        print("Using default parameters for FeatureEngineer.")
        delta_t_minutes = 30
        fill_nulls = True
        fill_strategy = 'zero'
    else:
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        delta_t_minutes = metadata['delta_t_minutes']
        fill_nulls = metadata['processing_config']['fill_nulls']
        fill_strategy = metadata['processing_config']['fill_strategy']

    fe = FeatureEngineer(
        delta_t_minutes=delta_t_minutes,
        fill_nulls=fill_nulls,
        fill_strategy=fill_strategy
    )
    print(f"FeatureEngineer initialized with delta_t={delta_t_minutes} mins, fill_strategy='{fill_strategy}'")

    # 4. Load weather data and merge with inference trips
    print("Loading and matching weather data...")
    weather_collector = WeatherDataCollector()
    
    # Determine date range for weather data from inference trips
    start_date = inference_trips_df.select(pl.col('fecha_origen_recorrido').min()).item().date()
    end_date = inference_trips_df.select(pl.col('fecha_destino_recorrido').max()).item().date()
    
    print(f"Inference trips date range: {start_date} to {end_date}")
    
    weather_df = weather_collector.get_weather_data(
        file_path=weather_data_path,
        start_date=str(start_date),
        end_date=str(end_date)
    )

    inference_trips_with_weather_df = weather_collector.match_weather_to_trips(
        trips_df=inference_trips_df.to_pandas(),
        weather_df=weather_df
    )
    inference_trips_with_weather_df = pl.from_pandas(inference_trips_with_weather_df)
    print("Weather data matched successfully.")

    # fix: cast datetime columns to nanoseconds to match expectations
    # of the feature engineering script and avoid schemaerror on join.
    inference_trips_with_weather_df = inference_trips_with_weather_df.with_columns([
        pl.col("fecha_origen_recorrido").cast(pl.Datetime("ns")),
        pl.col("fecha_destino_recorrido").cast(pl.Datetime("ns")),
    ])

    # 5. Convert inference trips to delta_t format (without historical features)
    print("Converting inference trips to delta_t format...")
    # This calls a "private" method, which is necessary given the current structure
    # of FeatureEngineer. This is a reasonable approach for a dedicated script.
    inference_delta_df = fe._convert_to_delta_t(inference_trips_with_weather_df)

    # Add features that don't depend on historical context
    inference_delta_df = fe._add_time_features(inference_delta_df)
    
    # This is a disabled method in the source file, but we call it for completeness
    inference_delta_df = fe._add_user_features(inference_delta_df)
    print(f"Inference data converted to {len(inference_delta_df)} delta_t intervals.")

    # 6. Concatenate with historical data to build context for historical features
    print("Concatenating historical data with new inference data...")
    combined_df = pl.concat([historical_df, inference_delta_df], how='diagonal')
    
    # Ensure dtypes are consistent after concat, especially for categorical/string columns
    for col in historical_df.columns:
        if col in combined_df.columns and historical_df[col].dtype != combined_df[col].dtype:
            combined_df = combined_df.with_columns(pl.col(col).cast(historical_df[col].dtype))
    
    # 7. Calculate history-dependent features on the combined dataframe
    print("Calculating historical features (lags, moving averages)...")
    combined_df_with_features = fe._add_station_features(combined_df)
    combined_df_with_features = fe._add_climate_features(combined_df_with_features)

    # 8. Filter to get only the inference rows, now with all features calculated
    inference_start_time = inference_delta_df.select(pl.col('datetime').min()).item()
    inference_data_final = combined_df_with_features.filter(
        pl.col('datetime') >= inference_start_time
    )
    print(f"Filtered to {len(inference_data_final)} final inference rows.")

    # 9. Final cleaning
    print("Cleaning final feature set...")
    inference_data_final = fe._clean_features(inference_data_final)
    
    # Ensure final columns match historical data
    final_cols = historical_df.columns
    inference_data_final = inference_data_final.select(final_cols)

    # 10. Save the result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    inference_data_final.write_csv(output_path)
    print(f" Inference data successfully created at: {output_path}")
    print("--- Preparation Complete ---")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare GNN inference data by applying feature engineering.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--historical_data_path', 
        type=str, 
        required=True,
        help='Path to the historical processed features file (e.g., train_features.csv).'
    )
    parser.add_argument(
        '--inference_trips_path', 
        type=str, 
        required=True,
        help='Path to the new raw trip data for which to create predictions.'
    )
    parser.add_argument(
        '--weather_data_path', 
        type=str, 
        default='data/raw/meteo/hourly_open_meteo.csv',
        help='Path to the hourly weather data file.'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='data/inference/inference_features.csv',
        help='Path to save the final inference-ready data.'
    )
    parser.add_argument(
        '--metadata_path', 
        type=str,
        help='Optional path to feature metadata JSON. If not provided, it is inferred from the historical data path.'
    )
    
    args = parser.parse_args()

    prepare_inference_data(
        historical_data_path=args.historical_data_path,
        inference_trips_path=args.inference_trips_path,
        weather_data_path=args.weather_data_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path
    )


if __name__ == '__main__':
    main() 