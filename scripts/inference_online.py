import argparse
from pathlib import Path
import polars as pl
import torch
import json
import sys
import pandas as pd
from typing import Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.weather import WeatherDataCollector
from scripts.feature_engineering import FeatureEngineer
from scripts.prepare_gnn_dataset import prepare_data_for_gnn, create_station_id_mapping

def load_all_data(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Loads all required data files for the inference pipeline using Polars.
    
    This includes historical and inference trips, weather data, and the static
    graph structure (edge_index) from the GNN training data directory.
    
    Args:
        args: Command line arguments.
        
    Returns:
        A dictionary containing all loaded data components.
    """
    print("Step 1: Loading all required data using Polars...")
    try:
        print(f"  - Reading historical trips (Parquet): {args.historical_trips}")
        historical_trips_df = pl.read_parquet(args.historical_trips)
        
        print(f"  - Reading inference trips (CSV): {args.inference_trips}")
        inference_trips_df = pl.read_csv(args.inference_trips, try_parse_dates=True)
        
        print(f"  - Reading weather data (CSV): {args.weather_data}")
        weather_df = pl.read_csv(args.weather_data, try_parse_dates=True)
        
        print(f"  - Reading static graph edge_index: {args.gnn_data_dir}")
        edge_index = torch.load(Path(args.gnn_data_dir) / 'edge_index.pt', weights_only=False)

        data = {
            "historical_trips": historical_trips_df,
            "inference_trips": inference_trips_df,
            "weather": weather_df,
            "edge_index": edge_index
        }
        
        print(f"  ✓ Loaded historical trips: {len(data['historical_trips'])} rows")
        print(f"  ✓ Loaded inference trips: {len(data['inference_trips'])} rows")
        print(f"  ✓ Loaded weather data: {len(data['weather'])} rows")
        print(f"  ✓ Loaded static graph edge_index")
        return data
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        sys.exit(1)

def process_full_dataset(data: Dict[str, Any], delta_t_minutes: int) -> tuple[pl.DataFrame, Any]:
    """
    Combines historical and inference data, merges with weather, and runs
    feature engineering on the entire dataset to ensure correct context.
    
    Args:
        data: Dictionary with loaded data.
        delta_t_minutes: Time interval for feature aggregation.
        
    Returns:
        A tuple of (processed_polars_df, inference_start_time).
    """
    print("\nStep 2: Combining data, matching weather, and running feature engineering...")
    
    # Store the start time of the new data to isolate it later
    inference_start_time = data["inference_trips"]['fecha_origen_recorrido'].min()
    
    # Combine trips (Polars operation)
    combined_trips_df = pl.concat([data["historical_trips"], data["inference_trips"]])
    
    # --- PANDAS CONVERSION SCOPE ---
    # Convert to pandas ONLY for the weather matching step, as it's a pandas-based function
    print("  ... Temporarily converting to Pandas for weather matching...")
    combined_trips_pd = combined_trips_df.to_pandas()
    weather_pd = data["weather"].to_pandas()
    
    collector = WeatherDataCollector()
    full_data_with_weather_pd = collector.match_weather_to_trips(combined_trips_pd, weather_pd)
    
    full_data_with_weather = pl.from_pandas(full_data_with_weather_pd)
    print("  ... Converted back to Polars.")
    
    print("  ✓ Combined trips and merged with weather data.")

    fe = FeatureEngineer(delta_t_minutes=delta_t_minutes, fill_nulls=True, fill_strategy='zero')
    processed_polars_df = fe.process_data(full_data_with_weather)
    print("  ✓ Feature engineering complete on the full dataset.")
    
    return processed_polars_df, inference_start_time

def prepare_inference_batch(
    processed_df: pl.DataFrame,
    inference_start_time: pd.Timestamp,
    edge_index: torch.Tensor,
    target_cols: list[str]
) -> Dict[str, Any]:
    """
    Isolates featurized inference data and prepares the final GNN batch.
    
    Args:
        processed_df: The fully featurized DataFrame (historical + inference).
        inference_start_time: The timestamp marking the start of inference data.
        edge_index: The static graph structure.
        target_cols: The target columns for prediction.
        
    Returns:
        A dictionary containing the final GNN batch data and metadata.
    """
    print("\nStep 3: Isolating featurized inference data and preparing GNN batch...")
    
    # Isolate the new data rows after feature engineering
    prediction_rows_df = processed_df.filter(pl.col("datetime") >= inference_start_time)
    print(f"  ✓ Isolated {len(prediction_rows_df)} rows for prediction.")

    if prediction_rows_df.is_empty():
        print("Error: No data rows found for the inference period after processing. Check timestamps.")
        sys.exit(1)

    # Create station mapping from the full processed data to ensure consistency
    all_station_ids = processed_df.select(pl.col("station_id").unique().sort()).to_series().to_list()
    station_id_mapping = create_station_id_mapping(all_station_ids)

    # Convert only the prediction rows to tensors
    x, y, feature_cols = prepare_data_for_gnn(prediction_rows_df, station_id_mapping, target_cols)
    print("  ✓ Converted prediction data to tensors.")

    return {
        "batch": {'x': x, 'y': y, 'edge_index': edge_index},
        "station_mapping": station_id_mapping,
        "feature_cols": feature_cols
    }

def save_output(output_dir: str, batch_data: Dict[str, Any], target_cols: list[str]):
    """
    Saves the final GNN batch and metadata files.
    
    Args:
        output_dir: The directory to save the files in.
        batch_data: The dictionary containing the prepared batch and metadata.
        target_cols: The target column names.
    """
    print("\nStep 4: Saving output files...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_batch_file = output_path / 'inference_batch.pt'
    torch.save(batch_data["batch"], output_batch_file)
    
    # Save metadata for consistency
    with open(output_path / 'station_id_mapping.json', 'w') as f:
        json.dump(batch_data["station_mapping"], f, indent=2)
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump({'features': batch_data["feature_cols"], 'targets': target_cols}, f, indent=2)
        
    print(f"  ✓ Saved GNN batch to: {output_batch_file}")



def main():
    parser = argparse.ArgumentParser(
        description='Online inference with historical context for GNN.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input Data ---
    parser.add_argument('--inference_trips', type=str, required=True, 
                        help='Path to new trips data for which to make predictions (.csv).')
    parser.add_argument('--historical_trips', type=str, required=True,
                        help='Path to the full historical trips dataset used for training (.csv).')
    parser.add_argument('--weather_data', type=str, default='data/meteo/hourly_open_meteo.csv', 
                        help='Path to the historical weather data file (.csv).')
    parser.add_argument('--gnn_data_dir', type=str, default='data/gnn_ready',
                        help='Directory containing the original GNN data files (e.g., edge_index.pt).')

    # --- Output ---
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for the generated GNN batch file.')

    # --- Feature Engineering & GNN Parameters ---
    parser.add_argument('--delta_t_minutes', type=int, default=30, 
                        help='Time interval in minutes for feature aggregation.')
    parser.add_argument('--target', type=str, default='arrivals', choices=['arrivals', 'departures', 'both'],
                        help='Target variable(s) for the GNN.')
    
    args = parser.parse_args()

    print("--- Starting Online Inference with Historical Context ---")

    # 1. Load data
    loaded_data = load_all_data(args)
    
    # 2. Process data
    processed_df, inference_start_time = process_full_dataset(loaded_data, args.delta_t_minutes)

    # 3. Prepare GNN batch
    target_map = {'arrivals': ['arrivals'], 'departures': ['departures'], 'both': ['arrivals', 'departures']}
    target_cols = target_map[args.target]
    
    inference_data = prepare_inference_batch(processed_df, inference_start_time, loaded_data["edge_index"], target_cols)

    # 4. Save results
    save_output(args.output_dir, inference_data, target_cols)
        
    print(f"\n--- Inference Batch Created Successfully ---")
    x_tensor = inference_data["batch"]['x']
    print(f"  The batch contains {x_tensor.shape[0]} nodes with {x_tensor.shape[1]} features each.")
    print(f"  Graph structure is static and taken from the original training data.")

if __name__ == '__main__':
    main() 