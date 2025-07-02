import argparse
from pathlib import Path
import polars as pl
import torch
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from feature_engineering import load_and_process_data

# import GNN dataset utilities
from prepare_gnn_dataset import (
    load_station_coordinates_from_data,
    create_station_adjacency_matrix,
    create_edge_index_from_adjacency,
    create_station_id_mapping,
    prepare_data_for_gnn
)

# import weather utilities
sys.path.append(str(Path(__file__).parent.parent))
from src.features.weather import WeatherDataCollector

def trips_to_features(trips_path, weather_path=None, delta_t_minutes=30, fill_nulls=True, fill_strategy='zero'):
    """
    Load trips and generate delta T features using feature engineering pipeline.
    If weather_path is provided, merge weather data before feature engineering.
    """
    print(f"Loading trips from {trips_path}")
    if weather_path is not None:
        print(f"Loading weather from {weather_path}")
        trips_df = pd.read_csv(trips_path, parse_dates=True)
        weather_df = pd.read_csv(weather_path, parse_dates=['date'])
        collector = WeatherDataCollector()
        trips_df = collector.match_weather_to_trips(trips_df, weather_df)
        # convert to polars for feature engineering
        features_df, _ = load_and_process_data(
            data_path=None,  
            delta_t_minutes=delta_t_minutes,
            fill_nulls=fill_nulls,
            fill_strategy=fill_strategy,
            create_splits=False,
            df=pl.from_pandas(trips_df)
        )
    else:
        features_df, _ = load_and_process_data(
            data_path=trips_path,
            delta_t_minutes=delta_t_minutes,
            fill_nulls=fill_nulls,
            fill_strategy=fill_strategy,
            create_splits=False
        )
    return features_df

def features_to_gnn(features_df, target_cols, k_neighbors=5, distance_threshold=5.0):
    """
    Prepare GNN-ready tensors and graph structure from features DataFrame.
    """
    station_coordinates = load_station_coordinates_from_data(features_df)
    unique_station_ids = sorted(list(station_coordinates.keys()))
    station_id_mapping = create_station_id_mapping(unique_station_ids)
    adj_matrix = create_station_adjacency_matrix(
        station_coordinates,
        k_neighbors=k_neighbors,
        distance_threshold=distance_threshold
    )
    edge_index = create_edge_index_from_adjacency(adj_matrix)
    x, y, feature_cols = prepare_data_for_gnn(features_df, station_id_mapping, target_cols)
    return x, y, edge_index, station_id_mapping, feature_cols

def main():
    parser = argparse.ArgumentParser(description='Online inference: trips to GNN-ready batch')
    parser.add_argument('--input_trips', type=str, required=True, help='Path to trips.parquet or .csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for GNN batch')
    parser.add_argument('--weather_data', type=str, default='data/meteo/hourly_open_meteo.csv', help='Path to weather CSV file')
    parser.add_argument('--target', type=str, default='arrivals', choices=['arrivals', 'departures', 'both'])
    parser.add_argument('--delta_t_minutes', type=int, default=30)
    parser.add_argument('--k_neighbors', type=int, default=5)
    parser.add_argument('--distance_threshold', type=float, default=5.0)
    parser.add_argument('--fill_nulls', action='store_true', default=True)
    parser.add_argument('--fill_strategy', type=str, default='zero', choices=['zero', 'mean', 'median', 'forward_fill'])
    args = parser.parse_args()

    target_map = {
        'arrivals': ['arrivals'],
        'departures': ['departures'],
        'both': ['arrivals', 'departures']
    }
    target_cols = target_map[args.target]

    features_df = trips_to_features(
        trips_path=args.input_trips,
        weather_path=args.weather_data,
        delta_t_minutes=args.delta_t_minutes,
        fill_nulls=args.fill_nulls,
        fill_strategy=args.fill_strategy
    )

    x, y, edge_index, station_id_mapping, feature_cols = features_to_gnn(
        features_df, target_cols,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'x': x, 'y': y, 'edge_index': edge_index}, output_dir / 'inference_batch.pt')
    with open(output_dir / 'station_id_mapping.json', 'w') as f:
        import json
        json.dump(station_id_mapping, f, indent=2)
    with open(output_dir / 'feature_names.json', 'w') as f:
        import json
        json.dump(feature_cols, f, indent=2)
    print(f"Saved GNN batch to {output_dir}/inference_batch.pt")

if __name__ == '__main__':
    main() 