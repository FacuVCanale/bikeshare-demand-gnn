#!/usr/bin/env python
"""
Dataset Generation Script - Feature Engineering Only

This script only runs feature engineering to generate processed_data.parquet
without training models.

Usage:
    python generate_dataset.py \
        --data_path data/trips_with_weather.parquet \
        --output_dir results \
        --lookback_intervals 6
"""

import os
import argparse
from feature_engineering import load_and_process_data

def generate_dataset(
    data_path: str,
    output_dir: str = "results",
    station_id: int | None = None,
    delta_t_minutes: int = 30,
    fill_nulls: bool = True,
    fill_strategy: str = "zero",
    add_long_lags: bool = True,
    lookback_intervals: int = 6,
):
    """Generate processed dataset using feature engineering only."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n==============================")
    print("DATASET GENERATION")
    print("==============================")
    print(f"Input data       : {data_path}")
    print(f"Output directory : {output_dir}")
    print(f"Lookback intervals: {lookback_intervals}")
    print(f"Delta T          : {delta_t_minutes} minutes")
    print(f"Station ID       : {station_id or 'All stations'}\n")

    # Run feature engineering
    print("Running feature engineering...")
    processed_df, feature_engineer = load_and_process_data(
        data_path=data_path,
        station_id=station_id,
        delta_t_minutes=delta_t_minutes,
        fill_nulls=fill_nulls,
        fill_strategy=fill_strategy,
        add_long_lags=add_long_lags,
        lookback_intervals=lookback_intervals
    )

    # Save processed data
    output_path = os.path.join(output_dir, 'processed_data.parquet')
    processed_df.write_parquet(output_path, compression='snappy')
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Shape: {processed_df.shape}")
    print(f"🎯 Features: {len(feature_engineer.get_feature_names())}")
    
    return processed_df

def main():
    parser = argparse.ArgumentParser(description='Generate processed dataset for bike prediction.')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data file (CSV or Parquet)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--station_id', type=int, default=None,
                       help='Specific station ID (default: all stations)')
    parser.add_argument('--delta_t_minutes', type=int, default=30,
                       help='Time interval in minutes (default: 30)')
    parser.add_argument('--lookback_intervals', type=int, default=6,
                       help='Number of lookback intervals for lag features (default: 6)')
    parser.add_argument('--no_fill_nulls', action='store_true',
                       help='Do not fill null values')
    parser.add_argument('--fill_strategy', type=str, default='zero',
                       choices=['zero', 'mean', 'median', 'forward_fill'],
                       help='Strategy for filling nulls (default: zero)')
    parser.add_argument('--disable_long_lags', action='store_true',
                       help='Disable 1-week lags and moving averages')

    args = parser.parse_args()

    generate_dataset(
        data_path=args.data_path,
        output_dir=args.output_dir,
        station_id=args.station_id,
        delta_t_minutes=args.delta_t_minutes,
        fill_nulls=not args.no_fill_nulls,
        fill_strategy=args.fill_strategy,
        add_long_lags=not args.disable_long_lags,
        lookback_intervals=args.lookback_intervals,
    )

if __name__ == "__main__":
    main() 