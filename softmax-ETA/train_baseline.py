"""
Baseline model for ETA and destination prediction using historical statistics.
This serves as the simplest possible baseline using historical distributions.
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import our evaluation utilities
from evaluation import (
    evaluate_eta_prediction, 
    evaluate_destination_prediction,
    print_evaluation_results,
    save_evaluation_results,
    load_and_prepare_data
)


class HistoricalBaseline:
    """
    Simple baseline model using historical statistics.
    
    For ETA: Uses historical mean duration for same route, fallback to overall mean
    For Destination: Uses historical frequency of destinations from origin stations
    """
    
    def __init__(self):
        self.route_durations = {}  # (origin, destination) -> list of durations
        self.station_destinations = {}  # origin -> Counter of destinations
        self.overall_mean_duration = None
        self.overall_dest_frequencies = None
        self.fitted = False
        
    def fit(self, train_df: pl.DataFrame):
        """
        Fit the baseline model on training data.
        
        Args:
            train_df: Training dataset
        """
        print("🏗️  Fitting Historical Baseline Model...")
        
        # Convert to pandas for easier groupby operations
        df_pd = train_df.to_pandas()
        
        # Remove any invalid data
        df_pd = df_pd.dropna(subset=['duracion_recorrido', 'id_estacion_origen', 'id_estacion_destino'])
        
        print(f"   Training on {len(df_pd):,} valid trips")
        
        # 1. Fit ETA prediction model (route-based historical means)
        print("   📊 Learning route duration patterns...")
        self.route_durations = {}
        
        for _, row in df_pd.iterrows():
            route = (row['id_estacion_origen'], row['id_estacion_destino'])
            duration = row['duracion_recorrido']
            
            if route not in self.route_durations:
                self.route_durations[route] = []
            self.route_durations[route].append(duration)
        
        # Calculate means for each route
        self.route_means = {
            route: np.mean(durations) 
            for route, durations in self.route_durations.items()
        }
        
        # Overall fallback mean
        self.overall_mean_duration = df_pd['duracion_recorrido'].mean()
        
        print(f"   📈 Learned patterns for {len(self.route_means)} unique routes")
        print(f"   📈 Overall mean duration: {self.overall_mean_duration:.1f} seconds")
        
        # 2. Fit destination prediction model (station-based frequencies)
        print("   🎯 Learning destination patterns...")
        self.station_destinations = {}
        
        for _, row in df_pd.iterrows():
            origin = row['id_estacion_origen']
            destination = row['id_estacion_destino']
            
            if origin not in self.station_destinations:
                self.station_destinations[origin] = Counter()
            self.station_destinations[origin][destination] += 1
        
        # Convert to probabilities
        self.station_dest_probs = {}
        for origin, dest_counter in self.station_destinations.items():
            total_trips = sum(dest_counter.values())
            self.station_dest_probs[origin] = {
                dest: count / total_trips 
                for dest, count in dest_counter.items()
            }
        
        # Overall destination frequencies as fallback
        all_destinations = Counter(df_pd['id_estacion_destino'])
        total_all_trips = sum(all_destinations.values())
        self.overall_dest_frequencies = {
            dest: count / total_all_trips 
            for dest, count in all_destinations.items()
        }
        
        print(f"   🗺️  Learned patterns for {len(self.station_dest_probs)} origin stations")
        print(f"   🗺️  Total unique destinations: {len(self.overall_dest_frequencies)}")
        
        self.fitted = True
        print("   ✅ Model fitting complete!")
        
    def predict_eta(self, test_df: pl.DataFrame) -> np.ndarray:
        """
        Predict ETA using historical route means.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Array of predicted durations
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        df_pd = test_df.to_pandas()
        predictions = []
        
        fallback_count = 0
        for _, row in df_pd.iterrows():
            route = (row['id_estacion_origen'], row['id_estacion_destino'])
            
            if route in self.route_means:
                pred_duration = self.route_means[route]
            else:
                # Fallback to overall mean
                pred_duration = self.overall_mean_duration
                fallback_count += 1
            
            predictions.append(pred_duration)
        
        print(f"   📊 ETA predictions: {len(predictions) - fallback_count:,} route-specific, {fallback_count:,} fallback")
        return np.array(predictions)
    
    def predict_destination(self, test_df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict destinations using historical station frequencies.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Tuple of (predicted_destinations, prediction_probabilities)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        df_pd = test_df.to_pandas()
        predictions = []
        probabilities = []
        
        # Get all unique destinations for consistent probability matrix
        all_destinations = sorted(self.overall_dest_frequencies.keys())
        dest_to_idx = {dest: idx for idx, dest in enumerate(all_destinations)}
        
        fallback_count = 0
        for _, row in df_pd.iterrows():
            origin = row['id_estacion_origen']
            
            if origin in self.station_dest_probs:
                # Use station-specific probabilities
                station_probs = self.station_dest_probs[origin]
                # Get most frequent destination for this station
                pred_dest = max(station_probs.keys(), key=station_probs.get)
                
                # Build probability vector
                prob_vector = np.zeros(len(all_destinations))
                for dest, prob in station_probs.items():
                    if dest in dest_to_idx:
                        prob_vector[dest_to_idx[dest]] = prob
                        
            else:
                # Fallback to overall most frequent destinations
                pred_dest = max(self.overall_dest_frequencies.keys(), 
                              key=self.overall_dest_frequencies.get)
                
                # Use overall probabilities
                prob_vector = np.zeros(len(all_destinations))
                for dest, prob in self.overall_dest_frequencies.items():
                    if dest in dest_to_idx:
                        prob_vector[dest_to_idx[dest]] = prob
                        
                fallback_count += 1
            
            predictions.append(pred_dest)
            probabilities.append(prob_vector)
        
        print(f"   🎯 Destination predictions: {len(predictions) - fallback_count:,} station-specific, {fallback_count:,} fallback")
        
        return np.array(predictions), np.array(probabilities)


def main():
    """Main training and evaluation pipeline."""
    
    # Define data paths
    data_dir = Path("../data/processed")
    train_path = data_dir / "trips_train.parquet"
    val_path = data_dir / "trips_val.parquet"
    test_path = data_dir / "trips_test.parquet"
    
    # Check if data files exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"❌ Data file not found: {path}")
            print("   Please run feat-eng.py first to generate the train/val/test splits")
            return
    
    print("🚀 HISTORICAL BASELINE MODEL TRAINING")
    print("="*60)
    
    # Load data
    train_df, val_df, test_df = load_and_prepare_data(
        str(train_path), str(val_path), str(test_path)
    )
    
    # Initialize and fit model
    model = HistoricalBaseline()
    model.fit(train_df)
    
    # Evaluate on validation set
    print(f"\n🔍 EVALUATING ON VALIDATION SET")
    print("-" * 40)
    
    # ETA predictions
    val_eta_pred = model.predict_eta(val_df)
    val_eta_true = val_df.select("duracion_recorrido").to_numpy().flatten()
    
    # Destination predictions
    val_dest_pred, val_dest_proba = model.predict_destination(val_df)
    val_dest_true = val_df.select("id_estacion_destino").to_numpy().flatten()
    
    # Calculate metrics
    eta_metrics = evaluate_eta_prediction(val_eta_true, val_eta_pred)
    dest_metrics = evaluate_destination_prediction(
        val_dest_true, val_dest_pred, val_dest_proba
    )
    
    # Print results
    print_evaluation_results(eta_metrics, dest_metrics, "Historical Baseline (Validation)")
    
    # Evaluate on test set
    print(f"\n🔍 EVALUATING ON TEST SET")
    print("-" * 40)
    
    # ETA predictions
    test_eta_pred = model.predict_eta(test_df)
    test_eta_true = test_df.select("duracion_recorrido").to_numpy().flatten()
    
    # Destination predictions
    test_dest_pred, test_dest_proba = model.predict_destination(test_df)
    test_dest_true = test_df.select("id_estacion_destino").to_numpy().flatten()
    
    # Calculate metrics
    test_eta_metrics = evaluate_eta_prediction(test_eta_true, test_eta_pred)
    test_dest_metrics = evaluate_destination_prediction(
        test_dest_true, test_dest_pred, test_dest_proba
    )
    
    # Print results
    print_evaluation_results(test_eta_metrics, test_dest_metrics, "Historical Baseline (Test)")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    save_evaluation_results(
        test_eta_metrics, 
        test_dest_metrics, 
        "Historical_Baseline",
        str(results_dir / "baseline_results.csv")
    )
    
    # Save predictions for analysis
    results_df = test_df.with_columns([
        pl.Series("eta_pred", test_eta_pred),
        pl.Series("dest_pred", test_dest_pred)
    ])
    
    predictions_path = results_dir / "baseline_predictions.parquet"
    results_df.write_parquet(predictions_path)
    print(f"💾 Predictions saved to: {predictions_path}")
    
    print(f"\n✅ Baseline training and evaluation complete!")
    
    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"   Model Type:           Historical Statistics Baseline")
    print(f"   Training Data:        {len(train_df):,} trips")
    print(f"   Validation Data:      {len(val_df):,} trips") 
    print(f"   Test Data:            {len(test_df):,} trips")
    print(f"   Test ETA R²:          {test_eta_metrics['r2']:.4f}")
    print(f"   Test Destination Acc: {test_dest_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main() 