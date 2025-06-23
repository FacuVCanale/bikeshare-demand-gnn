#!/usr/bin/env python3
"""
Script to find optimal number of clusters for station clustering.

This script demonstrates how to use the StationClusterer's find_optimal_k method
to determine the best K value using elbow method and silhouette analysis.
"""

import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from clustering import StationClusterer, DataPreparator
import logging

def plot_clusters_on_map(stations_df, clusterer, k_value, title_suffix):
    """Plot clustered stations on a map."""
    print(f"Creating map visualization for K={k_value}...")
    
    # Get cluster assignments
    stations_with_clusters = clusterer.predict_cluster(stations_df)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Get unique clusters and assign colors
    unique_clusters = sorted(stations_with_clusters['cluster_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_clusters), 20)))
    if len(unique_clusters) > 20:
        # Use a larger colormap for many clusters
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_stations = stations_with_clusters.filter(pl.col('cluster_id') == cluster_id)
        lats = cluster_stations['lat'].to_list()
        lons = cluster_stations['lon'].to_list()
        
        color = colors[i % len(colors)]
        plt.scatter(lons, lats, c=[color], alpha=0.7, s=30, 
                   label=f'Cluster {cluster_id}' if len(unique_clusters) <= 10 else None)
    
    # Plot cluster centroids
    centroids = clusterer.get_cluster_centroids()
    centroid_lats = [centroids[cid]['lat'] for cid in centroids.keys()]
    centroid_lons = [centroids[cid]['lon'] for cid in centroids.keys()]
    
    plt.scatter(centroid_lons, centroid_lats, c='red', marker='x', s=100, 
               linewidths=3, label='Centroids', alpha=0.8)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Station Clustering Map - {title_suffix} (K={k_value})\n'
              f'{len(stations_df)} stations, {len(unique_clusters)} clusters')
    plt.grid(True, alpha=0.3)
    
    # Add legend only if few clusters
    if len(unique_clusters) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print some cluster statistics
    cluster_sizes = stations_with_clusters.group_by('cluster_id').agg(pl.len().alias('size'))
    print(f"Cluster size distribution:")
    for row in cluster_sizes.sort('cluster_id').iter_rows():
        cluster_id, size = row
        print(f"  Cluster {cluster_id}: {size} stations")

def main():
    """Main function to run optimal K analysis."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting optimal K analysis for station clustering...")
    
    # Use script location as reference for robust path resolution
    script_dir = Path(__file__).parent
    
    try:
        # Initialize data preparator
        data_dir = script_dir.parent / "data"
        data_prep = DataPreparator(data_dir=str(data_dir))
        
        # Load only the required dataset
        logger.info("Loading trips_with_weather dataset...")
        trips_with_weather_path = script_dir.parent / "data" / "processed" / "trips_with_weather.parquet"
        logger.info(f"Looking for data at: {trips_with_weather_path.absolute()}")
        
        if not trips_with_weather_path.exists():
            logger.error(f"Data file not found at: {trips_with_weather_path.absolute()}")
            logger.error("Please make sure the trips_with_weather.parquet file exists in the data/processed/ directory")
            sys.exit(1)
            
        trips_with_weather = pl.read_parquet(trips_with_weather_path)
        logger.info(f"Loaded {trips_with_weather.height:,} trips")
        
        # Extract station metadata directly (avoiding potential type issues in DataPreparator)
        logger.info("Extracting station metadata directly from trips data...")
        
        # Extract from origin stations with immediate type conversion
        origin_stations = trips_with_weather.select([
            pl.col("id_estacion_origen").cast(pl.Int64, strict=False).alias("station_id"),
            pl.col("nombre_estacion_origen").alias("station_name"),
            pl.col("lat_estacion_origen").cast(pl.Float64, strict=False).alias("lat"),
            pl.col("long_estacion_origen").cast(pl.Float64, strict=False).alias("lon")
        ]).drop_nulls().unique(subset=["station_id"], keep="first")
        
        logger.info(f"Extracted {origin_stations.height} unique origin stations")

        # Extract from destination stations with immediate type conversion
        dest_stations = trips_with_weather.select([
            pl.col("id_estacion_destino").cast(pl.Int64, strict=False).alias("station_id"),
            pl.col("nombre_estacion_destino").alias("station_name"),
            pl.col("lat_estacion_destino").cast(pl.Float64, strict=False).alias("lat"),
            pl.col("long_estacion_destino").cast(pl.Float64, strict=False).alias("lon")
        ]).drop_nulls().unique(subset=["station_id"], keep="first")
        
        logger.info(f"Extracted {dest_stations.height} unique destination stations")

        # Combine and deduplicate
        stations = pl.concat([origin_stations, dest_stations]).unique(
            subset=["station_id"], keep="first"
        ).drop_nulls().sort("station_id")

        
        logger.info(f"Extracted {stations.height} unique stations")
        
        # Debug: Check data types
        logger.info("Station data types:")
        for col in stations.columns:
            logger.info(f"  {col}: {stations[col].dtype}")
        
        # Final cleanup - remove any remaining nulls
        stations = stations.filter(
            pl.col("lat").is_not_null() & 
            pl.col("lon").is_not_null() &
            pl.col("station_id").is_not_null()
        )
        
        logger.info(f"After final null removal: {stations.height} stations")
        logger.info(f"Coordinate ranges:")
        logger.info(f"  Lat: {stations['lat'].min():.6f} to {stations['lat'].max():.6f}")
        logger.info(f"  Lon: {stations['lon'].min():.6f} to {stations['lon'].max():.6f}")
        
        # Filter stations by trip count (minimum 5000 trips)
        logger.info("Filtering stations by trip count (minimum 5000 trips)...")
        
        # Count trips for each station (both as origin and destination)
        origin_counts = trips_with_weather.group_by("id_estacion_origen").agg(
            pl.len().alias("origin_trips")
        ).rename({"id_estacion_origen": "station_id"}).with_columns(
            pl.col("station_id").cast(pl.Int64, strict=False)
        )
        
        dest_counts = trips_with_weather.group_by("id_estacion_destino").agg(
            pl.len().alias("dest_trips")
        ).rename({"id_estacion_destino": "station_id"}).with_columns(
            pl.col("station_id").cast(pl.Int64, strict=False)
        )
        
        # Combine origin and destination counts
        station_trip_counts = stations.select("station_id").join(
            origin_counts, on="station_id", how="left"
        ).join(
            dest_counts, on="station_id", how="left"
        ).with_columns([
            pl.col("origin_trips").fill_null(0),
            pl.col("dest_trips").fill_null(0)
        ]).with_columns([
            (pl.col("origin_trips") + pl.col("dest_trips")).alias("total_trips")
        ])
        
        logger.info(f"Trip count statistics:")
        logger.info(f"  Min trips: {station_trip_counts['total_trips'].min()}")
        logger.info(f"  Max trips: {station_trip_counts['total_trips'].max()}")
        logger.info(f"  Mean trips: {station_trip_counts['total_trips'].mean():.1f}")
        logger.info(f"  Median trips: {station_trip_counts['total_trips'].median():.1f}")
        
        # Filter stations with at least 5000 trips
        stations_with_enough_trips = station_trip_counts.filter(
            pl.col("total_trips") >= 5000
        ).select("station_id")
        
        # Filter the stations DataFrame to keep only stations with enough trips
        stations_filtered = stations.join(stations_with_enough_trips, on="station_id", how="inner")
        
        logger.info(f"Stations before filtering: {stations.height}")
        logger.info(f"Stations after filtering (≥5000 trips): {stations_filtered.height}")
        logger.info(f"Removed {stations.height - stations_filtered.height} stations with <5000 trips")
        
        if stations_filtered.height == 0:
            logger.error("No stations left after filtering! Consider reducing the trip threshold.")
            sys.exit(1)
        
        # Use filtered stations for clustering
        stations = stations_filtered
        
        # Initialize clusterer
        clusterer = StationClusterer(random_state=42)
        
        # Run optimal K analysis
        # You can adjust the range based on your needs
        k_elbow, k_silhouette, results = clusterer.find_optimal_k(
            stations, 
            min_k=2, 
            max_k=150,  # Reduced from 250 for faster execution
            sample_size=5000,  # Sample size for silhouette calculation
            plot=True
        )
        
        # Print detailed results
        print("\n" + "="*60)
        print("OPTIMAL K ANALYSIS RESULTS")
        print("="*60)
        print(f"Optimal K (Elbow Method):     {k_elbow}")
        print(f"Optimal K (Silhouette):       {k_silhouette}")
        print(f"Maximum Silhouette Score:     {results['max_silhouette']:.4f}")
        
        if results.get('elbow_inertia') is not None:
            print(f"Elbow Method Inertia:         {results['elbow_inertia']:.2f}")
        
        print("\nRecommendation:")
        # Find the best K >= 50 and best K > 100 based on silhouette scores
        K_range = results['K_range']
        silhouette_scores = results['silhouette_scores']
        
        # Find best K >= 50
        valid_indices_50 = [i for i, k in enumerate(K_range) if k >= 50]
        if valid_indices_50:
            valid_k_scores_50 = [(K_range[i], silhouette_scores[i]) for i in valid_indices_50]
            best_k_50, best_score_50 = max(valid_k_scores_50, key=lambda x: x[1])
            print(f"Best K >= 50: K={best_k_50} (silhouette score: {best_score_50:.4f})")
        else:
            best_k_50, best_score_50 = 50, None
            print(f"No K >= 50 tested. Using default K=50")
        
        # Find best K > 100
        valid_indices_100 = [i for i, k in enumerate(K_range) if k > 100]
        if valid_indices_100:
            valid_k_scores_100 = [(K_range[i], silhouette_scores[i]) for i in valid_indices_100]
            best_k_100, best_score_100 = max(valid_k_scores_100, key=lambda x: x[1])
            print(f"Best K > 100: K={best_k_100} (silhouette score: {best_score_100:.4f})")
        else:
            best_k_100, best_score_100 = None, None
            print(f"No K > 100 tested in the range.")
        
        print(f"\nOriginal algorithm recommendations:")
        print(f"  Elbow method: K={k_elbow}")
        print(f"  Silhouette method: K={k_silhouette}")
        
        # Test and visualize both clustering scenarios
        print(f"\n" + "="*60)
        print("CLUSTERING ANALYSIS & VISUALIZATION")
        print("="*60)
        
        def test_and_plot_clustering(k_value, title_suffix, stations_data):
            """Test clustering and create visualization."""
            if k_value is None:
                print(f"Skipping {title_suffix} - K value not available")
                return None
                
            print(f"\nTesting clustering with K={k_value} ({title_suffix})...")
            test_clusterer = StationClusterer(n_clusters=k_value, random_state=42)
            test_clusterer.fit_clustering(stations_data)
            
            # Get cluster statistics
            centroids = test_clusterer.get_cluster_centroids()
            cluster_sizes = [centroids[cid]['station_count'] for cid in centroids.keys()]
            
            print(f"Cluster size statistics with K={k_value}:")
            print(f"  Min: {min(cluster_sizes)} stations")
            print(f"  Max: {max(cluster_sizes)} stations") 
            print(f"  Mean: {sum(cluster_sizes)/len(cluster_sizes):.1f} stations")
            
            # Create map visualization
            plot_clusters_on_map(stations_data, test_clusterer, k_value, title_suffix)
            
            return test_clusterer
        
        # Test both scenarios
        clusterer_50 = test_and_plot_clustering(best_k_50, f"K≥50 Optimal", stations)
        if best_k_100:
            clusterer_100 = test_and_plot_clustering(best_k_100, f"K>100 Optimal", stations)
        
        recommended_k = best_k_50  # Use the K>=50 as the main recommendation
        
        logger.info("✓ Optimal K analysis completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Make sure you have the required data files in the data/ directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 