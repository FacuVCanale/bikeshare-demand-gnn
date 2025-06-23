"""
Station clustering utilities using KMeans algorithm.

This module handles:
- KMeans clustering of bike stations based on coordinates
- Cluster centroid calculation
- Station-to-cluster mapping
- Clustering quality metrics
"""

import polars as pl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from kneed import KneeLocator


class StationClusterer:
    """Handles KMeans clustering of bike stations based on geographical coordinates."""
    
    def __init__(self, n_clusters: int = 93, random_state: int = 42, 
                 init: str = "k-means++", log_level: str = "INFO"):
        """Initialize station clusterer.
        
        Args:
            n_clusters: Number of clusters (default: 93)
            random_state: Random state for reproducibility
            init: Initialization method for KMeans
            log_level: Logging level
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init
        self.kmeans = None
        self.scaler = None
        self.station_to_cluster = None
        self.cluster_centroids = None
        
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def fit_clustering(self, stations: pl.DataFrame) -> 'StationClusterer':
        """Fit KMeans clustering on station coordinates.
        
        Args:
            stations: DataFrame with columns ['station_id', 'lat', 'lon']
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting KMeans clustering with {self.n_clusters} clusters...")
        
        # validate input
        required_cols = ['station_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in stations.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # extract coordinates for clustering
        coords = stations.select(['lat', 'lon']).to_numpy()
        self.logger.info(f"  -> Clustering {coords.shape[0]} stations")
        self.logger.info(f"  -> Coordinate ranges:")
        self.logger.info(f"     Lat: {coords[:, 0].min():.6f} to {coords[:, 0].max():.6f}")
        self.logger.info(f"     Lon: {coords[:, 1].min():.6f} to {coords[:, 1].max():.6f}")
        
        # standardize coordinates for better clustering
        self.scaler = StandardScaler()
        coords_scaled = self.scaler.fit_transform(coords)
        
        self.logger.info(f"  -> Standardized coordinates:")
        self.logger.info(f"     Lat: mean={coords_scaled[:, 0].mean():.6f}, std={coords_scaled[:, 0].std():.6f}")
        self.logger.info(f"     Lon: mean={coords_scaled[:, 1].mean():.6f}, std={coords_scaled[:, 1].std():.6f}")
        
        # fit kmeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.kmeans.fit_predict(coords_scaled)
        
        # calculate quality metrics
        inertia = self.kmeans.inertia_
        self.logger.info(f"  -> KMeans inertia: {inertia:.2f}")
        
        # create station to cluster mapping
        station_ids = stations.select('station_id').to_series().to_list()
        self.station_to_cluster = dict(zip(station_ids, cluster_labels))
        
        # calculate cluster centroids in original coordinates
        self.cluster_centroids = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.any():
                cluster_coords = coords[cluster_mask]
                centroid_lat = cluster_coords[:, 0].mean()
                centroid_lon = cluster_coords[:, 1].mean()
                station_count = cluster_mask.sum()
                
                self.cluster_centroids[cluster_id] = {
                    'lat': centroid_lat,
                    'lon': centroid_lon,
                    'station_count': station_count
                }
            else:
                self.logger.warning(f"Cluster {cluster_id} has no stations assigned")
                
        # log cluster distribution
        cluster_sizes = [self.cluster_centroids[cid]['station_count'] 
                        for cid in self.cluster_centroids.keys()]
        
        self.logger.info(f"  -> Cluster size statistics:")
        self.logger.info(f"     Min: {min(cluster_sizes)} stations")
        self.logger.info(f"     Max: {max(cluster_sizes)} stations")
        self.logger.info(f"     Mean: {np.mean(cluster_sizes):.1f} stations")
        self.logger.info(f"     Std: {np.std(cluster_sizes):.1f} stations")
        
        # show distribution of cluster sizes
        size_bins = [0, 1, 2, 5, 10, 20, float('inf')]
        size_labels = ['0', '1', '2-4', '5-9', '10-19', '20+']
        size_counts = [0] * (len(size_bins) - 1)
        
        for size in cluster_sizes:
            for i, (lower, upper) in enumerate(zip(size_bins[:-1], size_bins[1:])):
                if lower <= size < upper:
                    size_counts[i] += 1
                    break
                    
        self.logger.info(f"  -> Cluster size distribution:")
        for label, count in zip(size_labels, size_counts):
            if count > 0:
                self.logger.info(f"     {label} stations: {count} clusters")
                
        self.logger.info("✓ KMeans clustering completed successfully")
        return self
        
    def find_optimal_k(self, stations: pl.DataFrame, min_k: int = 2, max_k: int = 250, 
                      sample_size: int = 10000, plot: bool = True) -> Tuple[int, int, Dict]:
        """Find optimal number of clusters using Elbow method and Silhouette analysis.
        
        Args:
            stations: DataFrame with columns ['station_id', 'lat', 'lon']
            min_k: Minimum number of clusters to test
            max_k: Maximum number of clusters to test
            sample_size: Sample size for silhouette score calculation (for performance)
            plot: Whether to display the analysis plots
            
        Returns:
            Tuple of (k_elbow, k_silhouette, analysis_results)
        """
        self.logger.info(f"Finding optimal K for clustering (range: {min_k}-{max_k})...")
        
        # validate input
        required_cols = ['station_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in stations.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # extract and prepare coordinates
        coords = stations.select(['lat', 'lon']).to_numpy()
        self.logger.info(f"  -> Analyzing {coords.shape[0]} stations")
        
        # standardize coordinates
        scaler = StandardScaler()
        X = scaler.fit_transform(coords)
        
        # prepare K range
        K_range = range(min_k, max_k + 1)
        inertias = []
        sil_scores = []
        
        self.logger.info(f"  -> Testing {len(K_range)} different K values...")
        
        # perform K analysis
        for i, k in enumerate(K_range):
            if i % 25 == 0:  # progress logging
                self.logger.info(f"     Progress: {i+1}/{len(K_range)} (K={k})")
            
            # fit KMeans
            km = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)
            labels = km.fit_predict(X)
            
            # calculate metrics
            inertias.append(km.inertia_)
            
            # calculate silhouette score (with sampling for performance)
            if len(X) > sample_size:
                sil_score = silhouette_score(X, labels, sample_size=sample_size, random_state=self.random_state)
            else:
                sil_score = silhouette_score(X, labels)
            sil_scores.append(sil_score)
        
        # find optimal K values
        # 1. Elbow method
        kneedle = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
        k_elbow = kneedle.knee or min_k  # fallback if no clear elbow
        
        # 2. Silhouette method
        k_sil = int(np.argmax(sil_scores) + min_k)
        
        self.logger.info(f"  -> Optimal K (Elbow method): {k_elbow}")
        self.logger.info(f"  -> Optimal K (Silhouette): {k_sil}")
        self.logger.info(f"  -> Max Silhouette score: {max(sil_scores):.4f}")
        
        # prepare results
        analysis_results = {
            'K_range': list(K_range),
            'inertias': inertias,
            'silhouette_scores': sil_scores,
            'k_elbow': k_elbow,
            'k_silhouette': k_sil,
            'max_silhouette': max(sil_scores),
            'elbow_inertia': inertias[k_elbow - min_k] if k_elbow else None
        }
        
        if plot:
            self._plot_k_analysis(K_range, inertias, sil_scores, k_elbow, k_sil)
        
        self.logger.info("✓ Optimal K analysis completed successfully")
        return k_elbow, k_sil, analysis_results
    
    def _plot_k_analysis(self, K_range: range, inertias: List[float], 
                        sil_scores: List[float], k_elbow: int, k_sil: int):
        """Create visualization plots for K analysis.
        
        Args:
            K_range: Range of K values tested
            inertias: List of inertia values
            sil_scores: List of silhouette scores
            k_elbow: Optimal K from elbow method
            k_sil: Optimal K from silhouette method
        """
        # normalize metrics for combined plot
        inertias_arr = np.array(inertias)
        sil_scores_arr = np.array(sil_scores)
        inertia_norm = (inertias_arr - np.min(inertias_arr)) / (np.max(inertias_arr) - np.min(inertias_arr))
        silhouette_norm = (sil_scores_arr - np.min(sil_scores_arr)) / (np.max(sil_scores_arr) - np.min(sil_scores_arr))
        
        # create plots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Optimal K Analysis for Station Clustering", fontsize=16, fontweight="bold")
        
        # 1. Elbow method plot
        ax[0].plot(K_range, inertias, "o-", color="royalblue", markersize=4)
        ax[0].axvline(k_elbow, color="red", ls="--", label=f"Optimal K (Elbow): {k_elbow}")
        ax[0].set_title("Elbow Method", fontweight="bold")
        ax[0].set_xlabel("Number of Clusters (K)")
        ax[0].set_ylabel("Within-cluster Sum of Squares (Inertia)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # 2. Silhouette score plot
        ax[1].plot(K_range, sil_scores, "o", color="forestgreen", markersize=4)
        ax[1].axvline(k_sil, color="red", ls="--", label=f"Optimal K (Silhouette): {k_sil}")
        ax[1].set_title("Silhouette Score", fontweight="bold")
        ax[1].set_xlabel("Number of Clusters (K)")
        ax[1].set_ylabel("Average Silhouette Score")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        # 3. Combined analysis
        ax3 = ax[2]
        ax3.plot(K_range, inertia_norm, label="Normalized Inertia", color="blue")
        ax3.set_ylabel("Normalized Inertia", color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")
        
        ax4 = ax3.twinx()
        ax4.plot(K_range, silhouette_norm, label="Normalized Silhouette", color="green")
        ax4.set_ylabel("Normalized Silhouette Score", color="green")
        ax4.tick_params(axis="y", labelcolor="green")
        
        ax3.set_title("Combined Analysis", fontweight="bold")
        ax3.set_xlabel("Number of Clusters (K)")
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    def predict_cluster(self, stations: pl.DataFrame) -> pl.DataFrame:
        """Predict cluster assignments for stations.
        
        Args:
            stations: DataFrame with station coordinates
            
        Returns:
            DataFrame with added cluster information
        """
        if self.kmeans is None or self.scaler is None:
            raise ValueError("Clusterer must be fitted before prediction")
            
        self.logger.info("Predicting cluster assignments...")
        
        # extract and scale coordinates
        coords = stations.select(['lat', 'lon']).to_numpy()
        coords_scaled = self.scaler.transform(coords)
        
        # predict clusters
        cluster_labels = self.kmeans.predict(coords_scaled)
        
        # add cluster information to dataframe
        result = stations.with_columns([
            pl.Series("cluster_id", cluster_labels),
        ])
        
        # add centroid information
        centroid_data = []
        for _, row in enumerate(cluster_labels):
            centroid_info = self.cluster_centroids.get(row, {'lat': None, 'lon': None, 'station_count': 0})
            centroid_data.append([
                centroid_info['lat'],
                centroid_info['lon'], 
                centroid_info['station_count']
            ])
            
        centroid_df = pl.DataFrame(
            centroid_data,
            schema=['cluster_centroid_lat', 'cluster_centroid_lon', 'cluster_station_count'],
            orient="row"
        )
        
        result = pl.concat([result, centroid_df], how="horizontal")
        
        self.logger.info(f"  -> Predicted clusters for {result.height} stations")
        return result
        
    def get_station_cluster_mapping(self) -> Dict[int, int]:
        """Get station to cluster mapping dictionary.
        
        Returns:
            Dictionary mapping station_id to cluster_id
        """
        if self.station_to_cluster is None:
            raise ValueError("Clusterer must be fitted before getting mapping")
        return self.station_to_cluster.copy()
        
    def get_cluster_centroids(self) -> Dict[int, Dict[str, float]]:
        """Get cluster centroid information.
        
        Returns:
            Dictionary with cluster centroid coordinates and station counts
        """
        if self.cluster_centroids is None:
            raise ValueError("Clusterer must be fitted before getting centroids")
        return self.cluster_centroids.copy()
        
    def save_model(self, filepath: str):
        """Save fitted clustering model.
        
        Args:
            filepath: Path to save the model
        """
        if self.kmeans is None:
            raise ValueError("No fitted model to save")
            
        self.logger.info(f"Saving clustering model to {filepath}")
        
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'station_to_cluster': self.station_to_cluster,
            'cluster_centroids': self.cluster_centroids,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'init': self.init
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info("✓ Model saved successfully")
        
    def load_model(self, filepath: str):
        """Load fitted clustering model.
        
        Args:
            filepath: Path to load the model from
        """
        self.logger.info(f"Loading clustering model from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.station_to_cluster = model_data['station_to_cluster']
        self.cluster_centroids = model_data['cluster_centroids']
        self.n_clusters = model_data['n_clusters']
        self.random_state = model_data['random_state']
        self.init = model_data['init']
        
        self.logger.info("✓ Model loaded successfully")
        self.logger.info(f"  -> {self.n_clusters} clusters")
        self.logger.info(f"  -> {len(self.station_to_cluster)} stations mapped")
        
    def compute_cluster_distances(self, stations_with_clusters: pl.DataFrame) -> pl.DataFrame:
        """Compute distance from each station to its cluster centroid.
        
        Args:
            stations_with_clusters: DataFrame with station coordinates and cluster assignments
            
        Returns:
            DataFrame with added distance_to_centroid column
        """
        self.logger.info("Computing distances to cluster centroids...")
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate haversine distance between two points in kilometers."""
            r = 6371  # earth radius in km
            
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            
            return r * c
            
        # calculate distances
        distances = []
        for row in stations_with_clusters.iter_rows():
            station_lat = row[stations_with_clusters.columns.index('lat')]
            station_lon = row[stations_with_clusters.columns.index('lon')]
            centroid_lat = row[stations_with_clusters.columns.index('cluster_centroid_lat')]
            centroid_lon = row[stations_with_clusters.columns.index('cluster_centroid_lon')]
            
            if None not in [station_lat, station_lon, centroid_lat, centroid_lon]:
                dist = haversine_distance(station_lat, station_lon, centroid_lat, centroid_lon)
                distances.append(dist)
            else:
                distances.append(None)
                
        result = stations_with_clusters.with_columns([
            pl.Series("distance_to_centroid_km", distances)
        ])
        
        # log distance statistics
        valid_distances = [d for d in distances if d is not None]
        if valid_distances:
            self.logger.info(f"  -> Distance statistics (km):")
            self.logger.info(f"     Min: {min(valid_distances):.3f}")
            self.logger.info(f"     Max: {max(valid_distances):.3f}")
            self.logger.info(f"     Mean: {np.mean(valid_distances):.3f}")
            self.logger.info(f"     Std: {np.std(valid_distances):.3f}")
            
        return result 