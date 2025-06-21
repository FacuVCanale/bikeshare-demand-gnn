"""
Clustering module for EcoBici station clustering and feature generation.

This module provides utilities for:
- K-means clustering of bike stations based on coordinates
- Generation of internal vs external trip features
- Handling of "ghost trips" within clusters
- User demographics and bike model aggregations
"""

from .station_clustering import StationClusterer
from .cluster_features import ClusterFeatureGenerator
from .data_preparation import DataPreparator

__all__ = [
    'StationClusterer',
    'ClusterFeatureGenerator', 
    'DataPreparator'
] 