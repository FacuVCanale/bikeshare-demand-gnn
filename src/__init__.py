"""
EcoBici-AI: Machine Learning for Buenos Aires Bike Share System

This package provides tools for analyzing and predicting bike share demand
using historical EcoBici trip data with advanced feature engineering and
machine learning models.

Reorganized module structure:
- data/: Data processing and loading utilities
- features/: Feature engineering and enrichment
- analysis/: Data analysis and visualization
- utils/: General utilities (spatial, system, preprocessing, embeddings)
- models/: Machine learning models
- training/: Training pipelines
"""

__version__ = "0.1.0"
__author__ = "EcoBici-AI Team"

# Core data processing modules
try:
    from .data.data_loading import *
    from .data.data_processing import *
    from .data.temporal_processing import *
    from .data.weather_processing import WeatherDataCollector
except ImportError as e:
    print(f"Warning: Could not import data modules: {e}")

# Feature engineering modules
try:
    from .features.feature_engineering import *
    from .features.core_pipeline import engineer_ecobici_features
except ImportError as e:
    print(f"Warning: Could not import feature engineering modules: {e}")

# Analysis modules
try:
    from .analysis.data_analysis import *
    from .analysis.station_analytics import *
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

# Utility modules
try:
    from .utils.spatial_utils import *
    from .utils.system_utils import *
    from .utils.preprocessing_utils import *
    from .utils.embedding_utils import *
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")

# Model modules
try:
    from .models import *
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")

# Main pipeline function for backward compatibility
def run_feature_pipeline(df_raw, **kwargs):
    """
    Run the complete feature engineering pipeline.
    
    This is a wrapper function that maintains backward compatibility
    while using the reorganized modules.
    """
    from .features.core_pipeline import engineer_ecobici_features
    return engineer_ecobici_features(df_raw, **kwargs)


def pivot_features_for_global_model(df_long, **kwargs):
    """
    Legacy function - this functionality should be in a separate modeling module
    """
    # This function will need to be implemented in a separate module
    # For now, raise an informative error
    raise NotImplementedError(
        "pivot_features_for_global_model has been moved. "
        "Please implement this in a dedicated modeling module."
    )


# Convenience imports for common functions
__all__ = [
    # Main pipeline
    'run_feature_pipeline',
    'engineer_ecobici_features',
    
    # Data loading and processing
    'load_ecobici_data',
    'process_station_data', 
    'create_temporal_features',
    'WeatherDataCollector',
    
    # Feature engineering
    'apply_all_enrichments',
    'create_station_metadata_enhanced',
    'normalize_coordinates',
    
    # Analysis functions
    'analyze_station_activity',
    'compare_stations',
    'station_ranking_analysis',
    'analyze_raw_data',
    'analyze_weather_impact',
    
    # Utilities
    'haversine_np',
    'find_nearest_neighbors',
    'log_memory_usage',
    'calculate_safe_neighbor_limit',
    
    # Embeddings
    'create_station_embeddings_mapping',
    'prepare_features_for_neural_network'
]

print("EcoBici-AI package loaded successfully!")
print("Available modules: data, features, analysis, utils, models")
print("Use help(src) for more information")
