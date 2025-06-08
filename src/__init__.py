"""
EcoBici-AI: AI-powered bicycle sharing demand prediction system for Buenos Aires.

This package provides tools for processing EcoBici trip data and training
machine learning models to predict bicycle demand at stations.
"""

try:
    from .data_pipeline import load_csv_files, load_users, load_trips, compute_counts
    from .data_analysis import (
        load_combined_data, 
        filter_data_until_date, 
        temporal_split_data
    )
    from .time_series_rnn import EcoBiciTimeSeriesPredictor, BikeStationLSTM
    from .train_rnn import (
        create_default_config,
        load_config,
        setup_mlflow,
        run_experiment,
        load_and_prepare_data,
        create_predictor_from_config,
        train_and_evaluate_model,
        log_results_to_mlflow,
        create_visualization_plots
    )
    from .preprocess import preprocess_trips_data
except ImportError as e:
    # graceful degradation for missing optional dependencies
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)



__all__ = [
    # data processing
    "load_csv_files",
    "load_users", 
    "load_trips",
    "compute_counts",
    "preprocess_trips_data",
    
    # data analysis
    "load_combined_data",
    "filter_data_until_date",
    "temporal_split_data",
    
    # models
    "EcoBiciTimeSeriesPredictor",
    "BikeStationLSTM",
    
    # training utilities
    "create_default_config",
    "load_config",
    "setup_mlflow", 
    "run_experiment",
    "load_and_prepare_data",
    "create_predictor_from_config",
    "train_and_evaluate_model",
    "log_results_to_mlflow",
    "create_visualization_plots",
]
