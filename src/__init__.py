"""
EcoBici-AI: AI-powered bicycle sharing demand prediction system for Buenos Aires.

This package provides tools for processing EcoBici trip data and training
machine learning models to predict bicycle demand at stations.
"""


__all__ = [
    # data processing
    "load_csv_files",
    "load_users", 
    "load_trips",
    "preprocess_trips_data",
    "preprocess_whole_dataset", 
    "process_columns",
    "test_preprocess_whole_dataset",
    
    # data analysis
    "filter_data_until_date",
    "temporal_split_data",
    
    # weather data
    "WeatherDataCollector",
    
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
