# Individual Station Bike Prediction Pipeline

This folder contains a complete pipeline for predicting bike arrivals and departures at individual stations using XGBoost and LightGBM models. The implementation follows the approach document requirements for converting trip-level data to 30-minute delta T format and creating comprehensive features.

## Files Overview

- **`feature_engineering.py`** - Converts trip data to delta T format and creates features
- **`training.py`** - Trains XGBoost and LightGBM models with proper validation
- **`evaluation.py`** - Comprehensive evaluation with metrics, plots, and feature importance
- **`main.py`** - Main pipeline orchestrating the complete workflow

## Features Implemented

Based on the approach document, the pipeline creates:

### Climate Features
- **Only 3 key weather features**: temperature, relative humidity, and apparent temperature
- 1-week lagged weather features for each

### Station-wise Features
- Last delta T arrivals/departures
- 1-week lag predictions (same time last week)
- Weekly moving averages (4-week and 12-week for same time slots)

### Time-wise Features
- Cyclical encoding for hour, day of week, month
- Weekend indicators
- Year normalization

### User-wise Features
- **Disabled**: User features are not included in the current implementation

## Quick Start

### Command Line Usage

```bash
# Basic usage with your data file (supports both CSV and Parquet)
python main.py --data_path /path/to/your/trips_with_weather.parquet
python main.py --data_path /path/to/your/trips_with_weather.csv

# For a specific station
python main.py --data_path /path/to/your/data.parquet --station_id 123

# Custom output directory
python main.py --data_path /path/to/your/data.parquet --output_dir my_results

# Use only XGBoost
python main.py --data_path /path/to/your/data.parquet --models xgboost

# Control null handling
python main.py --data_path /path/to/your/data.parquet --no_fill_nulls
python main.py --data_path /path/to/your/data.parquet --fill_strategy mean
python main.py --data_path /path/to/your/data.parquet --fill_strategy median

# Help
python main.py --help
```

### Programmatic Usage

```python
from main import BikePrediictionPipeline

# Configure pipeline
config = {
    'data_path': 'data/trips_with_weather.parquet',  # Supports .parquet or .csv
    'station_id': None,  # None for all stations, int for specific station
    'delta_t_minutes': 30,
    'model_types': ['xgboost', 'lightgbm'],
    'output_dir': 'results',
    'fill_nulls': True,  # Set to False to keep nulls
    'fill_strategy': 'zero'  # 'zero', 'mean', 'median', 'forward_fill'
}

# Run complete pipeline
pipeline = BikePrediictionPipeline(config)
results = pipeline.run_complete_pipeline()

# Access results
feature_engineering_results = results['feature_engineering']
training_results = results['training']
evaluation_results = results['evaluation']
```

## Input Data Format

The pipeline expects a CSV file with the following columns:

### Trip Data Columns
- `id_recorrido`, `duracion_recorrido`
- `fecha_origen_recorrido`, `id_estacion_origen`, `nombre_estacion_origen`
- `direccion_estacion_origen`, `long_estacion_origen`, `lat_estacion_origen`
- `fecha_destino_recorrido`, `id_estacion_destino`, `nombre_estacion_destino`
- `direccion_estacion_destino`, `long_estacion_destino`, `lat_estacion_destino`
- `id_usuario`, `modelo_bicicleta`, `genero`

### Weather Data Columns (Only 3 Required)
- `weather_temperature_2m` - Air temperature at 2 meters
- `weather_relative_humidity_2m` - Relative humidity at 2 meters  
- `weather_apparent_temperature` - Apparent/feels-like temperature
- **Note**: Only these 3 weather features are used, other weather columns are ignored

## Output Structure

```
results/
├── processed_data.csv          # Processed delta T format data
├── models/                     # Trained models
│   ├── xgboost_arrivals.joblib
│   ├── xgboost_departures.joblib
│   ├── lightgbm_arrivals.joblib
│   ├── lightgbm_departures.joblib
│   ├── feature_scaler.joblib
│   └── training_history.json
└── evaluation/                 # Evaluation results
    ├── metrics_summary.csv
    ├── feature_importance.csv
    └── plots/
        ├── predictions_vs_actual_arrivals.png
        ├── predictions_vs_actual_departures.png
        ├── residual_analysis_arrivals.png
        ├── residual_analysis_departures.png
        ├── feature_importance_arrivals.png
        ├── feature_importance_departures.png
        ├── time_series_arrivals_station_X.png
        ├── time_series_departures_station_X.png
        └── model_comparison.png
```

## Key Features

### 1. Feature Engineering
- Converts trip-level data to 30-minute time intervals
- Creates comprehensive features for bike prediction
- Handles missing values and data cleaning
- Supports both single station and multi-station processing

### 2. Model Training
- Trains both XGBoost and LightGBM models
- Proper time-based train/validation/test splits to avoid data leakage
- Feature scaling and hyperparameter optimization
- Model saving and loading capabilities

### 3. Comprehensive Evaluation
- Multiple regression metrics (RMSE, MAE, R², MAPE)
- Prediction vs actual scatter plots
- Residual analysis plots
- Time series prediction visualization
- **Feature importance analysis** (as required)
- Model comparison plots

### 4. Flexible Configuration
- Command line interface
- Programmatic API
- Configurable parameters for all pipeline steps
- Support for different time intervals and station filtering

### 5. Configurable Null Handling
- **Fill nulls** (default): Multiple strategies available
  - `zero`: Fill with zeros (default, safe for GBDT models)
  - `mean`: Fill with column means
  - `median`: Fill with column medians  
  - `forward_fill`: Forward fill within each station's time series
- **Keep nulls**: `--no_fill_nulls` preserves nulls for models that can handle them
- **Detailed reporting**: Shows null counts before and after processing

## Model Performance

The pipeline automatically identifies the best performing models for each target (arrivals/departures) based on validation RMSE and provides comprehensive performance analysis including:

- Training vs validation vs test metrics
- Overfitting analysis
- Feature importance rankings
- Visual performance comparisons

## Requirements

All required dependencies are already listed in the main `requirements.txt` file:
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- matplotlib, seaborn, plotly
- joblib

## Notes

- The pipeline uses time-based splitting by default to avoid data leakage
- Feature scaling is applied by default but can be disabled
- **Configurable null handling**: Choose whether and how to fill missing values
- Results are saved in organized directory structure for easy analysis
- Feature importance analysis provides insights into which features are most predictive
- **Focused approach**: Uses only 3 key weather features and excludes user features to avoid overwhelming the GBDT models
- **Experimentation-friendly**: Easy to test different preprocessing strategies

This implementation follows the "stations individual" approach mentioned in the document as the best performing method so far, with streamlined feature engineering optimized for GBDT performance. 