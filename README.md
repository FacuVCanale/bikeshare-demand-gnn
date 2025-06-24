# EcoBici-AI: Bike Trip Prediction Pipeline

A comprehensive machine learning pipeline for predicting bike trip durations, destinations, and station arrival patterns using Buenos Aires EcoBici public bike sharing data.

## 🚀 Features

- **Trip Duration Prediction (ETA)**: Predict how long a bike trip will take
- **Destination Prediction**: Predict which station a user will return the bike to
- **Station Arrival Analysis**: Predict arrival counts at each station over time windows
- **Multiple Models**: Baseline, XGBoost, and LightGBM implementations
- **Ultra-Fast Processing**: Optimized skeleton-based approach for station arrival analysis
- **Leak Prevention**: Optional temporal leak prevention for more realistic predictions

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Data Setup

Ensure you have the following data files:
- `data/processed/trips_with_weather.parquet` - Trip data with weather features
- `data/raw/combined/users.csv` - User data

## 🏃‍♂️ Quick Start

### Basic Training (Individual Trip Prediction Only)

```bash
# Train LightGBM model for trip prediction
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model lgb

# Train XGBoost model
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model xgb

# Train baseline model
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model baseline
```

### Full Pipeline with Station Arrival Analysis

```bash
# LightGBM with station arrivals (ultra-fast)
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model lgb \
  --arrivals

# With leak prevention (recommended for realistic evaluation)
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model lgb \
  --arrivals \
  --not-leak
```

### Advanced Options

```bash
# Custom time window and logging
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model xgb \
  --arrivals \
  --not-leak \
  --delta-t 3600 \
  --output-dir custom_results \
  --log-file logs/pipeline_run.log

# Force rebuild of train/test splits
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model lgb \
  --arrivals \
  --force-rebuild
```

## 📊 Command Line Arguments

### Required Arguments
- `--trips-path`: Path to trips_with_weather.parquet file
- `--users-path`: Path to users.csv file  
- `--model`: Model type (`baseline`, `xgb`, or `lgb`)

### Optional Arguments
- `--arrivals`: Enable station arrival analysis (default: disabled)
- `--not-leak`: **🚫 Prevent temporal leakage** - Skip predictions when departure and arrival are in same time window (recommended)
- `--delta-t`: Time window for arrival analysis in seconds (default: 1800 = 30 minutes)
- `--output-dir`: Directory to save results (default: `results`)
- `--predictions-dir`: Directory to save predictions (default: `predictions`)
- `--force-rebuild`: Force rebuilding of train/val/test splits
- `--experiment-name`: Optional experiment name for this run
- `--log-file`: Path to save complete log of the pipeline run

## 🎯 What is Leak Prevention (`--not-leak`)?

**Temporal leakage** occurs when a model makes predictions using information that wouldn't be available at prediction time. In our case:

- **Without `--not-leak`**: If a trip starts at 2:00 PM and is predicted to end at 2:15 PM, both fall in the same 30-minute window (2:00-2:30 PM), creating unrealistic "instant arrival" predictions.

- **With `--not-leak`**: These same-window predictions are excluded, creating more realistic evaluation conditions.

**Recommendation**: Always use `--not-leak` for realistic model evaluation.

## 📈 Pipeline Stages

1. **🔧 Feature Engineering**: Creates train/validation/test splits with engineered features
2. **🏗️ Model Training**: Trains the specified model (baseline/XGBoost/LightGBM)
3. **🔍 Individual Trip Evaluation**: Evaluates ETA and destination predictions
4. **🏢 Station Arrival Analysis**: Ultra-fast analysis of station arrival patterns (if `--arrivals` enabled)
5. **💾 Results Saving**: Saves all metrics and predictions

## 📊 Output Files

### Results Directory (`results/`)
- `{model_name}_pipeline_results.csv` - Complete evaluation metrics
- `fast_station_arrivals_metrics.csv` - Station arrival specific metrics (if `--arrivals` used)
- `station_arrivals_data.npz` - Raw arrival matrices (if `--arrivals` used)

### Predictions Directory (`predictions/`)
- `trips_test.parquet` - Test dataset
- `eta_predictions.npy` - Predicted trip durations
- `dest_predictions.npy` - Predicted destinations
- `dest_probabilities.npy` - Destination probability distributions

## ⚡ Standalone Fast Analysis

After running the pipeline, you can run station arrival analysis independently:

```bash
python softmax-ETA/fast_station_arrivals.py \
  --test-data predictions/trips_test.parquet \
  --eta-pred predictions/eta_predictions.npy \
  --dest-pred predictions/dest_predictions.npy \
  --delta-t 1800 \
  --not-leak
```

## 📋 Example Workflows

### Research & Development
```bash
# Quick baseline evaluation with leak prevention
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model baseline \
  --arrivals \
  --not-leak

# Full XGBoost evaluation with logging
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model xgb \
  --arrivals \
  --not-leak \
  --log-file logs/xgb_experiment.log
```

### Production Evaluation
```bash
# Best performing model with all features
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model lgb \
  --arrivals \
  --not-leak \
  --experiment-name "production_model_v1" \
  --output-dir production_results
```

## 🔍 Interpreting Results

The pipeline outputs comprehensive metrics:

### Individual Trip Metrics
- **ETA R²**: How well trip durations are predicted (higher = better)
- **Destination Accuracy**: Percentage of correctly predicted destinations
- **Destination Top-K Accuracy**: Percentage where true destination is in top K predictions

### Station Arrival Metrics  
- **Overall RMSE**: Root mean square error for arrival counts
- **Station Correlation**: How well station-level patterns are captured
- **Total Arrivals**: Comparison of predicted vs actual total arrivals

## 🚀 Performance Notes

- **Ultra-Fast Processing**: Station arrival analysis uses optimized skeleton method
- **Memory Efficient**: Processes large datasets without excessive memory usage
- **Parallel Ready**: Automatically uses available CPU cores
- **Incremental**: Skips feature engineering if splits already exist

## 🐛 Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `trips_with_weather.parquet` and `users.csv` exist
2. **Memory errors**: Try reducing dataset size or using `--delta-t` with larger time windows
3. **Slow performance**: Ensure you're using `--arrivals` (ultra-fast) vs old parallel methods

### Debug Mode
```bash
# Run with detailed logging
python softmax-ETA/pipeline.py \
  --trips-path data/processed/trips_with_weather.parquet \
  --users-path data/raw/combined/users.csv \
  --model lgb \
  --arrivals \
  --not-leak \
  --log-file debug.log
```

## 📞 Support

For issues or questions:
1. Check the log files for detailed error messages
2. Ensure all dependencies are correctly installed
3. Verify input data paths and formats

---

**Happy modeling! 🚴‍♀️🚴‍♂️**