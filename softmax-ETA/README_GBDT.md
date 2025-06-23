# GBDT Training for ETA and Destination Prediction

This module provides a comprehensive solution for training Gradient Boosting Decision Tree (GBDT) models for both ETA (Estimated Time of Arrival) regression and destination prediction classification tasks.

## 📁 Files Overview

- **`evaluation.py`**: Reusable evaluation utilities for both regression and classification metrics
- **`train_baseline.py`**: Simple historical baseline model for comparison
- **`train_gbdt.py`**: Main GBDT training script supporting XGBoost and LightGBM
- **`run_experiments.py`**: Example script for running multiple experiments
- **`feat-eng.py`**: Feature engineering and data preprocessing script

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have the required packages installed:

```bash
pip install polars pandas numpy scikit-learn xgboost lightgbm joblib
```

### 2. Generate Data Splits

First, run the feature engineering script to create train/val/test splits:

```bash
python feat-eng.py
```

This will create:
- `../data/processed/trips_train.parquet`
- `../data/processed/trips_val.parquet` 
- `../data/processed/trips_test.parquet`

### 3. Train Baseline Model

Start with the simple baseline:

```bash
python train_baseline.py
```

### 4. Train GBDT Models

#### Basic Usage

```bash
# LightGBM with default parameters
python train_gbdt.py --model lgb

# XGBoost with default parameters
python train_gbdt.py --model xgb
```

#### Custom Hyperparameters

```bash
# LightGBM with custom parameters
python train_gbdt.py --model lgb \
  --eta-params "num_leaves=50,learning_rate=0.05,n_estimators=200" \
  --dest-params "num_leaves=100,learning_rate=0.03,n_estimators=300"

# XGBoost with custom parameters  
python train_gbdt.py --model xgb \
  --eta-params "max_depth=8,learning_rate=0.05,n_estimators=200" \
  --dest-params "max_depth=10,learning_rate=0.03,n_estimators=300"
```

#### Advanced Usage

```bash
python train_gbdt.py \
  --model lgb \
  --experiment-name "lgb_optimized" \
  --eta-params "num_leaves=50,learning_rate=0.05,n_estimators=200,subsample=0.9" \
  --dest-params "num_leaves=100,learning_rate=0.03,n_estimators=300,subsample=0.9" \
  --data-dir "../data/processed" \
  --output-dir "results"
```

## 📊 Command Line Arguments

### `train_gbdt.py` Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `lgb` | Model type: `xgb` (XGBoost) or `lgb` (LightGBM) |
| `--eta-params` | str | `""` | ETA model hyperparameters (comma-separated key=value) |
| `--dest-params` | str | `""` | Destination model hyperparameters (comma-separated key=value) |
| `--data-dir` | str | `../data/processed` | Directory with train/val/test parquet files |
| `--output-dir` | str | `results` | Directory to save results and models |
| `--experiment-name` | str | `{model}_default` | Name for this experiment |

### Hyperparameter Format

Hyperparameters should be provided as comma-separated key=value pairs:

```bash
--eta-params "num_leaves=50,learning_rate=0.05,n_estimators=200,subsample=0.8"
```

## 🎯 Default Hyperparameters

### LightGBM Defaults

**ETA Regression:**
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

**Destination Classification:**
```python
{
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### XGBoost Defaults

**ETA Regression:**
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
```

**Destination Classification:**
```python
{
    'objective': 'multi:softprob',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
```

## 📈 Evaluation Metrics

### ETA Prediction (Regression)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **R²**: R-squared coefficient

### Destination Prediction (Classification)
- **Accuracy**: Overall accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **Top-5 Accuracy**: Accuracy considering top 5 predictions
- **Top-10 Accuracy**: Accuracy considering top 10 predictions

## 🗂️ Output Files

After training, the following files are generated:

```
results/
├── {experiment_name}_results.csv          # Evaluation metrics
├── {experiment_name}_predictions.parquet  # Test set predictions
├── {experiment_name}_feature_importance.csv # Feature importance
└── models/
    └── {experiment_name}/
        ├── {model}_eta_model.pkl           # ETA regression model
        ├── {model}_dest_model.pkl          # Destination classification model
        ├── {model}_label_encoder.pkl       # Label encoder for destinations
        └── {model}_metadata.json           # Model metadata and hyperparameters
```

## 🧪 Running Experiments

Use the provided experiment runner:

```bash
python run_experiments.py
```

This will run multiple experiments with different configurations:
- LightGBM with default parameters
- XGBoost with default parameters  
- LightGBM with tuned parameters
- XGBoost with tuned parameters
- LightGBM with fast training parameters

## 📝 Example Usage Scenarios

### 1. Quick Baseline Comparison

```bash
# Train baseline
python train_baseline.py

# Train default LightGBM
python train_gbdt.py --model lgb --experiment-name lgb_baseline

# Train default XGBoost  
python train_gbdt.py --model xgb --experiment-name xgb_baseline
```

### 2. Hyperparameter Tuning

```bash
# Test different learning rates
python train_gbdt.py --model lgb --experiment-name lgb_lr01 --eta-params "learning_rate=0.01"
python train_gbdt.py --model lgb --experiment-name lgb_lr05 --eta-params "learning_rate=0.05" 
python train_gbdt.py --model lgb --experiment-name lgb_lr1 --eta-params "learning_rate=0.1"

# Test different tree complexity
python train_gbdt.py --model lgb --experiment-name lgb_leaves31 --eta-params "num_leaves=31"
python train_gbdt.py --model lgb --experiment-name lgb_leaves50 --eta-params "num_leaves=50"
python train_gbdt.py --model lgb --experiment-name lgb_leaves100 --eta-params "num_leaves=100"
```

### 3. Production Model Training

```bash
# Train final optimized model
python train_gbdt.py \
  --model lgb \
  --experiment-name production_lgb \
  --eta-params "num_leaves=64,learning_rate=0.03,n_estimators=500,subsample=0.9,colsample_bytree=0.9" \
  --dest-params "num_leaves=128,learning_rate=0.02,n_estimators=800,subsample=0.9,colsample_bytree=0.9"
```

## 🔧 Extending the Framework

### Adding New Models

To add support for new models, modify the `GBDTTrainer` class in `train_gbdt.py`:

1. Add model import and availability check
2. Add default hyperparameters in `_get_default_*_params()` methods
3. Add training logic in the `fit()` method
4. Add prediction logic in the `predict()` method

### Adding New Metrics

To add new evaluation metrics, modify `evaluation.py`:

1. Add metric calculation functions
2. Update `evaluate_eta_prediction()` or `evaluate_destination_prediction()` 
3. Update `print_evaluation_results()` to display new metrics

### Custom Feature Engineering

To modify feature engineering:

1. Edit the `_prepare_features()` method in `GBDTTrainer`
2. Update the feature selection logic
3. Add new categorical encodings or transformations

## 🚨 Troubleshooting

### Common Issues

1. **Data files not found**: Make sure to run `feat-eng.py` first
2. **Import errors**: Install required packages (`pip install xgboost lightgbm`)
3. **Memory errors**: Reduce dataset size or use smaller hyperparameters
4. **Poor performance**: Try different hyperparameters or feature engineering

### Performance Tips

1. Use `num_leaves` instead of `max_depth` for LightGBM
2. Start with smaller `n_estimators` for faster iteration
3. Use `n_jobs=-1` to utilize all CPU cores
4. Monitor validation metrics to avoid overfitting

## 📊 Comparing Results

After running multiple experiments, compare results using:

```python
import pandas as pd

# Load all result files
results = []
for file in Path("results").glob("*_results.csv"):
    df = pd.read_csv(file)
    results.append(df)

# Combine and compare
combined_results = pd.concat(results, ignore_index=True)
print(combined_results[['model', 'eta_r2', 'dest_accuracy']].sort_values('eta_r2', ascending=False))
``` 