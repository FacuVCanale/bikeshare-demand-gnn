# Gate Model Training System

This document explains how to use the refactored Gate model training system for EcoBici demand prediction.

## Overview

The Gate model is a binary classifier that predicts whether there will be any bike activity (departures or arrivals) in a cluster during the next time window. This is the first stage of a two-stage prediction pipeline.

## Architecture

The system is built with a modular, extensible architecture:

- **Feature Selection**: `src/features/feature_selector.py` - Handles feature filtering for Gate model
- **Dataset Management**: `src/dataset/gate_dataset.py` - Loads and preprocesses data with double dispatch
- **Model Implementations**: `src/models/` - CatBoost, LightGBM, XGBoost models
- **Training Pipeline**: `src/training/gate_trainer.py` - Main training orchestrator
- **CLI Script**: `scripts/train_gate_model.py` - Command-line interface

## Quick Start

### 1. Basic Training

Train a CatBoost model with default settings:

```bash
python scripts/train_gate_model.py --model-type catboost
```

### 2. Custom Hyperparameters

Train with specific hyperparameters:

```bash
python scripts/train_gate_model.py \
    --model-type catboost \
    --iterations 500 \
    --learning-rate 0.05 \
    --depth 8
```

### 3. Multiple Experiments

Run multiple experiments from a configuration file:

```bash
python scripts/train_gate_model.py --experiment-config configs/gate_experiments.json
```

## Feature Selection

The system automatically filters features to focus on those relevant for the Gate model:

### Included Features:
- **Cluster Info**: ID, centroid coordinates, station count
- **Temporal**: Hour, day of week, month, cyclic encodings, time flags
- **Activity Counts**: Departure/arrival counts (internal/external)
- **Activity Flags**: Binary indicators of activity
- **Weather**: Temperature, humidity, precipitation, wind, etc.
- **Lag Features**: Historical values at different time lags
- **Rolling Features**: Moving averages and standard deviations
- **Same Hour Features**: Values from same hour yesterday/last week

### Excluded Features:
- Demographic breakdowns (male/female, age groups)
- Trip durations and their statistics
- Bike model information
- Weather comfort derived features
- Trip-specific weather aggregations

This reduces features from ~470 to ~150 while maintaining predictive power.

## Usage Examples

### Command Line Options

```bash
# Basic usage
python scripts/train_gate_model.py --model-type catboost

# Custom data paths
python scripts/train_gate_model.py \
    --train-path data/custom/train.parquet \
    --val-path data/custom/val.parquet \
    --test-path data/custom/test.parquet

# Hyperparameter tuning
python scripts/train_gate_model.py \
    --model-type lightgbm \
    --hyperparameters '{"n_estimators": 800, "learning_rate": 0.05}'

# Feature group selection
python scripts/train_gate_model.py \
    --feature-groups temporal_base weather_base activity_counts

# Custom output
python scripts/train_gate_model.py \
    --model-type xgboost \
    --output-dir models/custom \
    --model-name my_gate_model
```

### Programmatic Usage

```python
from src.training.gate_trainer import GateTrainer

# Create trainer
trainer = GateTrainer(
    train_path='data/clustered/train_cluster_features.parquet',
    val_path='data/clustered/val_cluster_features.parquet', 
    test_path='data/clustered/test_cluster_features.parquet',
    output_dir='models/gate'
)

# Prepare data
trainer.prepare_data()

# Train model
trainer.train(
    model_type='catboost',
    hyperparameters={
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6
    }
)

# Get feature importance
importance_df = trainer.get_feature_importance()

# Save model
trainer.save_model('my_gate_model')
```

### Experiment Configuration

Create a JSON file with multiple experiment configurations:

```json
{
  "catboost_baseline": {
    "model_type": "catboost",
    "hyperparameters": {
      "iterations": 500,
      "learning_rate": 0.1,
      "depth": 6,
      "random_seed": 42
    }
  },
  "lightgbm_tuned": {
    "model_type": "lightgbm", 
    "hyperparameters": {
      "n_estimators": 800,
      "learning_rate": 0.05,
      "max_depth": 8
    }
  }
}
```

## Model Types

### CatBoost
- **Advantages**: Handles categorical features natively, robust to overfitting
- **Best for**: Default choice, good performance out-of-the-box
- **Key parameters**: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`

### LightGBM  
- **Advantages**: Fast training, memory efficient
- **Best for**: Large datasets, quick experimentation
- **Key parameters**: `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`

### XGBoost
- **Advantages**: Well-established, extensive documentation
- **Best for**: When you need proven reliability
- **Key parameters**: `n_estimators`, `learning_rate`, `max_depth`, `reg_alpha`, `reg_lambda`

## Output Files

For each trained model, the system saves:

- **Model file**: `.model` - The trained model
- **Results**: `_results.json` - Training metrics and configuration
- **Features**: `_features.json` - Feature information and metadata

Example output structure:
```
models/gate/
├── gate_catboost_20240115_143022.model
├── gate_catboost_20240115_143022_results.json
├── gate_catboost_20240115_143022_features.json
└── experiments_summary_20240115_143500.json
```

## Evaluation Metrics

The system reports the following metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

For the Gate model, **Recall** is particularly important since missing positive cases (false negatives) means the second-stage model won't be called when it should be.

## Adding New Models

To add a new model type:

1. Create a new model class in `src/models/` inheriting from `BaseModel`
2. Implement the required methods (`train`, `predict`, `get_metrics`, etc.)
3. Add a `prepare_for_yourmodel` method to `GateDataset`
4. Update the `ModelFactory` in `gate_trainer.py`
5. Add the new model type to the CLI choices

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use LightGBM instead of CatBoost
2. **Long training times**: Reduce iterations or increase learning rate
3. **Poor performance**: Check feature selection and class imbalance
4. **Import errors**: Ensure all dependencies are installed (`catboost`, `lightgbm`, `xgboost`)

### Dependencies

Make sure you have the required packages:
```bash
pip install catboost lightgbm xgboost polars pandas scikit-learn
```

## Next Steps

After training a Gate model:

1. Evaluate performance on the test set
2. Analyze feature importance to understand what drives predictions
3. Consider hyperparameter tuning with Optuna or similar
4. Deploy the model for the first stage of the two-stage pipeline
5. Train the second-stage Informer model for detailed predictions 