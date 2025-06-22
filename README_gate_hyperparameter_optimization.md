# Gate Model Hyperparameter Optimization with OPTUNA

This document explains how to use the OPTUNA-based hyperparameter optimization system for Gate models, which is fully compatible with the existing gate training system.

## Overview

The hyperparameter optimization script (`scripts/optimize_gate_hyperparameters.py`) uses OPTUNA to automatically find the best hyperparameters for Gate models. It integrates seamlessly with the existing training pipeline and reuses all the existing code.

## Key Features

- **Full compatibility** with existing gate training system
- **Automatic hyperparameter search** using OPTUNA TPE (Tree-structured Parzen Estimator)
- **GPU support** with automatic fallback to CPU
- **Complete logging** of all trials and metrics
- **Automatic model saving** of the best performing model
- **Test set evaluation** of the best model
- **Progress monitoring** with real-time updates

## Quick Start

### 1. Basic Optimization

Run 50 trials to optimize a CatBoost model:

```bash
python scripts/optimize_gate_hyperparameters.py --model-type catboost --n-trials 50
```

### 2. Optimize Different Models

```bash
# optimize LightGBM for 100 trials
python scripts/optimize_gate_hyperparameters.py --model-type lightgbm --n-trials 100

# optimize XGBoost for 75 trials  
python scripts/optimize_gate_hyperparameters.py --model-type xgboost --n-trials 75
```

### 3. Choose Optimization Metric

```bash
# optimize for AUC instead of F1-score
python scripts/optimize_gate_hyperparameters.py --optimization-metric auc --n-trials 100

# optimize for recall (important for gate model)
python scripts/optimize_gate_hyperparameters.py --optimization-metric recall --n-trials 100
```

## Advanced Usage

### Timeout-based Optimization

Run optimization for a specific time limit:

```bash
# run for 1 hour (3600 seconds)
python scripts/optimize_gate_hyperparameters.py --timeout 3600 --model-type catboost
```

### Force CPU Training

Disable GPU and use only CPU:

```bash
python scripts/optimize_gate_hyperparameters.py --force-cpu --n-trials 50
```

### Custom Output and Naming

```bash
python scripts/optimize_gate_hyperparameters.py \
    --model-type catboost \
    --n-trials 100 \
    --output-dir models/optimized \
    --model-name best_catboost_gate \
    --results-filename catboost_optimization_results.json
```

### Skip Model Saving or Test Evaluation

```bash
# don't save the best model
python scripts/optimize_gate_hyperparameters.py --no-save-model --n-trials 50

# don't evaluate on test set
python scripts/optimize_gate_hyperparameters.py --no-test-eval --n-trials 50
```

## Hyperparameter Search Spaces

The script automatically defines sensible search spaces for each model type:

### CatBoost
- **iterations**: 500-3000 (step 100)
- **learning_rate**: 0.01-0.3 (log scale)
- **depth**: 4-10
- **l2_leaf_reg**: 1-20
- **border_count**: [32, 64, 128, 255]

### LightGBM
- **n_estimators**: 500-3000 (step 100)
- **learning_rate**: 0.01-0.3 (log scale)
- **max_depth**: 4-12
- **num_leaves**: 15-255
- **reg_alpha**: 0.01-10 (log scale)
- **reg_lambda**: 0.01-10 (log scale)
- **min_child_samples**: 10-100

### XGBoost
- **n_estimators**: 500-3000 (step 100)
- **learning_rate**: 0.01-0.3 (log scale)
- **max_depth**: 4-12
- **reg_alpha**: 0.01-10 (log scale)
- **reg_lambda**: 0.01-10 (log scale)
- **min_child_weight**: 1-10
- **subsample**: 0.6-1.0
- **colsample_bytree**: 0.6-1.0

## Output Files

For each optimization run, the following files are generated:

### Optimization Results
`optuna_optimization_{model_type}_{timestamp}.json` - Contains:
- Best hyperparameters found
- All trial results and metrics
- Optimization metadata and timing
- Test set evaluation (if enabled)

### Best Model Files
`gate_{model_type}_optuna_best_{timestamp}.model` - The best trained model
`gate_{model_type}_optuna_best_{timestamp}_results.json` - Training results
`gate_{model_type}_optuna_best_{timestamp}_features.json` - Feature information

## Example Output

```
=== EcoBici Gate Model Hyperparameter Optimization ===
Model type: catboost
Trials: 50
Optimization metric: f1_score
Training data: data/clustered/train_cluster_features.parquet
Validation data: data/clustered/val_cluster_features.parquet
Test data: data/clustered/test_cluster_features.parquet
Output directory: models/gate

=== Preparing data for optimization ===
Loading datasets...
Train shape: (1,234,567, 156)
Validation shape: (234,567, 156)
Test shape: (345,678, 156)

=== Starting OPTUNA Optimization ===
Model: CATBOOST
Optimization metric: f1_score (maximize)
Trials: 50
GPU: Yes (if available)

Trial 0: f1_score=0.8234
  Params: {"iterations": 1200, "learning_rate": 0.085, "depth": 6}
🎯 New best f1_score: 0.8234

Trial 1: f1_score=0.8156
  Params: {"iterations": 800, "learning_rate": 0.12, "depth": 5}

...

=== Optimization Completed ===
Total time: 1234.56 seconds
Best f1_score: 0.8456
Best hyperparameters: {
  "iterations": 1800,
  "learning_rate": 0.067,
  "depth": 7,
  "l2_leaf_reg": 4.2,
  "border_count": 128
}

=== Evaluating Best Model on Test Set ===
Test Metrics:
  accuracy: 0.9123
  precision: 0.8234
  recall: 0.8678
  f1_score: 0.8451
  auc: 0.9345

✅ Hyperparameter optimization completed!
```

## Integration with Existing Workflow

The optimization script is designed to work seamlessly with the existing gate training system:

1. **Uses same datasets**: Automatically loads from `data/clustered/`
2. **Reuses all code**: Built on top of `GateTrainer`, `GateDataset`, and model classes
3. **Compatible output**: Saves models in the same format as manual training
4. **Same feature selection**: Uses the existing `GateFeatureSelector`

You can use optimized models with any other part of the pipeline without modification.

## Best Practices

### Number of Trials
- **Quick exploration**: 20-50 trials
- **Thorough optimization**: 100-200 trials
- **Production optimization**: 300+ trials

### Optimization Metrics
- **f1_score**: Good balance for gate model (default)
- **auc**: Focus on ranking performance
- **recall**: Minimize false negatives (critical for gate)
- **precision**: Minimize false positives

### GPU Usage
- **Enable GPU** for faster training (default behavior)
- **Force CPU** only if GPU causes issues

### Time Management
- Use `--timeout` for time-limited optimization
- Monitor progress and stop early if convergence is reached

## Troubleshooting

### Common Issues

1. **OPTUNA not installed**:
   ```bash
   pip install optuna
   ```

2. **Memory issues**: Reduce number of parallel trials or use smaller datasets

3. **GPU errors**: Use `--force-cpu` to disable GPU training

4. **Long training times**: Reduce `--n-trials` or add `--timeout`

### Performance Tips

- Start with fewer trials (20-50) to get initial results quickly
- Use AUC as optimization metric for faster convergence
- Enable GPU for significant speedup (3-5x faster)
- Use timeout to avoid extremely long optimization runs

## Next Steps

After running optimization:

1. **Analyze results**: Review the optimization results JSON file
2. **Use best model**: The best model is automatically saved and ready to use
3. **Fine-tune further**: Use the best parameters as starting point for manual tuning
4. **Production deployment**: Deploy the optimized model in your pipeline

The optimized models are fully compatible with the existing EcoBici prediction pipeline and can be used as drop-in replacements for manually tuned models. 