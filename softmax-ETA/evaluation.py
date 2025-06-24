"""
Evaluation utilities for ETA prediction and destination prediction models.
This module provides reusable functions for computing metrics across different model types.
"""

import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd


def evaluate_eta_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ETA (duration) prediction performance.
    
    Args:
        y_true: True duration values
        y_pred: Predicted duration values
        
    Returns:
        Dictionary with MSE and R2 metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {"mse": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    return {
        "mse": mse,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int, class_labels: Optional[np.ndarray] = None) -> float:
    """
    Calculate top-k accuracy for multi-class classification.
    
    Args:
        y_true: True class labels (original labels, not encoded)
        y_proba: Prediction probabilities matrix (n_samples, n_classes)
        k: Number of top predictions to consider
        class_labels: Array of class labels corresponding to probability columns (optional)
        
    Returns:
        Top-k accuracy score
    """
    if y_proba.ndim == 1:
        # Binary case - convert to 2D
        y_proba = np.column_stack([1 - y_proba, y_proba])
    
    # Get top k prediction indices
    top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
    
    # If class_labels provided, map indices to actual labels
    if class_labels is not None:
        top_k_preds = class_labels[top_k_indices]
    else:
        # Assume labels are indices 0, 1, 2, ...
        top_k_preds = top_k_indices
    
    # Check if true label is in top k predictions
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def evaluate_destination_prediction(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None,
    class_labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Evaluate destination prediction performance.
    
    Args:
        y_true: True destination station IDs
        y_pred: Predicted destination station IDs
        y_proba: Prediction probabilities (optional, for top-k metrics)
        class_labels: List of all possible class labels (optional, needed for top-k accuracy)
        
    Returns:
        Dictionary with precision, recall, accuracy, and top-k accuracy metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-class, use macro averaging
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    # Top-k accuracy if probabilities are provided
    if y_proba is not None:
        try:
            class_labels_array = np.array(class_labels) if class_labels is not None else None
            metrics["top_5_accuracy"] = top_k_accuracy(y_true, y_proba, k=5, class_labels=class_labels_array)
            metrics["top_10_accuracy"] = top_k_accuracy(y_true, y_proba, k=10, class_labels=class_labels_array)
        except Exception as e:
            print(f"Warning: Could not compute top-k accuracy: {e}")
            metrics["top_5_accuracy"] = np.nan
            metrics["top_10_accuracy"] = np.nan
    else:
        metrics["top_5_accuracy"] = np.nan
        metrics["top_10_accuracy"] = np.nan
    
    return metrics


def print_evaluation_results(eta_metrics: Dict[str, float], dest_metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print formatted evaluation results for both tasks.
    
    Args:
        eta_metrics: ETA prediction metrics
        dest_metrics: Destination prediction metrics
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📊 ETA PREDICTION METRICS:")
    print(f"  MSE:  {eta_metrics['mse']:.2f}")
    print(f"  RMSE: {eta_metrics['rmse']:.2f}")
    print(f"  MAE:  {eta_metrics['mae']:.2f}")
    print(f"  R²:   {eta_metrics['r2']:.4f}")
    
    print(f"\n🎯 DESTINATION PREDICTION METRICS:")
    print(f"  Accuracy:       {dest_metrics['accuracy']:.4f}")
    print(f"  Precision:      {dest_metrics['precision']:.4f}")
    print(f"  Recall:         {dest_metrics['recall']:.4f}")
    
    if not np.isnan(dest_metrics['top_5_accuracy']):
        print(f"  Top-5 Accuracy: {dest_metrics['top_5_accuracy']:.4f}")
    if not np.isnan(dest_metrics['top_10_accuracy']):
        print(f"  Top-10 Accuracy: {dest_metrics['top_10_accuracy']:.4f}")


def save_evaluation_results(
    eta_metrics: Dict[str, float], 
    dest_metrics: Dict[str, float], 
    model_name: str,
    output_path: str
):
    """
    Save evaluation results to CSV file.
    
    Args:
        eta_metrics: ETA prediction metrics
        dest_metrics: Destination prediction metrics
        model_name: Name of the model
        output_path: Path to save the results
    """
    # Combine all metrics
    all_metrics = {
        "model": model_name,
        **{f"eta_{k}": v for k, v in eta_metrics.items()},
        **{f"dest_{k}": v for k, v in dest_metrics.items()}
    }
    
    # Convert to DataFrame and save
    df = pd.DataFrame([all_metrics])
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")


def compare_models(results_df: pd.DataFrame):
    """
    Compare multiple models from a results DataFrame.
    
    Args:
        results_df: DataFrame with model comparison results
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    # ETA metrics comparison
    print(f"\n📊 ETA PREDICTION COMPARISON:")
    eta_cols = [col for col in results_df.columns if col.startswith('eta_')]
    eta_comparison = results_df[['model'] + eta_cols].round(4)
    print(eta_comparison.to_string(index=False))
    
    # Destination metrics comparison
    print(f"\n🎯 DESTINATION PREDICTION COMPARISON:")
    dest_cols = [col for col in results_df.columns if col.startswith('dest_')]
    dest_comparison = results_df[['model'] + dest_cols].round(4)
    print(dest_comparison.to_string(index=False))
    
    # Highlight best models
    print(f"\n🏆 BEST PERFORMERS:")
    best_r2_idx = results_df['eta_r2'].idxmax()
    best_accuracy_idx = results_df['dest_accuracy'].idxmax()
    
    print(f"  Best ETA (R²):        {results_df.loc[best_r2_idx, 'model']} (R² = {results_df.loc[best_r2_idx, 'eta_r2']:.4f})")
    print(f"  Best Destination:     {results_df.loc[best_accuracy_idx, 'model']} (Acc = {results_df.loc[best_accuracy_idx, 'dest_accuracy']:.4f})")


def load_and_prepare_data(
    train_path: str, 
    val_path: str, 
    test_path: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load and prepare train/val/test datasets.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data  
        test_path: Path to test data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("Loading datasets...")
    
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)
    
    print(f"  Train: {len(train_df):,} trips")
    print(f"  Val:   {len(val_df):,} trips")
    print(f"  Test:  {len(test_df):,} trips")
    
    return train_df, val_df, test_df 