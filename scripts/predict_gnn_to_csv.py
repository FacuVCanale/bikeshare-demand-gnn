#!/usr/bin/env python3
"""
Generate GNN predictions on test data and save to CSV format.

This script loads a trained GNN model (best_model.pt or final_model.pt) along with 
its architecture, runs predictions on the test dataset, and saves the results in 
CSV format with columns: datetime, station_id, pred_arrivals, arrivals.

The CSV includes both predicted and actual values for easy comparison and evaluation.

Usage:
    python scripts/predict_gnn_to_csv.py \
        --model_path experiments/gnn/my_experiment/best_model.pt \
        --data_dir data/gnn_ready \
        --output_csv predictions/test_predictions.csv

Author: EcoBici-AI

## **📊 Salida del Script**

El script genera un CSV con formato:
```csv
datetime,station_id,pred_arrivals,arrivals
2023-07-01 12:00:00,101,2.34,2.0
2023-07-01 12:00:00,102,1.78,2.0
2023-07-01 12:00:00,103,0.92,1.0
2023-07-01 12:30:00,101,3.21,3.0
...
```

Y muestra estadísticas como:
```
Prediction Summary:
  Total predictions: 125,430
  Unique timestamps: 8,362
  Unique stations: 15
  Target column: pred_arrivals
  Prediction range: [0.0000, 12.4567]
  Mean prediction: 2.1234
  Actual arrivals range: [0.0000, 11.2345] 
  Mean actual arrivals: 2.0567

Evaluation Metrics:
  MSE: 0.8234
  RMSE: 0.9074
  MAE: 0.6543
  R²: 0.8542
```
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
import sys
from typing import Dict, Any, List, Optional
import warnings

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_models import create_gnn_model
from src.training.gnn_trainer import GNNTrainer


def load_trained_model(model_path: str, device: str = 'auto') -> torch.nn.Module:
    """
    Load a trained GNN model from checkpoint.
    
    Args:
        model_path: Path to saved model (.pt file)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading trained model from {model_path}")
    
    # determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # extract model information
    if 'model_class' in checkpoint and 'model_init_params' in checkpoint:
        # new format with complete model info
        model_class = checkpoint['model_class']
        model_params = checkpoint['model_init_params']
        state_dict = checkpoint['model_state_dict']
        
        print(f"  Model type: {model_class}")
        print(f"  Model parameters: {model_params}")
        
        # map class name to model type string for create_gnn_model
        class_to_type = {
            'TemporalGCN': 'gcn',
            'SpatialGAT': 'gat', 
            'GraphSAGE': 'sage',
            'GraphTransformer': 'transformer',
            'HybridSpatioTemporalGNN': 'hybrid'
        }
        
        model_type = class_to_type.get(model_class)
        if model_type is None:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # create model
        model = create_gnn_model(
            model_type=model_type,
            **model_params
        )
        
    else:
        # old format - try to infer from state dict
        raise ValueError(
            "Model checkpoint missing architecture information. "
            "Please use a model trained with the current training pipeline."
        )
    
    # load state dict, handling DataParallel wrapper
    # handle both "module." at the beginning and ".module." in the middle of keys
    if any('module.' in key for key in state_dict.keys()):
        print("  Detected DataParallel model, cleaning state_dict keys...")
        new_state_dict = {}
        for key, value in state_dict.items():
            # remove 'module.' from anywhere in the key
            new_key = key.replace('.module.', '.').replace('module.', '')
            new_state_dict[new_key] = value
            if key != new_key:
                print(f"    {key} -> {new_key}")
        state_dict = new_state_dict
    
    # try to load state dict with error handling
    try:
        model.load_state_dict(state_dict)
        print(f"  State dict loaded successfully")
    except RuntimeError as e:
        print(f"  Warning: Error loading state_dict: {e}")
        print("  Attempting to load with strict=False...")
        
        # try loading with strict=False to ignore missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
        
        print("  Model loaded with some parameter mismatches (strict=False)")
    
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully on {device}")
    return model


def load_gnn_data(data_dir: str) -> Dict[str, Any]:
    """
    Load preprocessed GNN data and metadata.
    
    Args:
        data_dir: Directory containing GNN data files
        
    Returns:
        Dictionary containing test data and metadata
    """
    data_path = Path(data_dir)
    
    print(f"Loading GNN data from {data_dir}")
    
    # load test data
    test_data = torch.load(data_path / 'test_data.pt', weights_only=False)
    
    # load metadata
    with open(data_path / 'train_feature_names.json', 'r') as f:
        metadata = json.load(f)
    
    with open(data_path / 'processing_config.json', 'r') as f:
        processing_config = json.load(f)
    
    # load station mapping
    station_mapping_file = data_path / 'station_mapping.json'
    if station_mapping_file.exists():
        with open(station_mapping_file, 'r') as f:
            station_id_mapping = json.load(f)
    else:
        # fallback to metadata
        station_id_mapping = metadata.get('station_id_mapping', {})
    
    # create reverse mapping: node_idx -> station_id
    node_to_station = {int(v): int(k) for k, v in station_id_mapping.items()}
    
    print(f"  Test data type: {type(test_data)}")
    if isinstance(test_data, (list, tuple)):
        print(f"  Number of test snapshots: {len(test_data)}")
        if len(test_data) > 0:
            print(f"  Nodes per snapshot: {test_data[0].x.shape[0]}")
            print(f"  Features per node: {test_data[0].x.shape[1]}")
            print(f"  Targets per node: {test_data[0].y.shape[1] if len(test_data[0].y.shape) > 1 else 1}")
    else:
        print(f"  Test nodes: {test_data.x.shape[0] if hasattr(test_data, 'x') else 'N/A'}")
    
    print(f"  Station mapping: {len(station_id_mapping)} stations")
    print(f"  Feature names: {len(metadata['features'])} features")
    print(f"  Target names: {metadata['targets']}")
    
    return {
        'test_data': test_data,
        'station_id_mapping': station_id_mapping,
        'node_to_station': node_to_station,
        'feature_names': metadata['features'],
        'target_names': metadata['targets'],
        'n_features': metadata['n_features'],
        'n_targets': metadata['n_targets'],
        'processing_config': processing_config
    }


def make_predictions(model: torch.nn.Module, test_data, device: str) -> np.ndarray:
    """
    Make predictions on test data.
    
    Args:
        model: Trained GNN model
        test_data: Test data (either list of snapshots or single graph)
        device: Device to run predictions on
        
    Returns:
        Predictions as numpy array
    """
    print("Making predictions on test data...")
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        if isinstance(test_data, (list, tuple)):
            # snapshot mode - predict on each snapshot
            print(f"  Processing {len(test_data)} snapshots...")
            
            for i, snapshot in enumerate(test_data):
                snapshot = snapshot.to(device)
                predictions = model(snapshot)
                all_predictions.append(predictions.cpu().numpy())
                
                if (i + 1) % 1000 == 0 or i + 1 == len(test_data):
                    print(f"    Processed {i+1}/{len(test_data)} snapshots")
            
            # stack predictions: [n_snapshots, n_nodes, n_targets]
            predictions = np.stack(all_predictions, axis=0)
            
        else:
            # single graph mode
            print("  Processing single graph...")
            test_data = test_data.to(device)
            predictions = model(test_data).cpu().numpy()
            # add snapshot dimension for consistency
            predictions = predictions[np.newaxis, ...]
    
    print(f"  Predictions shape: {predictions.shape}")
    return predictions


def create_predictions_dataframe(
    predictions: np.ndarray,
    test_data,
    node_to_station: Dict[int, int],
    target_names: List[str],
    target_col: str = 'pred_arrivals'
) -> pd.DataFrame:
    """
    Create DataFrame with predictions mapped to datetime and station_id.
    
    Args:
        predictions: Predictions array [n_snapshots, n_nodes, n_targets]
        test_data: Test data with datetime information
        node_to_station: Mapping from node index to station_id
        target_names: Names of target columns
        target_col: Name for prediction column in output
        
    Returns:
        DataFrame with columns: datetime, station_id, pred_arrivals, arrivals
    """
    print("Creating predictions DataFrame...")
    
    rows = []
    
    if isinstance(test_data, (list, tuple)):
        # snapshot mode
        print(f"  Processing {len(test_data)} snapshots...")
        
        for snapshot_idx, snapshot in enumerate(test_data):
            datetime_str = getattr(snapshot, 'datetime', None)
            if datetime_str is None:
                # fallback to index-based datetime
                datetime_str = f"snapshot_{snapshot_idx}"
                warnings.warn(f"No datetime found in snapshot {snapshot_idx}, using index")
            
            # get predictions and targets for this snapshot
            snapshot_preds = predictions[snapshot_idx]  # [n_nodes, n_targets]
            snapshot_targets = snapshot.y.cpu().numpy()  # [n_nodes, n_targets]
            
            # create row for each node (station)
            for node_idx in range(snapshot_preds.shape[0]):
                if node_idx in node_to_station:
                    station_id = node_to_station[node_idx]
                    
                    # handle single vs multiple targets for predictions
                    if len(target_names) == 1:
                        pred_value = snapshot_preds[node_idx, 0] if snapshot_preds.ndim > 1 else snapshot_preds[node_idx]
                        actual_value = snapshot_targets[node_idx, 0] if snapshot_targets.ndim > 1 else snapshot_targets[node_idx]
                    else:
                        # for multiple targets, use arrivals if available
                        if 'arrivals' in target_names:
                            arrivals_idx = target_names.index('arrivals') 
                            pred_value = snapshot_preds[node_idx, arrivals_idx]
                            actual_value = snapshot_targets[node_idx, arrivals_idx]
                        else:
                            # fallback to first target
                            pred_value = snapshot_preds[node_idx, 0]
                            actual_value = snapshot_targets[node_idx, 0]
                    
                    rows.append({
                        'datetime': datetime_str,
                        'station_id': station_id,
                        target_col: float(pred_value),
                        'arrivals': float(actual_value)
                    })
    else:
        # single graph mode - create single timestamp
        print("  Processing single graph...")
        
        datetime_str = getattr(test_data, 'datetime', 'single_prediction')
        snapshot_preds = predictions[0]  # [n_nodes, n_targets]
        snapshot_targets = test_data.y.cpu().numpy()  # [n_nodes, n_targets]
        
        for node_idx in range(snapshot_preds.shape[0]):
            if node_idx in node_to_station:
                station_id = node_to_station[node_idx]
                
                # handle single vs multiple targets
                if len(target_names) == 1:
                    pred_value = snapshot_preds[node_idx, 0] if snapshot_preds.ndim > 1 else snapshot_preds[node_idx]
                    actual_value = snapshot_targets[node_idx, 0] if snapshot_targets.ndim > 1 else snapshot_targets[node_idx]
                else:
                    # for multiple targets, use arrivals or first target
                    if 'arrivals' in target_names:
                        arrivals_idx = target_names.index('arrivals')
                        pred_value = snapshot_preds[node_idx, arrivals_idx]
                        actual_value = snapshot_targets[node_idx, arrivals_idx]
                    else:
                        pred_value = snapshot_preds[node_idx, 0]
                        actual_value = snapshot_targets[node_idx, 0]
                
                rows.append({
                    'datetime': datetime_str,
                    'station_id': station_id,
                    target_col: float(pred_value),
                    'arrivals': float(actual_value)
                })
    
    df = pd.DataFrame(rows)
    
    # try to parse datetime if it's a string
    if df['datetime'].dtype == 'object':
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"  Successfully parsed datetime column")
        except:
            print(f"  Warning: Could not parse datetime column, keeping as string")
    
    # sort by datetime and station_id
    df = df.sort_values(['datetime', 'station_id']).reset_index(drop=True)
    
    print(f"  Created DataFrame with {len(df)} rows")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Stations: {df['station_id'].nunique()} unique stations")
    print(f"  Prediction stats: min={df[target_col].min():.4f}, max={df[target_col].max():.4f}, mean={df[target_col].mean():.4f}")
    print(f"  Actual arrivals stats: min={df['arrivals'].min():.4f}, max={df['arrivals'].max():.4f}, mean={df['arrivals'].mean():.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate GNN predictions on test data and save to CSV')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--data_dir', type=str, default='data/gnn_ready',
                        help='Directory containing GNN data and metadata')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output path for predictions CSV file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--target_col', type=str, default='pred_arrivals',
                        help='Name for prediction column in output CSV')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Model type if not auto-detected (gcn, gat, sage, transformer, hybrid)')
    
    args = parser.parse_args()
    
    # validate inputs
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # create output directory
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"GNN Prediction Generation")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print(f"{'='*50}")
    
    # load data
    data = load_gnn_data(str(data_dir))
    
    # check if test data is empty (production mode)
    if isinstance(data['test_data'], (list, tuple)):
        if len(data['test_data']) == 0:
            print("ERROR: No test data available (production mode detected)")
            print("This script requires test data to generate predictions.")
            print("Please run the data preparation with validation/test splits enabled.")
            return
    else:
        if data['test_data'].x.numel() == 0:
            print("ERROR: No test data available (production mode detected)")
            print("This script requires test data to generate predictions.")
            print("Please run the data preparation with validation/test splits enabled.")
            return
    
    # load model
    model = load_trained_model(str(model_path), args.device)
    
    # determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # make predictions
    predictions = make_predictions(model, data['test_data'], device)
    
    # create predictions dataframe
    df_predictions = create_predictions_dataframe(
        predictions=predictions,
        test_data=data['test_data'],
        node_to_station=data['node_to_station'],
        target_names=data['target_names'],
        target_col=args.target_col
    )
    
    # save to CSV
    print(f"\nSaving predictions to {output_path}")
    df_predictions.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # calculate evaluation metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(df_predictions['arrivals'], df_predictions[args.target_col])
    mae = mean_absolute_error(df_predictions['arrivals'], df_predictions[args.target_col])
    rmse = np.sqrt(mse)
    r2 = r2_score(df_predictions['arrivals'], df_predictions[args.target_col])
    
    # print summary
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(df_predictions):,}")
    print(f"  Unique timestamps: {df_predictions['datetime'].nunique():,}")
    print(f"  Unique stations: {df_predictions['station_id'].nunique():,}")
    print(f"  Target column: {args.target_col}")
    print(f"  Prediction range: [{df_predictions[args.target_col].min():.4f}, {df_predictions[args.target_col].max():.4f}]")
    print(f"  Mean prediction: {df_predictions[args.target_col].mean():.4f}")
    print(f"  Actual arrivals range: [{df_predictions['arrivals'].min():.4f}, {df_predictions['arrivals'].max():.4f}]")
    print(f"  Mean actual arrivals: {df_predictions['arrivals'].mean():.4f}")
    
    print(f"\nEvaluation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # show sample of results
    print(f"\nSample predictions:")
    print(df_predictions.head(10).to_string(index=False))
    
    print(f"\n✓ Predictions saved successfully to {output_path}")


if __name__ == '__main__':
    main() 