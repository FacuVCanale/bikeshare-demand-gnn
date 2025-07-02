import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.gnn_models import create_gnn_model
from torch_geometric.data import Data

def load_batch(batch_path):
    batch = torch.load(batch_path, weights_only=True)
    x = batch['x']
    y = batch['y']
    edge_index = batch['edge_index']
    return x, y, edge_index

def load_metadata(metadata_dir):
    with open(Path(metadata_dir) / 'feature_names.json') as f:
        feature_names = json.load(f)
    with open(Path(metadata_dir) / 'station_id_mapping.json') as f:
        station_id_mapping = json.load(f)
    return feature_names, station_id_mapping

def evaluate_metrics(y_true, y_pred, metrics):
    results = {}
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    for i in range(y_true.shape[1]):
        res = {}
        if 'rmse' in metrics:
            res['rmse'] = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        if 'mae' in metrics:
            res['mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        if 'r2' in metrics:
            res['r2'] = r2_score(y_true[:, i], y_pred[:, i])
        results[i] = res
    return results

def load_model_from_checkpoint(model_dir, device='cpu', model_filename='best_model.pt'):
    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / model_filename, map_location=device, weights_only=False)
    
    model_class_name = checkpoint.get('model_class', 'gcn').lower()
    if model_class_name == 'graphtransformer':
        model_class_name = 'transformer'
    
    model_init_params = checkpoint.get('model_init_params', {})
    
    if not model_init_params.get('num_features'):
        raise ValueError("Could not determine 'num_features' from model checkpoint.")
    if not model_init_params.get('num_targets'):
        raise ValueError("Could not determine 'num_targets' from model checkpoint.")
        
    model = create_gnn_model(
        model_type=model_class_name,
        **model_init_params
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='GNN predict and evaluate metrics')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with best_model.pt and model_architecture.json')
    parser.add_argument('--batch_path', type=str, required=True, help='Path to inference_batch.pt')
    parser.add_argument('--metadata_dir', type=str, required=True, help='Directory with feature_names.json')
    parser.add_argument('--metrics', type=str, default='rmse,mae,r2', help='Comma-separated metrics: rmse,mae,r2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_filename', type=str, default='best_model.pt', help='Model checkpoint filename')
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',')]

    # Cargar batch
    x, y, edge_index = load_batch(args.batch_path)
    feature_names, station_id_mapping = load_metadata(args.metadata_dir)

    # Cargar modelo desde checkpoint
    model = load_model_from_checkpoint(args.model_dir, device=args.device, model_filename=args.model_filename)
    model = model.to(args.device)

    # Mover a device
    x = x.to(args.device)
    edge_index = edge_index.to(args.device)
    y = y.to(args.device)

    # Predicción
    with torch.no_grad():
            data_obj = Data(x=x, edge_index=edge_index)
            y_pred = model(data_obj)

    # Evaluación
    results = evaluate_metrics(y, y_pred, metrics)
    for i, res in results.items():
        print(f"Target {i}:")
        for m, v in res.items():
            print(f"  {m}: {v:.4f}")

if __name__ == '__main__':
    main() 