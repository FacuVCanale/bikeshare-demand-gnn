import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from src.models.gnn_models import create_gnn_model
import sys
sys.path.append(str(Path(__file__).parent.parent))

def load_batch(batch_path):
    batch = torch.load(batch_path)
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
    checkpoint = torch.load(model_dir / model_filename, map_location=device)
    model_class = checkpoint.get('model_class', 'gcn').lower()
    model_init_params = checkpoint.get('model_init_params', {})
    # fallback for missing params
    num_features = model_init_params.get('num_features')
    num_targets = model_init_params.get('num_targets')
    hidden_dim = model_init_params.get('hidden_dim', 128)
    dropout = model_init_params.get('dropout', 0.2)
    num_layers = model_init_params.get('num_layers', 3)
    num_heads = model_init_params.get('num_heads', 8)
    # reconstruir modelo
    model = create_gnn_model(
        model_type=model_class,
        num_features=num_features,
        num_targets=num_targets,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
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
        y_pred = model(x, edge_index)

    # Evaluación
    results = evaluate_metrics(y, y_pred, metrics)
    for i, res in results.items():
        print(f"Target {i}:")
        for m, v in res.items():
            print(f"  {m}: {v:.4f}")

if __name__ == '__main__':
    main() 