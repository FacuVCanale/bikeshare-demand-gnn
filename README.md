# bikeshare-demand-gnn

Graph Neural Network models for real-time demand forecasting in the EcoBici public bike-sharing system of Buenos Aires, Argentina.

---

## Overview

EcoBici operates ~500 docking stations across Buenos Aires. Predicting how many arrivals and departures will occur at each station cluster in the next time window lets the operator rebalance bikes proactively, reduce empty/full station events, and improve service quality.

The challenge is inherently **spatial and temporal**: nearby station clusters are correlated, usage patterns shift by hour/day/season, and weather drives demand spikes. We model stations as nodes in a spatial graph and apply Graph Neural Networks to capture both dimensions simultaneously.

This project was developed as part of an academic course on machine learning applied to urban systems (1st semester 2025). All data comes from the publicly available EcoBici trip dataset published by the City of Buenos Aires.

---

## Problem Setup

| Dimension | Detail |
|---|---|
| Prediction target | Arrivals + departures between station clusters per time step |
| Time resolution | 30-minute intervals |
| Spatial resolution | 93 K-Means clusters of individual docking stations |
| Prediction horizon | Next time step (t+1) |
| Data range | Full EcoBici historical dataset; train / val / test split: up to 2023-01-01 / up to 2023-07-01 / rest |

---

## Why Graph Neural Networks?

Station clusters are not independent — a surge of departures at a business-district cluster will show up as arrivals at residential clusters 20 minutes later. A standard temporal model (e.g., per-station LSTM) misses this structure.

We build a **spatial graph** where each node is a station cluster and edges connect geographically close clusters (k-NN with distance threshold). A GNN message-passing step lets each node aggregate information from its neighbours before making a prediction, encoding spatial dependencies directly into the model.

---

## Architecture

### Graph construction

- Nodes: 93 station clusters (K-Means over GPS coordinates)
- Edges: k-nearest-neighbour connectivity (default k=5, max distance 5 km)
- Node features: 196-dimensional vector (temporal + lag + weather, described below)

### Input features (per node, per time step)

| Feature group | Examples |
|---|---|
| Temporal | Hour of day, day of week, month, rush-hour flags, holidays |
| Lag | Arrival/departure counts at t-1, t-2, … (sliding window of 24 steps) |
| Weather | Temperature, precipitation, wind speed (joined from Open-Meteo) |
| Cluster identity | Centroid lat/lon, number of docking stations |

### GNN models implemented

| Name | Key idea | Notes |
|---|---|---|
| `TemporalGCN` | Graph Convolutional Network | Fast baseline |
| `SpatialGAT` | Graph Attention Network | Learns edge weights via attention |
| `GraphSAGE` | Sample-and-aggregate | Scales to larger graphs |
| `GraphTransformer` | Transformer-style attention on graph | Best overall performance |
| `HybridSpatioTemporalGNN` | GCN + GAT combined | Strong temporal modelling |

The best-performing model is **GraphTransformer** — 5 `TransformerConv` layers, hidden dim 512, 8 attention heads, BatchNorm, GELU activation, 23.4M parameters.

---

## Results

Best checkpoint (GraphTransformer, validation set):

| Metric | Value |
|---|---|
| R² | **0.906** |
| MAE | 0.138 |
| RMSE | 0.225 |
| MSE | 0.051 |

The model jointly predicts arrivals and departures (2-dimensional output per cluster node) in a single forward pass.

---

## Stack

| Layer | Library / tool |
|---|---|
| GNN framework | PyTorch Geometric |
| Deep learning | PyTorch |
| Data processing | Polars, Pandas, NumPy |
| Clustering | scikit-learn (K-Means) |
| Baseline models | XGBoost, LightGBM, CatBoost |
| Experiment tracking | JSON checkpoints |
| Python version | 3.10+ |

---

## Repository Structure

```
EcoBici-AI/
├── src/
│   ├── models/          # GNN architectures (gnn_models.py, catboost_model.py, gbdt_models.py)
│   ├── clustering/      # K-Means station clustering and feature engineering
│   ├── dataset/         # Dataset classes for graph data loading
│   ├── training/        # Training loop, early stopping, metric logging
│   └── notebooks/       # Exploration and baseline notebooks
├── scripts/
│   ├── generate_cluster_dataset.py  # Step 1: cluster stations, build features
│   ├── prepare_gnn_dataset.py       # Step 2: build PyG graph objects
│   ├── train_gnn.py                 # Step 3: train and compare models
│   ├── test_model.py                # Step 4: evaluate a checkpoint
│   └── load_experiment.py           # Inspect saved experiments
├── data/                # Raw and processed data (not committed)
├── reports/             # Analysis outputs
├── model.md             # Architecture design notes
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or, for development extras:
pip install -e ".[dev]"
```

PyTorch Geometric requires a matching PyTorch + CUDA version. Follow [the official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2. Prepare data

Place the EcoBici raw files in `data/`:

```
data/
├── trips_with_weather.parquet
├── trips.parquet
└── users.parquet
```

The EcoBici trip dataset is published by the City of Buenos Aires at [data.buenosaires.gob.ar](https://data.buenosaires.gob.ar).

### 3. Build the dataset pipeline

```bash
# Cluster stations and generate aggregate features
python scripts/generate_cluster_dataset.py --n_clusters 93 --dt_minutes 30 --use_checkpoints

# Convert to PyTorch Geometric graph objects
python scripts/prepare_gnn_dataset.py --target both_external_counts --sequence_length 24 --k_neighbors 5
```

### 4. Train

```bash
# Train the best-performing model
python scripts/train_gnn.py \
    --model transformer \
    --hidden_dim 512 \
    --num_layers 5 \
    --num_heads 8 \
    --dropout 0.1 \
    --epochs 5000 \
    --patience 100 \
    --learning_rate 3e-4

# Or compare all architectures
python scripts/train_gnn.py --model all --epochs 200 --save_results
```

### 5. Evaluate

```bash
python scripts/test_model.py \
    --model_path experiments/gnn/<experiment_name>/final_model.pt \
    --hidden_dim 512 \
    --num_heads 8 \
    --save_predictions
```

---

## Key design choices

**Station clustering.** Individual docking stations are too granular and sparse — many have very few trips per 30-minute window. K-Means (k=93) reduces noise while preserving meaningful geographic units. The optimal k was found by elbow analysis (`scripts/find_optimal_k.py`).

**External vs. internal trips.** We separate _external_ flows (trips that cross cluster boundaries) from _internal_ ones (origin and destination in the same cluster). The GNN targets external flows because those are the ones that require rebalancing between stations.

**Data leakage prevention.** The dataset preparation script explicitly removes any features that would leak future information into the input window.

**Reproducibility.** All scripts accept a `--seed` parameter and use deterministic operations where PyTorch allows it. A `test_reproducibility.py` script verifies that two training runs with the same seed produce identical checkpoints.

---

## License

MIT — see [LICENSE](LICENSE).
