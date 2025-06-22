# ─────────────────────────────────────────────────────────────────────────────
# 0. Librerías
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import polars as pl
from cluster_utils import *

# ─────────────────────────────────────────────────────────────────────────────
# 1. Funciones utilitarias
def build_edge_index(coords: pd.DataFrame, k: int = 5):
    """Devuelve edge_index (2, E) y edge_weight (E,) usando KNN geo."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(
        coords[["lat", "lon"]]
    )
    dists, idxs = nbrs.kneighbors(coords[["lat", "lon"]])
    src, dst, w = [], [], []
    sigma = dists[:, 1:].mean()
    for i in range(coords.shape[0]):
        for j, dist in zip(idxs[i, 1:], dists[i, 1:]):
            src.append(i), dst.append(j)
            w.append(np.exp(-dist / sigma))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight


def create_tensors(df: pd.DataFrame, node_cols, global_cols, p, target_col, N):
    """
    df: tidy frame con multi-index (timestamp, cluster)
    Devuelve tensores X [T, N, F], y [T, N]
    """
    df = df.sort_index()

    # Asegúrate de que TODAS las columnas sean numéricas PRIMERO
    for col in node_cols + global_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Opcional: convierte booleanos a 0/1 (evita quedar como object)
    bool_mask = df.dtypes == bool
    df.loc[:, bool_mask.index[bool_mask]] = df.loc[:, bool_mask.index[bool_mask]].astype(int)

    # Elimina o completa los NaN creados al forzar numérico
    print(f"   ✓ Shape antes de dropna: {df.shape}")
    df = df.dropna()
    print(f"   ✓ Shape después de dropna: {df.shape}")

    # -------------------- features --------------------
    T = df.index.get_level_values(0).nunique()  # Number of unique timestamps

    X_node = df[node_cols].values.reshape(T, N, len(node_cols))
    
    # Fix: Global features are duplicated for each cluster, so we need to extract unique values per timestamp
    # Instead of reshaping, we'll extract one value per timestamp
    global_data = []
    timestamps = df.index.get_level_values(0).unique()
    
    for ts in timestamps:
        # Get the first cluster's global features for this timestamp (they should all be the same)
        ts_data = df.loc[ts]
        if isinstance(ts_data, pd.Series):
            # Only one cluster for this timestamp
            global_features = ts_data[global_cols].values
        else:
            # Multiple clusters, take the first one (they should be identical)
            global_features = ts_data.iloc[0][global_cols].values
        global_data.append(global_features)
    
    X_global = np.array(global_data)  # Shape: (T, G)
    X_global = np.repeat(X_global[:, None, :], N, axis=1)  # (T, N, G)
    
    y = df[target_col].values.reshape(T, N)

    # Castea todo a un tipo flotante homogéneo
    X_node   = X_node.astype(np.float32)
    X_global = X_global.astype(np.float32)
    y        = y.astype(np.float32)


    X : np.ndarray = np.concatenate([X_node, X_global], axis=-1)  # (T, N, F)
    y : np.ndarray = df[target_col].values.reshape(T, N)

    assert not np.isnan(X).any(), "Hay NaN en X"
    assert not np.isnan(y).any(), "Hay NaN en y"

    # -------------------- escaladores -----------------
    node_scaler = StandardScaler().fit(X_node.reshape(-1, len(node_cols)))
    X_node = node_scaler.transform(X_node.reshape(-1, len(node_cols))).reshape(
        X_node.shape
    )
    glob_scaler = StandardScaler().fit(X_global.reshape(-1, len(global_cols)))
    X_global = glob_scaler.transform(
        X_global.reshape(-1, len(global_cols))
    ).reshape(X_global.shape)
    X = np.concatenate([X_node, X_global], axis=-1)

    targ_scaler = MinMaxScaler().fit(y.reshape(-1, 1))      # ← reshape fix
    y = targ_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    # -------------------- ventanas --------------------
    seqs_X, seqs_y = [], []
    for t in range(p, X.shape[0] - 1):
        seqs_X.append(X[t - p : t])  # (p, N, F)
        seqs_y.append(y[t + 1])      # (N,)
    X_tensor = torch.tensor(np.stack(seqs_X), dtype=torch.float)
    y_tensor = torch.tensor(np.stack(seqs_y), dtype=torch.float)
    return X_tensor, y_tensor, (node_scaler, glob_scaler, targ_scaler)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Modelo con aristas aprendibles
class TGCNLearnableEdges(nn.Module):
    def __init__(self, in_feats, edge_index, edge_weight_init, hidden_feats=64):
        super().__init__()
        # edge_index fijo (buffer)  |  edge_weight optimizable (Parameter)
        self.register_buffer("edge_index", edge_index)
        self.edge_weight = nn.Parameter(edge_weight_init.clone())

        self.gcn1 = GCNConv(in_feats, hidden_feats, add_self_loops=True)
        self.gcn2 = GCNConv(hidden_feats, hidden_feats, add_self_loops=True)
        self.gru = nn.GRU(hidden_feats, hidden_feats, batch_first=True)
        self.fc = nn.Linear(hidden_feats, 1)

    def forward(self, x_seq):
        """
        x_seq: Tensor (B, p, N, F)
        """
        B, p, N, F = x_seq.size()
        g_outputs = []
        for t in range(p):
            x_t = x_seq[:, t].reshape(-1, F)  # (B*N, F)
            h = torch.relu(self.gcn1(x_t, self.edge_index, self.edge_weight))
            h = torch.relu(self.gcn2(h, self.edge_index, self.edge_weight))
            g_outputs.append(h.view(B, N, -1))  # (B, N, H)

        g_stack = torch.stack(g_outputs, dim=1)  # (B, p, N, H)
        B, p, N, H = g_stack.shape
        gru_in = g_stack.permute(0, 2, 1, 3).reshape(B * N, p, H)
        _, h_last = self.gru(gru_in)             # (1, B*N, H)
        out = self.fc(h_last.squeeze(0))         # (B*N, 1)
        return out.view(B, N)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main
if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento del modelo TGCN...")

    # 3.1 Carga de datos -----------------------------------------------------
    print("📊 Cargando datos...")
    
    # Load data with Polars first
    pdf_clustered = pl.read_parquet("data/processed/df_feat_final_clustered.parquet")
    print(f"   ✓ Datos cargados con Polars: {pdf_clustered.shape}")
    
    
    # Apply merge_intra_clusters to aggregate station-level data to cluster-level
    print("🔧 Agregando datos por cluster...")
    df_merged = merge_intra_clusters(pdf_clustered)
    print(f"   ✓ Datos cargados con Polars Post Merge: {df_merged.shape}")

    df_merged = df_merged.drop_nulls()
    print(f"   ✓ Datos cargados con Polars Post Nulls: {df_merged.shape}")

    del pdf_clustered

    # Convert to pandas for the rest of the pipeline
    df = df_merged.to_pandas()
    
    # Set MultiIndex (ts_start, cluster)
    df = df.set_index(['ts_start', 'cluster']).sort_index()
    
    # Fix: Calculate N as the number of clusters per timestamp, not unique clusters globally
    # Get the actual number of observations per timestamp
    obs_per_timestamp = df.groupby(level=0).size()
    N = obs_per_timestamp.iloc[0]  # Assume all timestamps have the same number of clusters
    
    # Verify consistency
    if not (obs_per_timestamp == N).all():
        print("⚠️  ADVERTENCIA: No todos los timestamps tienen el mismo número de clusters")
        print(f"   • Mínimo: {obs_per_timestamp.min()}")
        print(f"   • Máximo: {obs_per_timestamp.max()}")
        print(f"   • Promedio: {obs_per_timestamp.mean():.1f}")
        # Use the most common value
        N = obs_per_timestamp.mode().iloc[0]
        print(f"   • Usando el valor más común: {N}")
    
    print(f"   ✓ Datos procesados: {df.shape[0]} filas, {N} clusters por timestamp")

    # Now safely extract coordinates using the properly named index
    # Fix: We need to get coordinates for each unique cluster ID, not just group by level
    unique_clusters = df.index.get_level_values("cluster").unique()
    coords_list = []
    
    for cluster_id in unique_clusters:
        # Get first occurrence of this cluster to extract coordinates
        cluster_data = df[df.index.get_level_values("cluster") == cluster_id]
        if not cluster_data.empty:
            lat = cluster_data["lat"].iloc[0]
            lon = cluster_data["lon"].iloc[0] 
            coords_list.append({"cluster": cluster_id, "lat": lat, "lon": lon})
    
    coords = pd.DataFrame(coords_list).set_index("cluster")[["lat", "lon"]].reset_index(drop=True)

    edge_index, edge_weight_init = build_edge_index(coords, k=5)
    print(f"   ✓ Grafo construido: {edge_index.shape[1]} aristas")

    # 3.2 Tensores ----------------------------------------------------------
    print("🔧 Preparando tensores...")
    
    # Verify data consistency
    T = df.index.get_level_values(0).nunique()
    expected_rows = T * N
    actual_rows = df.shape[0]
    
    if expected_rows != actual_rows:
        print(f"⚠️  Datos inconsistentes: esperados {expected_rows}, reales {actual_rows}")
        print("   Filtrando a timestamps completos...")
        
        # Keep only timestamps that have exactly N observations
        complete_timestamps = df.groupby(level=0).size()
        complete_timestamps = complete_timestamps[complete_timestamps == N].index
        
        if len(complete_timestamps) == 0:
            raise ValueError("No hay timestamps con datos completos")
            
        df = df.loc[complete_timestamps]
        T = len(complete_timestamps)
        print(f"   ✓ Datos filtrados: {df.shape[0]} filas, {T} timestamps, {N} clusters")

    # Adapted feature columns for cluster-level data (after merge_intra_clusters)
    feature_cols_demand = [
        'dep_last_DT',
        'trip_dur_mean_last_DT',
        #'dep_lag_1', 'dep_lag_2', 'dep_lag_3', 'dep_lag_4', 'dep_lag_5', 'dep_lag_6',
        'arr_last_DT',
        #'arr_lag_1', 'arr_lag_2', 'arr_lag_3', 'arr_lag_4', 'arr_lag_5', 'arr_lag_6',
        # Add demand lags (now created by merge_intra_clusters)
        #'dem_lag_1', 'dem_lag_2', 'dem_lag_3', 'dem_lag_4', 'dem_lag_5', 'dem_lag_6',
        # Adding weather and temporal features
        'weather_temperature_2m',
        #'weather_relative_humidity_2m',
        'weather_apparent_temperature',
        'weather_precipitation',
        #'weather_wind_speed_10m',
        'weather_is_day',
        #'is_holiday_ar',
        'sin_hour', 'cos_hour',
        'sin_dow', 'cos_dow',
        'sin_month', 'cos_month',
        'is_weekend',
        #'payday_flag',
        #'vacation_season',
        'peak_commute'
    ]
    
    # Filter to only existing columns
    feature_cols_demand = [col for col in feature_cols_demand if col in df.columns]
    print(f"   ✓ Features disponibles: {len(feature_cols_demand)}")
    
    # Check what target columns are available
    possible_targets = ['y_demand_next_DT', 'y_arrivals_next_DT', 'y_departures_next_DT', 'delta_t']
    available_targets = [col for col in possible_targets if col in df.columns]
    
    if not available_targets:
        raise ValueError(f"No se encontraron columnas objetivo. Disponibles: {list(df.columns)}")
    
    # Use the first available target (prioritize demand, then arrivals)
    target = available_targets[0]
    print(f"   ✓ Variable objetivo: {target}")

    # Split features into node and global features
    # Node features: those that vary by cluster (aggregated arrivals/departures)
    # Global features: those that are the same for all clusters at a given time (weather, temporal, etc.)
    node_feature_prefixes = [
        'dep_last_DT', 'trip_dur_mean_last_DT',
        'dep_lag_', 'arr_last_DT', 'arr_lag_', 'dem_lag_'
    ]
    global_feature_prefixes = [
        'weather_', 'is_holiday_ar', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'sin_month', 'cos_month', 'is_weekend', 'payday_flag', 'vacation_season', 'peak_commute'
    ]

    def is_node_col(col):
        return any(col.startswith(prefix) for prefix in node_feature_prefixes)

    def is_global_col(col):
        return any(col.startswith(prefix) for prefix in global_feature_prefixes)

    node_cols = [c for c in feature_cols_demand if is_node_col(c)]
    global_cols = [c for c in feature_cols_demand if is_global_col(c)]

    p = 6
    print(f"   ✓ Features por nodo: {len(node_cols)}")
    print(f"   ✓ Features globales: {len(global_cols)}")
    print(f"   ✓ Ventana temporal: {p}")

    assert not df[feature_cols_demand + [target]].isna().any().any(), "Hay NaNs en el dataframe original"

    X_tensor, y_tensor, scalers = create_tensors(df, node_cols, global_cols, p, target, N)
    print(f"   ✓ Tensores creados: X={X_tensor.shape}, y={y_tensor.shape}")

    # 3.3 Split y DataLoaders ----------------------------------------------
    print("📂 Creando splits de datos...")
    total = X_tensor.size(0)
    tr, va = int(0.70 * total), int(0.85 * total)
    print(f"   ✓ Train: {tr} samples, Val: {va-tr} samples, Test: {total-va} samples")

    ds_train = torch.utils.data.TensorDataset(X_tensor[:tr], y_tensor[:tr])
    ds_val   = torch.utils.data.TensorDataset(X_tensor[tr:va], y_tensor[tr:va])
    ds_test  = torch.utils.data.TensorDataset(X_tensor[va:],  y_tensor[va:])

    ld_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    ld_val   = torch.utils.data.DataLoader(ds_val,   batch_size=64, shuffle=False)
    ld_test  = torch.utils.data.DataLoader(ds_test,  batch_size=64, shuffle=False)

    # 3.4 Dispositivo y modelo ---------------------------------------------
    print("🏗️  Inicializando modelo...")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Dispositivo: {dev}")

    edge_index = edge_index.to(dev)
    edge_weight_init = edge_weight_init.to(dev)

    model = TGCNLearnableEdges(
        in_feats=X_tensor.size(-1),
        edge_index=edge_index,
        edge_weight_init=edge_weight_init,
        hidden_feats=64,
    ).to(dev)
    print(f"   ✓ Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    mse = nn.MSELoss()

    # 3.5 Entrenamiento -----------------------------------------------------
    print("🎯 Comenzando entrenamiento...")
    best_val, wait, patience = np.inf, 0, 20
    for epoch in range(1, 151):
        # --- Train ---
        model.train()
        tr_loss = 0.0
        tr_y_true, tr_y_pred = [], []
        for xb, yb in ld_train:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            y_hat = model(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            # Save for R2 unnormalized
            tr_y_true.append(yb.cpu())
            tr_y_pred.append(y_hat.detach().cpu())

        # --- Val ---
        model.eval()
        val_loss = 0.0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for xv, yv in ld_val:
                xv, yv = xv.to(dev), yv.to(dev)
                y_pred = model(xv)
                val_loss += mse(y_pred, yv).item() * xv.size(0)
                # Guardar predicciones para R² desnormalizado
                val_y_true.append(yv.cpu())
                val_y_pred.append(y_pred.cpu())

        tr_loss /= len(ld_train.dataset)
        val_loss /= len(ld_val.dataset)
        
        # Calcular R² desnormalizado en validación
        val_y_true_cat = torch.cat(val_y_true).numpy().ravel()
        val_y_pred_cat = torch.cat(val_y_pred).numpy().ravel()
        
        # Desnormalizar usando el scaler del target
        targ_scaler = scalers[2]
        val_y_true_denorm = targ_scaler.inverse_transform(val_y_true_cat.reshape(-1, 1)).ravel()
        val_y_pred_denorm = targ_scaler.inverse_transform(val_y_pred_cat.reshape(-1, 1)).ravel()
        
        # Calcular R² desnormalizado
        val_r2_denorm = r2_score(val_y_true_denorm, val_y_pred_denorm)

        # Calcular R² desnormalizado en entrenamiento
        tr_y_true_cat = torch.cat(tr_y_true).numpy().ravel()
        tr_y_pred_cat = torch.cat(tr_y_pred).numpy().ravel()
        tr_y_true_denorm = targ_scaler.inverse_transform(tr_y_true_cat.reshape(-1, 1)).ravel()
        tr_y_pred_denorm = targ_scaler.inverse_transform(tr_y_pred_cat.reshape(-1, 1)).ravel()
        tr_r2_denorm = r2_score(tr_y_true_denorm, tr_y_pred_denorm)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:03d} | train MSE={tr_loss:.5f} | train R²={tr_r2_denorm:.4f} | val MSE={val_loss:.5f} | val R²={val_r2_denorm:.4f}")

        # Early-stopping
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), "best_tgcn_learnable.pt")
            if epoch % 10 == 0:
                print(f"   ✓ Nuevo mejor modelo guardado (val_loss={val_loss:.5f}, R²={val_r2_denorm:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("⏹️  Early stop!")
                break

    # 3.6 Evaluación --------------------------------------------------------
    print("📊 Evaluando modelo final...")
    model.load_state_dict(torch.load("best_tgcn_learnable.pt"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xt, yt in ld_test:
            xt = xt.to(dev)
            y_hat = model(xt).cpu()
            y_true.append(yt), y_pred.append(y_hat)
    y_true = torch.cat(y_true).numpy().ravel()
    y_pred = torch.cat(y_pred).numpy().ravel()

    # Des-normalizar demanda
    print("🔄 Des-normalizando predicciones...")
    targ_scaler = scalers[2]
    y_true = targ_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred = targ_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    print("\n─── Métricas test ────────────────────────────────")
    print(f"R²  = {r2_score(y_true, y_pred):.4f}")
    print(f"MAE = {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE = {mean_squared_error(y_true, y_pred):.4f}")
    print("✅ Entrenamiento completado!")
