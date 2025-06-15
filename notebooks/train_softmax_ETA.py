# ============================================
# 1. Librerías
# ============================================
import torch, pandas as pd, numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys

sys.path.append("../src")  # <-- Ajustá según tu estructura de carpetas
from src.models.nn_multihead_model import BikeDestETAEnhanced              # <-- tu archivo con la clase

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
torch.manual_seed(SEED);  np.random.seed(SEED)

# ============================================
# 2. Cargar DataFrame
# ============================================
df = pd.read_parquet("../data/processed/ecobici_features.parquet")

# ---------- 2.1 Selección de columnas ----------
# --------------------------------------------
# LISTAS DEFINITIVAS DE FEATURES
# --------------------------------------------

cat_cols = [
    # ───────── variables categóricas originales ─────────
    "id_estacion_origen",
    "modelo_bicicleta",
    "genero",
    "weather_weather_code_x",
    "weather_is_day_x",
    # ───────── flags e indicadores ─────────
    "is_holiday_ar",
    "is_weekend",
    "payday_flag",
    "vacation_season",
    "peak_commute",
    "precip_flag",
    # ───────── calendarios (enteros) ─────────
    "hour",
    "dow",
    "month",
    "day"
]

cont_cols = [
    # ───────── ubicación ─────────
    "long_estacion_origen", "lat_estacion_origen",
    # ───────── tiempo (t-1) ─────────
    "weather_temperature_2m_x", "weather_relative_humidity_2m_x",
    "weather_dew_point_2m_x", "weather_apparent_temperature_x",
    "weather_precipitation_x", "weather_rain_x",
    "weather_pressure_msl_x", "weather_surface_pressure_x",
    "weather_cloud_cover_x", "weather_cloud_cover_low_x",
    "weather_cloud_cover_mid_x", "weather_cloud_cover_high_x",
    "weather_et0_fao_evapotranspiration_x",
    "weather_vapour_pressure_deficit_x",
    "weather_wind_speed_10m_x",
    "weather_wind_gusts_10m_x",
    "weather_soil_temperature_0_to_7cm_x",
    "weather_soil_moisture_0_to_7cm_x",
    "weather_sunshine_duration_x", "weather_direct_radiation_x",
    # ───────── variables ingenierizadas del viento ─────────
    "wind_dir_sin", "wind_dir_cos",
    # ───────── lags/rolling de precipitación ─────────
    "precipitation_lag_1h", "rain_lag_1h",
    # ───────── one-hot del código de clima ─────────
    "weather_code_cat_Clear", "weather_code_cat_Clouds",
    "weather_code_cat_Drizzle", "weather_code_cat_Rain",
    # ───────── demanda y estadísticos de operación ─────────
    "dep_last_DT", "trip_dur_mean_last_DT",
    "model_FIT_cnt", "model_ICONIC_cnt",
    "share_male", "share_female", "share_other",
    "dep_lag_1", "dep_lag_2", "dep_lag_3",
    "dep_lag_4", "dep_lag_5", "dep_lag_6",
    "arr_last_DT", "arr_lag_1", "arr_lag_2",
    "arr_lag_3", "arr_lag_4", "arr_lag_5", "arr_lag_6",
    # ───────── rolling 24 h y proximidad ─────────
    "dep_ma_24h", "dep_std_24h", "dep_ratio_DT_24h",
    "near_dep_sum_DT", "near_dep_lag_1",
    # ───────── codificaciones cíclicas ─────────
    "sin_hour", "cos_hour",
    "sin_dow",  "cos_dow",
    "sin_month","cos_month"
]
target_cls = 'id_estacion_origen'
target_eta = 'duracion_recorrido'

# ---------- 2.2 Encoding de categóricas ----------
encoders = {}
for col in cat_cols + [target_cls]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ---------- 2.3 Train / Val / Test ----------
train_df, test_df = train_test_split(df, test_size=0.15, shuffle=False)   # corte temporal
train_df, val_df  = train_test_split(train_df, test_size=0.15, shuffle=False)

# ============================================
# 3. Dataset & DataLoader
# ============================================
class BikeDataset(Dataset):
    def __init__(self, data):
        self.x_cat  = torch.tensor(data[cat_cols ].values, dtype=torch.long)
        self.x_cont = torch.tensor(data[cont_cols].values, dtype=torch.float32)
        self.y_cls  = torch.tensor(data[target_cls].values, dtype=torch.long)
        self.y_eta  = torch.tensor(data[target_eta].values, dtype=torch.float32)
    def __len__(self): return len(self.y_cls)
    def __getitem__(self, idx):
        return (self.x_cat[idx], self.x_cont[idx], self.y_cls[idx], self.y_eta[idx])

BATCH = 1024
train_loader = DataLoader(BikeDataset(train_df), batch_size=BATCH, shuffle=True,  num_workers=4)
val_loader   = DataLoader(BikeDataset(val_df),   batch_size=BATCH, shuffle=False, num_workers=4)
test_loader  = DataLoader(BikeDataset(test_df),  batch_size=BATCH, shuffle=False, num_workers=4)

# ============================================
# 4. Instanciar modelo
# ============================================
cat_dims = {c: int(df[c].max()) + 1 for c in cat_cols}
model = BikeDestETAEnhanced(cat_dims=cat_dims,
                            cont_dim=len(cont_cols),
                            num_stations=int(df[target_cls].max()) + 1).to(DEVICE)

# ---------- 4.1 Pesos de clases (desbalance) ----------
hist = np.bincount(train_df[target_cls].values,
                   minlength=model.dest_head[-1].out_features)
class_w = torch.tensor(1 / np.sqrt(hist + 1e-6), dtype=torch.float32)
class_w = class_w / class_w.mean()   # normalizá si querés
class_w = class_w.to(DEVICE)

# ============================================
# 5. Optimizador y scheduler
# ============================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ============================================
# 6. Loop de entrenamiento
# ============================================
EPOCHS = 30
best_val_f1, best_state = 0, None

for epoch in range(1, EPOCHS + 1):
    # ---------- 6.1 Entrenamiento ----------
    model.train()
    for x_cat, x_cont, y_cls, y_eta in train_loader:
        x_cat, x_cont = x_cat.to(DEVICE), x_cont.to(DEVICE)
        y_cls, y_eta  = y_cls.to(DEVICE), y_eta.to(DEVICE)

        dest_log, eta_mu, eta_sigma = model(x_cat, x_cont)
        loss, _, _ = model.loss(dest_log, y_cls, eta_mu, eta_sigma, y_eta,
                                class_weights=class_w)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    # ---------- 6.2 Validación ----------
    model.eval()
    correct, total, mae, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for x_cat, x_cont, y_cls, y_eta in val_loader:
            x_cat, x_cont = x_cat.to(DEVICE), x_cont.to(DEVICE)
            y_cls, y_eta  = y_cls.to(DEVICE), y_eta.to(DEVICE)

            dest_log, eta_mu, _ = model(x_cat, x_cont)
            pred_cls = dest_log.argmax(1)
            correct += (pred_cls == y_cls).sum().item()
            total   += y_cls.size(0)

            mae   += (eta_mu - y_eta).abs().sum().item()
            n     += y_eta.size(0)

    acc = correct / total
    mae = mae / n
    print(f"Epoch {epoch:02d} | Val Acc={acc*100:.2f}% | Val MAE={mae:.2f} min")

    # ---------- 6.3 Guardar mejor modelo ----------
    if acc > best_val_f1:             # usá F1 si lo calculás
        best_val_f1 = acc
        best_state  = model.state_dict()

# ============================================
# 7. Evaluación final en test
# ============================================
model.load_state_dict(best_state)
model.eval()
# ...misma métrica que arriba para test_loader

# ============================================
# 8. Serializar modelo y encoders
# ============================================
torch.save({
    'state_dict': model.state_dict(),
    'cat_dims'  : cat_dims,
    'encoders'  : encoders           # usá joblib si preferís
}, "bike_multi_task.pth")
