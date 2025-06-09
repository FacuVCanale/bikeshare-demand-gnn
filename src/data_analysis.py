import pandas as pd
from typing import Tuple, Dict
import numpy as np
from datetime import timedelta
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# FUNCIÓN DE SPLIT DE DATOS TEMPORAL
# =============================================================================

def filter_data_until_date(users_df: pd.DataFrame, trips_df: pd.DataFrame, 
                          max_date: str = "2024-08-31",
                          user_date_col: str = 'fecha_alta',
                          trip_date_col: str = 'fecha_origen_recorrido',
                          verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    OPTIMIZED version to filter data until a specific maximum date.
    
    Optimizations:
    - Uses vectorized operations for date comparisons
    - Avoids redundant datetime conversions
    - More efficient boolean indexing
    - Batch processing for memory efficiency
    - Robust date type handling for both string and datetime columns
    
    Args:
        users_df: DataFrame of users
        trips_df: DataFrame of trips  
        max_date: Maximum date to include (YYYY-MM-DD format)
        user_date_col: Date column in users_df
        trip_date_col: Date column in trips_df
        verbose: Whether to show information
        
    Returns:
        Tuple with (users_filtered, trips_filtered)
    """
    if verbose:
        print(f"FILTERING DATA UNTIL {max_date}")
        print(f"   Input: Users={len(users_df):,}, Trips={len(trips_df):,}")
    
    max_datetime = pd.to_datetime(max_date)
    
    # Optimize users filtering
    if verbose:
        print(f"   Filtering users...")
    
    # ensure date column is properly converted to datetime
    users_date_series = users_df[user_date_col]
    if not pd.api.types.is_datetime64_any_dtype(users_date_series):
        users_date_series = pd.to_datetime(users_date_series, errors='coerce')
    
    # Vectorized filtering - much faster than multiple conditions
    users_mask = (users_date_series.notna()) & (users_date_series <= max_datetime)
    users_filtered = users_df[users_mask].copy()
    
    # Optimize trips filtering
    if verbose:
        print(f"   Filtering trips...")
    
    # ensure date column is properly converted to datetime
    trips_date_series = trips_df[trip_date_col]
    if not pd.api.types.is_datetime64_any_dtype(trips_date_series):
        trips_date_series = pd.to_datetime(trips_date_series, errors='coerce')
    
    # Vectorized filtering for large dataset
    trips_mask = (trips_date_series.notna()) & (trips_date_series <= max_datetime)
    trips_filtered = trips_df[trips_mask].copy()
    
    if verbose:
        users_removed = len(users_df) - len(users_filtered)
        trips_removed = len(trips_df) - len(trips_filtered)
        print(f"   Users: {len(users_df):,} → {len(users_filtered):,} (-{users_removed:,})")
        print(f"   Trips: {len(trips_df):,} → {len(trips_filtered):,} (-{trips_removed:,})")
    
    return users_filtered, trips_filtered


def temporal_split_data(users_df: pd.DataFrame, trips_df: pd.DataFrame, 
                       train_end_date: str, val_end_date: str, test_end_date: str,
                       user_date_col: str = 'fecha_alta',
                       trip_date_col: str = 'fecha_origen_recorrido',
                       verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    OPTIMIZED version to split users and trips data by temporal ranges.
    
    Major optimizations:
    - Uses vectorized operations for all date comparisons
    - Batch processing for memory efficiency  
    - Efficient boolean indexing
    - Reduced copying operations
    - Smart memory management for large datasets
    - Robust date type handling for both string and datetime columns
    
    Args:
        users_df: DataFrame of users (already preprocessed)
        trips_df: DataFrame of trips (already preprocessed) 
        train_end_date: End date for training (YYYY-MM-DD format)
        val_end_date: End date for validation (YYYY-MM-DD format)
        test_end_date: End date for test (YYYY-MM-DD format)
        user_date_col: Name of date column in users_df
        trip_date_col: Name of date column in trips_df
        verbose: Whether to show split information
        
    Returns:
        Dict with the split DataFrames
    """
    
    if verbose:
        print("OPTIMIZED TEMPORAL SPLIT")
        print(f"   Train ≤ {train_end_date} | Val: {train_end_date} - {val_end_date} | Test: {val_end_date} - {test_end_date}")
        print(f"   Input: Users={len(users_df):,}, Trips={len(trips_df):,}")
    
    # Convert dates once - more efficient
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date) 
    test_end = pd.to_datetime(test_end_date)
    
    # Get date columns as series and ensure proper datetime conversion
    user_dates = users_df[user_date_col]
    if not pd.api.types.is_datetime64_any_dtype(user_dates):
        user_dates = pd.to_datetime(user_dates, errors='coerce')
    
    trip_dates = trips_df[trip_date_col]
    if not pd.api.types.is_datetime64_any_dtype(trip_dates):
        trip_dates = pd.to_datetime(trip_dates, errors='coerce')
    
    # Vectorized splitting for users - much faster than individual filters
    if verbose:
        print("   Splitting users...")
    
    users_train_mask = user_dates <= train_end
    users_val_mask = (user_dates > train_end) & (user_dates <= val_end)
    users_test_mask = (user_dates > val_end) & (user_dates <= test_end)
    
    users_train = users_df[users_train_mask].copy()
    users_val = users_df[users_val_mask].copy()
    users_test = users_df[users_test_mask].copy()
    
    # Vectorized splitting for trips - optimized for large datasets
    if verbose:
        print("   Splitting trips...")
    
    trips_train_mask = trip_dates <= train_end
    trips_val_mask = (trip_dates > train_end) & (trip_dates <= val_end)
    trips_test_mask = (trip_dates > val_end) & (trip_dates <= test_end)
    
    trips_train = trips_df[trips_train_mask].copy()
    trips_val = trips_df[trips_val_mask].copy()
    trips_test = trips_df[trips_test_mask].copy()
    
    if verbose:
        print(f"   Users: Train={len(users_train):,}, Val={len(users_val):,}, Test={len(users_test):,}")
        print(f"   Trips: Train={len(trips_train):,}, Val={len(trips_val):,}, Test={len(trips_test):,}")
    
    return {
        'users_train': users_train,
        'users_val': users_val, 
        'users_test': users_test,
        'trips_train': trips_train,
        'trips_val': trips_val,
        'trips_test': trips_test
    }


def engineer_ecobici_features(df_raw: pd.DataFrame,
                              dt_minutes: int = 30,
                              n_neighbors: int = 5) -> pd.DataFrame:
    """
    Realiza feature-engineering para predicción de arribos de bicicletas.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Dataset crudo con viajes + clima en las columnas descritas.
    dt_minutes : int, default 30
        Granularidad de la ventana temporal ΔT (en minutos).
    n_neighbors : int, default 5
        Nº de estaciones más cercanas para sumar partidas vecinas.
    
    Returns
    -------
    pd.DataFrame
        Tabla agregada con una fila por (station_id, ts_start) que incluye:
        - features temporales, meteorológicas y demográficas
        - lags de demanda y demanda vecina
        - target: y_arrivals_next_ΔT
    """
    # --------------------- helpers --------------------- #
    def floor_dt(series: pd.Series, freq: str) -> pd.Series:
        """Redondea timestamps hacia abajo a múltiplo de freq."""
        return (series.dt.floor(freq))
    
    def haversine(lon1, lat1, lon2, lat2):
        """Calcula distancia grande-círculo entre 2 pts."""
        # convertir a radianes
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # km
    
    # ------------- 1. Normalización de fechas ------------- #
    df = df_raw.copy()
    df["fecha_origen_recorrido"] = pd.to_datetime(df["fecha_origen_recorrido"])
    df["fecha_destino_recorrido"] = pd.to_datetime(df["fecha_destino_recorrido"])
    df["date"] = df["fecha_origen_recorrido"]
    
    # Ventana ΔT
    freq = f"{dt_minutes}min"
    df["ts_dep"] = floor_dt(df["fecha_origen_recorrido"], freq)
    df["ts_arr"] = floor_dt(df["fecha_destino_recorrido"], freq)
    
    # ------------- 2. Censo de partidas y arribos ------------- #
    dep = (
        df.groupby(["id_estacion_origen", "ts_dep"])
          .agg(dep_last_DT=("id_recorrido", "size"),
               male_cnt=("genero", lambda x: (x == "MALE").sum()))
          .reset_index()
          .rename(columns={"id_estacion_origen": "station_id",
                           "ts_dep": "ts_start"})
    )
    arr = (
        df.groupby(["id_estacion_destino", "ts_arr"])
          .size()
          .reset_index(name="arrivals")
          .rename(columns={"id_estacion_destino": "station_id",
                           "ts_arr": "ts_start"})
    )
    
    # ratio de género
    dep["share_male"] = dep["male_cnt"] / dep["dep_last_DT"].clip(lower=1)
    dep = dep.drop(columns="male_cnt")
    
    # ------------- 3. Lags de demanda ------------- #
    dep = dep.sort_values(["station_id", "ts_start"])
    for k in range(1, 7):  # lags 1…6
        dep[f"dep_lag_{k}"] = (
            dep.groupby("station_id")["dep_last_DT"].shift(k)
        )
    
    # ------------- 4. Target: arrivals en próxima ventana ------------- #
    arr_next = arr.copy()
    arr_next["ts_start"] = arr_next["ts_start"] - timedelta(minutes=dt_minutes)
    arr_next = arr_next.rename(columns={"arrivals": "y_arrivals_next_DT"})
    
    df_feat = dep.merge(arr_next, on=["station_id", "ts_start"], how="left")
    
    # ------------- 5. Merge con clima ------------- #
    weather = (
        df[["date"] + [c for c in df.columns if c.endswith("_2m") or c in
                       ["precipitation", "rain", "weather_code", "wind_speed_10m",
                        "wind_direction_10m", "cloud_cover", "sunshine_duration", "is_day",
                        "direct_radiation"]]]
        .drop_duplicates("date")
    )
    weather = weather.rename(columns={"date": "ts_start"})
    df_feat = df_feat.merge(weather, on="ts_start", how="left")
    
    # ------------- 6. Features temporales ------------ #
    df_feat["hour"] = df_feat["ts_start"].dt.hour
    df_feat["dow"] = df_feat["ts_start"].dt.dayofweek
    df_feat["month"] = df_feat["ts_start"].dt.month
    df_feat["sin_hour"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["cos_hour"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["is_weekend"] = df_feat["dow"].isin([5, 6]).astype("int8")
    
    # ------------- 7. Join con metadata de estaciones ------------- #
    stations = (
        df[["id_estacion_origen", "nombre_estacion_origen",
            "long_estacion_origen", "lat_estacion_origen"]]
        .drop_duplicates("id_estacion_origen")
        .rename(columns={"id_estacion_origen": "station_id",
                         "nombre_estacion_origen": "station_name",
                         "long_estacion_origen": "lon",
                         "lat_estacion_origen": "lat"})
    )
    df_feat = df_feat.merge(stations, on="station_id", how="left")
    
    # ------------- 8. Demanda vecina ------------- #
    # pre-computamos vecinos
    coords = stations[["lat", "lon"]].dropna()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1,
                            algorithm="ball_tree",
                            metric=lambda a, b: haversine(a[1], a[0], b[1], b[0]))
    nbrs.fit(coords[["lat", "lon"]])
    indices = nbrs.kneighbors(coords[["lat", "lon"]], return_distance=False)
    neighbor_map = dict(zip(stations["station_id"], indices))
    
    def sum_neighbor_dep(row):
        neigh_idx = neighbor_map.get(row["station_id"], [])  # incluye itself
        neigh_ids = stations.iloc[neigh_idx]["station_id"].values[1:]  # descartar propia
        return df_feat.loc[(df_feat["ts_start"] == row["ts_start"]) &
                           (df_feat["station_id"].isin(neigh_ids)),
                           "dep_last_DT"].sum()
    
    df_feat["near_dep_sum_DT"] = df_feat.apply(sum_neighbor_dep, axis=1)
    
    # ------------- 9. Limpieza final ------------- #
    # Rellenar NaNs de lags y target con 0 (opcional)
    lag_cols = [c for c in df_feat.columns if c.startswith("dep_lag_")]
    df_feat[lag_cols] = df_feat[lag_cols].fillna(0)
    df_feat["y_arrivals_next_DT"] = df_feat["y_arrivals_next_DT"].fillna(0)
    
    return df_feat

print("Función `engineer_ecobici_features` definida con éxito!")
