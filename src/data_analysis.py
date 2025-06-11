from __future__ import annotations
import pandas as pd
from typing import Tuple, Dict
import numpy as np
import polars as pl
from datetime import timedelta
import math
from sklearn.neighbors import NearestNeighbors
import os

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


def _haversine_np(lon1, lat1, lon2, lat2):
    """
    Haversine vectorizado (km) – usado como métrica en NearestNeighbors.
    """
    # convertir a radianes
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _build_skeleton(stations_df: pl.DataFrame, start_ts: pl.Series,
                    end_ts: pl.Series, freq: str) -> pl.DataFrame:
    """
    Devuelve un DataFrame con todas las combinaciones (station_id, ts_start)
    entre start y end inclusive al paso indicado.
    """
    ts_range = pl.datetime_range(
        start=start_ts,
        end=end_ts,
        interval=freq,
        eager=True,
        time_unit="us"
    ).alias("ts_start")
    skeleton = stations_df.select(["station_id"]).join(
        ts_range.to_frame(),
        how="cross"
    )
    return skeleton


def engineer_ecobici_features(
    df_raw: pd.DataFrame,
    dt_minutes: int = 30,
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Pipeline de feature-engineering optimizado con Polars para predicción
    de arribos de bicicletas en Ecobici.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Tabla original con viajes + clima (columnas enumeradas).
    dt_minutes : int, default 30
        Granularidad de la ventana temporal.
    n_neighbors : int, default 5
        Nº de estaciones más cercanas consideradas para la demanda vecina.

    Returns
    -------
    pd.DataFrame
        Dataset final con una fila por (station_id, ts_start).
    """
    print("🚴‍♂️  Iniciando feature-engineering…")

    # --- Checkpointing setup ---
    checkpoint_dir = "feature_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(f"  Intermediate results will be saved to '{checkpoint_dir}/'")
    print(f"  Input raw data shape: {df_raw.shape}")

    # ------------------------------------------------------------------
    # 0. Conversión inicial a Polars y normalización de fechas
    # ------------------------------------------------------------------
    print("\n[Step 0/9] Converting to Polars and normalizing dates...")
    df = pl.from_pandas(df_raw)

    # Ensure station ID columns are the same integer type for joins
    df = df.with_columns([
        pl.col("id_estacion_origen").cast(pl.Int64),
        pl.col("id_estacion_destino").cast(pl.Int64),
    ])

    for col in ("fecha_origen_recorrido", "fecha_destino_recorrido"):
        if col in df.columns:
            if df[col].dtype == pl.String:
                df = df.with_columns(
                    pl.col(col).str.to_datetime(time_unit="us")
                )
            else:
                df = df.with_columns(pl.col(col).cast(pl.Datetime("us")))

    freq = f"{dt_minutes}m"

    df = df.with_columns(
        ts_dep=pl.col("fecha_origen_recorrido").dt.truncate(freq),
        ts_arr=pl.col("fecha_destino_recorrido").dt.truncate(freq)
    )
    print("  Conversion and date normalization complete.")

    # ------------------------------------------------------------------
    # 1. Tabla de partidas (dep) con lags y proporción de género
    # ------------------------------------------------------------------
    print("\n[Step 1/9] Creating departures table with lags...")
    dep_path = os.path.join(checkpoint_dir, "dep.parquet")
    if os.path.exists(dep_path):
        print(f"  -> Loading from checkpoint: {dep_path}")
        dep = pl.read_parquet(dep_path)
    else:
        dep = (
            df.group_by(["id_estacion_origen", "ts_dep"])
              .agg([
                  pl.len().alias("dep_last_DT"),
                  (pl.col("genero") == "MALE").sum().alias("male_cnt")
              ])
              .rename({"id_estacion_origen": "station_id",
                       "ts_dep": "ts_start"})
              .sort(["station_id", "ts_start"])
              .with_columns(
                  share_male=pl.when(pl.col("dep_last_DT") > 0)
                               .then(pl.col("male_cnt") / pl.col("dep_last_DT"))
                               .otherwise(0.0)
              )
              .drop("male_cnt")
        )
        # lags 1…6
        dep = dep.with_columns([
            pl.col("dep_last_DT").shift(k).over("station_id").alias(f"dep_lag_{k}")
            for k in range(1, 7)
        ])
        print(f"  -> Saving checkpoint: {dep_path}")
        dep.write_parquet(dep_path)

    print(f"  -> Departures table shape: {dep.shape}")

    # ------------------------------------------------------------------
    # 2. Tabla de arribos desplazada → target
    # ------------------------------------------------------------------
    print("\n[Step 2/9] Creating shifted arrivals table for target...")
    arr_next_path = os.path.join(checkpoint_dir, "arr_next.parquet")
    if os.path.exists(arr_next_path):
        print(f"  -> Loading from checkpoint: {arr_next_path}")
        arr_next = pl.read_parquet(arr_next_path)
    else:
        arr = (
            df.group_by(["id_estacion_destino", "ts_arr"])
              .agg(pl.len().alias("arrivals"))
              .rename({"id_estacion_destino": "station_id",
                       "ts_arr": "ts_start"})
        )
        arr_next = (
            arr.with_columns(
                (pl.col("ts_start") - timedelta(minutes=dt_minutes)).alias("ts_start")
            ).rename({"arrivals": "y_arrivals_next_DT"})
        )
        print(f"  -> Saving checkpoint: {arr_next_path}")
        arr_next.write_parquet(arr_next_path)

    print(f"  -> Arrivals target table shape: {arr_next.shape}")

    # ------------------------------------------------------------------
    # 3. Tabla de clima (una fila por ts_start)
    # ------------------------------------------------------------------
    print("\n[Step 3/9] Creating weather table...")
    weather_cols = [c for c in df.columns if (
        c.endswith("_2m") or c in [
            "precipitation", "rain", "weather_code", "pressure_msl",
            "surface_pressure", "cloud_cover", "sunshine_duration",
            "is_day", "direct_radiation", "wind_speed_10m",
            "wind_direction_10m"
        ])]

    if weather_cols:
        weather = (
            df.select(["ts_dep"] + weather_cols)
              .unique(subset=["ts_dep"], keep="first")
              .rename({"ts_dep": "ts_start"})
        )
        print(f"  -> Weather table shape: {weather.shape}")
    else:
        weather = None
        print("  -> No weather table created.")

    # ------------------------------------------------------------------
    # 4. Metadatos de estaciones
    # ------------------------------------------------------------------
    print("\n[Step 4/9] Creating station metadata table...")
    stations = (
        df.select([
            "id_estacion_origen",
            "nombre_estacion_origen",
            "long_estacion_origen",
            "lat_estacion_origen"
        ]).unique("id_estacion_origen", keep="first")
          .rename({
              "id_estacion_origen": "station_id",
              "nombre_estacion_origen": "station_name",
              "long_estacion_origen": "lon",
              "lat_estacion_origen": "lat"
          })
    )
    print(f"  -> Stations metadata table shape: {stations.shape}")

    # ------------------------------------------------------------------
    # 5. Esqueleto estación × tiempo completo
    # ------------------------------------------------------------------
    print("\n[Step 5/9] Building full station x time skeleton...")
    df_feat_path = os.path.join(checkpoint_dir, "df_feat_skeleton.parquet")
    if os.path.exists(df_feat_path):
        print(f"  -> Loading from checkpoint: {df_feat_path}")
        df_feat = pl.read_parquet(df_feat_path)
    else:
        min_ts = df["ts_dep"].min()
        max_ts = df["ts_dep"].max() + timedelta(minutes=dt_minutes)
        skeleton = _build_skeleton(stations, min_ts, max_ts, freq)
        print(f"  -> Full data skeleton shape: {skeleton.shape}")

    # ------------------------------------------------------------------
    # 6. Joins en orden LEFT para no perder filas
    # ------------------------------------------------------------------
    print("\n[Step 6/9] Joining data onto skeleton...")
    df_feat = (skeleton
               .join(dep, on=["station_id", "ts_start"], how="left")
               .join(arr_next, on=["station_id", "ts_start"], how="left"))
    print(f"  -> Saving checkpoint: {df_feat_path}")
    df_feat.write_parquet(df_feat_path)

    print(f"  -> Shape after joining departures and arrivals: {df_feat.shape}")

    if weather is not None:
        df_feat = df_feat.join(weather, on="ts_start", how="left")
        print(f"  -> Shape after joining weather: {df_feat.shape}")

    # ------------------------------------------------------------------
    # 7. Features temporales
    # ------------------------------------------------------------------
    print("\n[Step 7/9] Engineering temporal features...")
    df_feat = df_feat.with_columns(
        hour=pl.col("ts_start").dt.hour(),
        dow=pl.col("ts_start").dt.weekday(),
        month=pl.col("ts_start").dt.month(),
    ).with_columns(
        sin_hour=(pl.col("hour") * 2 * math.pi / 24).sin(),
        cos_hour=(pl.col("hour") * 2 * math.pi / 24).cos(),
        is_weekend=(pl.col("dow") >= 6).cast(pl.Int8)
    )
    print(f"  -> Shape after adding temporal features: {df_feat.shape}")

    # ------------------------------------------------------------------
    # 8. Fusión con coords + demanda vecina
    # ------------------------------------------------------------------
    print("\n[Step 8/9] Calculating neighbor demand...")
    df_feat_final_path = os.path.join(checkpoint_dir, "df_feat_final.parquet")
    if os.path.exists(df_feat_final_path):
        print(f"  -> Loading from checkpoint: {df_feat_final_path}")
        df_feat = pl.read_parquet(df_feat_final_path)
    else:
        df_feat = df_feat.join(stations.select(["station_id", "lat", "lon"]),
                               on="station_id", how="left")

        if n_neighbors > 0 and stations.height > 0:
            coords_pd = stations.drop_nulls(subset=["lat", "lon"]).to_pandas()
            print(f"  -> Found {len(coords_pd)} stations with coordinates for neighbor calculation.")

            nbrs = NearestNeighbors(
                n_neighbors=min(n_neighbors + 1, len(coords_pd)),
                algorithm="ball_tree",
                metric=lambda a, b: _haversine_np(a[1], a[0], b[1], b[0])
            )
            nbrs.fit(coords_pd[["lat", "lon"]].to_numpy())
            _, ind = nbrs.kneighbors(coords_pd[["lat", "lon"]].to_numpy())

            neighbor_map = pl.DataFrame({
                "station_id": np.repeat(coords_pd["station_id"].values,
                                        ind.shape[1] - 1),
                "neighbor_id": coords_pd["station_id"].values[ind[:, 1:].ravel()]
            })

            neighbor_dep = (
                df_feat.select(["ts_start", "station_id", "dep_last_DT"])
                       .rename({"station_id": "neighbor_id",
                                "dep_last_DT": "neighbor_dep"})
            )

            near_sum = (
                neighbor_map.join(neighbor_dep, on="neighbor_id", how="inner")
                             .group_by(["station_id", "ts_start"])
                            .agg(pl.sum("neighbor_dep").alias("near_dep_sum_DT"))
            )

            df_feat = df_feat.join(near_sum, on=["station_id", "ts_start"], how="left")

        # ------------------------------------------------------------------
        # 9. Relleno de nulos y limpieza final
        # ------------------------------------------------------------------
        print("\n[Step 9/9] Filling nulls and final cleanup...")
        lag_cols = [c for c in df_feat.columns if c.startswith("dep_lag_")]
        fill_zero = ["dep_last_DT", "y_arrivals_next_DT", "near_dep_sum_DT"] + lag_cols
        existing_fill = [c for c in fill_zero if c in df_feat.columns]
        df_feat = df_feat.with_columns([pl.col(existing_fill).fill_null(0)])

        # clima: rellenar con forward-fill por columna (opcional), o media
        if weather_cols:
            df_feat = df_feat.sort("ts_start")
            df_feat = df_feat.with_columns([
                pl.col(c).fill_null(strategy="forward") for c in weather_cols
            ])

        # descartar columnas gps si no se usan en el modelo
        df_feat = df_feat.drop(["lon", "lat"], strict=False)

        print(f"  -> Saving final checkpoint: {df_feat_final_path}")
        df_feat.write_parquet(df_feat_final_path)

    print("✅  Feature-engineering finalizado -> shape:", df_feat.shape)
