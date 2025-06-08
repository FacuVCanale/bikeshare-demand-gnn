import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from typing import List, Optional


def load_csv_files(input_dir: Path, parse_dates: Optional[List[str]] = None, dayfirst: bool = True) -> pd.DataFrame:
    """
    OPTIMIZED function to load all CSV files from a directory into a single DataFrame.
    
    Optimizations:
    - Efficient data type inference and optimization
    - Better memory management for large files
    - Optimized pandas reading parameters
    - Progress tracking for large datasets
    """
    files = sorted(input_dir.glob('*.csv'))
    if not files:
        raise FileNotFoundError("No se encontraron CSV en " + str(input_dir))
    
    dfs = []
    total_rows = 0
    
    for f in files:
        print(f"Cargando: {f.name}")
        start_time = time.time()
        
        # Optimized reading parameters for large datasets
        read_kwargs = {
            "low_memory": False,
            "engine": "c",  # Use C engine for speed
            "memory_map": True,  # Memory mapping for large files
        }
        
        if parse_dates:
            # Check which date columns exist without reading full file
            sample_df = pd.read_csv(f, nrows=0)
            existing_date_cols = [col for col in parse_dates if col in sample_df.columns]
            if existing_date_cols:
                        read_kwargs.update({
            "parse_dates": existing_date_cols, 
            "dayfirst": dayfirst,
            "date_format": "ISO8601"  # Faster date parsing
        })
        
        # Read the CSV with optimized parameters
        df = pd.read_csv(f, **read_kwargs)
        
        # Basic memory optimization during load
        for col in df.select_dtypes(include=['object']).columns:
            if col not in (existing_date_cols if parse_dates else []):
                # Convert object columns to category if they have low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        
        dfs.append(df)
        total_rows += len(df)
        
        load_time = time.time() - start_time
        print(f"   ✓ {len(df):,} filas cargadas en {load_time:.2f}s")
    
    print(f"🔗 Concatenando {len(dfs)} archivos con {total_rows:,} filas totales...")
    concat_start = time.time()
    
    # Efficient concatenation with optimized parameters
    result_df = pd.concat(dfs, ignore_index=True, copy=False)
    
    concat_time = time.time() - concat_start
    print(f"   ✓ Concatenación completada en {concat_time:.2f}s")
    
    return result_df


def load_users(input_dir: Path) -> pd.DataFrame:
    """
    OPTIMIZED function to load all user CSV files into a single DataFrame.
    
    Optimizations:
    - Uses optimized CSV loading
    - Better memory management for user data
    """
    print("👥 Cargando datos de usuarios...")
    start_time = time.time()
    
    users_df = load_csv_files(input_dir)
    
    # Additional optimizations for users data
    if 'edad_usuario' in users_df.columns:
        users_df['edad_usuario'] = pd.to_numeric(users_df['edad_usuario'], errors='coerce', downcast='integer')
    
    load_time = time.time() - start_time
    print(f"   ⏱️ Usuarios cargados en {load_time:.2f} segundos")
    
    return users_df


def load_trips(input_dir: Path) -> pd.DataFrame:
    """
    OPTIMIZED function to load all trip CSV files into a single DataFrame.
    
    Optimizations:
    - Uses optimized CSV loading with proper date parsing
    - Memory-efficient handling of large trip datasets
    - Optimized data types for trip-specific columns
    """
    print("🚲 Cargando datos de viajes...")
    start_time = time.time()
    
    # Date columns that might exist in trip files
    date_columns = ["fecha_origen", "fecha_destino", "fecha_origen_recorrido", "fecha_destino_recorrido"]
    
    # Use dayfirst=False because dates are in ISO format (YYYY-MM-DD)
    trips_df = load_csv_files(input_dir, parse_dates=date_columns, dayfirst=False)
    
    # Additional optimizations for trips data
    # Optimize numeric columns that are commonly integers
    int_columns = ['id_estacion_origen', 'id_estacion_destino', 'id_usuario', 'duracion_recorrido']
    for col in int_columns:
        if col in trips_df.columns:
            trips_df[col] = pd.to_numeric(trips_df[col], errors='coerce', downcast='integer')
    
    # Optimize coordinate columns
    float_columns = ['lat_estacion_origen', 'long_estacion_origen', 'lat_estacion_destino', 'long_estacion_destino']
    for col in float_columns:
        if col in trips_df.columns:
            trips_df[col] = pd.to_numeric(trips_df[col], errors='coerce', downcast='float')
    
    load_time = time.time() - start_time
    print(f"   ⏱️ Viajes cargados en {load_time:.2f} segundos")
    
    return trips_df


def load_all_raw_data(raw_dir: Path, save_dir: Optional[Path] = None, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    OPTIMIZED function to load all raw datasets (users and trips) from data/raw directory.
    
    Major optimizations:
    - Parallel-like loading with progress tracking
    - Memory-optimized data type handling
    - Better progress reporting and timing
    - Efficient memory usage during load process
    """
    print("🚀 INICIANDO CARGA OPTIMIZADA DE DATOS RAW")
    print("=" * 50)
    total_start = time.time()
    
    users_dir = raw_dir / "users"
    trips_dir = raw_dir / "trips"
    
    # Verify directories exist
    if not users_dir.exists():
        raise FileNotFoundError(f"Directorio de usuarios no encontrado: {users_dir}")
    if not trips_dir.exists():
        raise FileNotFoundError(f"Directorio de viajes no encontrado: {trips_dir}")
    
    # Load users with optimized function
    if verbose:
        print("=== CARGANDO DATASETS DE USUARIOS ===")
    
    users_df = load_users(users_dir)
    users_memory = users_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"✅ Total usuarios: {len(users_df):,} registros")
    print(f"📊 Columnas usuarios: {list(users_df.columns)}")
    print(f"💾 Memoria usuarios: {users_memory:.1f} MB")
    print()
    
    # Load trips with optimized function
    if verbose:
        print("=== CARGANDO DATASETS DE VIAJES ===")
    
    trips_df = load_trips(trips_dir)
    trips_memory = trips_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"✅ Total viajes: {len(trips_df):,} registros")
    print(f"📊 Columnas viajes: {list(trips_df.columns)}")
    print(f"💾 Memoria viajes: {trips_memory:.1f} MB")
    print()
    
    total_time = time.time() - total_start
    total_memory = users_memory + trips_memory
    
    print(f"🎉 CARGA OPTIMIZADA COMPLETADA!")
    print(f"⏱️ Tiempo total de carga: {total_time:.2f} segundos")
    print(f"💾 Memoria total utilizada: {total_memory:.1f} MB")
    print(f"📈 Velocidad promedio: {(len(users_df) + len(trips_df)) / total_time:,.0f} filas/segundo")
    
    return users_df, trips_df


def compute_counts(df: pd.DataFrame, dt_minutes: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    OPTIMIZED function to compute dispatch and arrival counts by station.
    
    Optimizations:
    - More efficient groupby operations
    - Better memory management
    - Vectorized time window calculations
    """
    print(f"📊 Computando conteos por ventanas de {dt_minutes} minutos...")
    start_time = time.time()
    
    # Work with copy to avoid modifying original
    df_work = df.copy()
    
    # Vectorized time window calculation - more efficient than individual operations
    df_work["window_dispatch"] = df_work["fecha_origen"].dt.floor(f"{dt_minutes}min")
    df_work["window_arrival"] = df_work["fecha_destino"].dt.floor(f"{dt_minutes}min")

    # Optimized groupby with efficient aggregation
    dispatch = (
        df_work.groupby(["window_dispatch", "estacion_origen_id"], sort=False)
        .size()
        .reset_index(name="dispatch_count")
        .rename(columns={"window_dispatch": "timestamp", "estacion_origen_id": "station_id"})
    )

    arrival = (
        df_work.groupby(["window_arrival", "estacion_destino_id"], sort=False)
        .size()
        .reset_index(name="arrival_count")
        .rename(columns={"window_arrival": "timestamp", "estacion_destino_id": "station_id"})
    )
    
    compute_time = time.time() - start_time
    print(f"   ✓ Conteos computados en {compute_time:.2f}s")
    print(f"   📈 Dispatch: {len(dispatch):,} registros, Arrival: {len(arrival):,} registros")

    return dispatch, arrival


def main(input_dir: str, output_dir: str, dt_minutes: int = 30) -> None:
    """
    OPTIMIZED main function with better error handling and progress tracking.
    """
    print("🏃 Ejecutando pipeline principal optimizado...")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load trips with optimized function
    trips = load_trips(input_path)
    
    # Compute counts with optimized function
    dispatch, arrival = compute_counts(trips, dt_minutes=dt_minutes)

    # Save results with progress tracking
    print("💾 Guardando resultados...")
    save_start = time.time()
    
    dispatch.to_parquet(output_path / "dispatch_counts.parquet", compression='snappy')
    arrival.to_parquet(output_path / "arrival_counts.parquet", compression='snappy')
    
    save_time = time.time() - save_start
    print(f"   ✓ Archivos guardados en {save_time:.2f}s")
    print(f"📁 Ubicación: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline optimizado de procesamiento de EcoBici")
    parser.add_argument("--input_dir", type=str, required=True, help="Directorio con CSV crudos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio destino para parquet")
    parser.add_argument("--dt_minutes", type=int, default=30, help="Ventana de tiempo en minutos")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, dt_minutes=args.dt_minutes)
