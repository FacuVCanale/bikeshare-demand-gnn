"""
Data loading and preprocessing utilities for EcoBici-AI.

This module contains functions for:
- Loading CSV files from different years with proper column mappings
- Normalizing column names and data types
- Basic data preprocessing and cleaning
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from typing import List, Optional, Dict, Any


# =============================================================================
# DATA STRUCTURE MAPPINGS BY YEAR
# =============================================================================

TRIPS_COLUMN_MAPPINGS = {
    2020: {
        "structure": ["", "Id_recorrido", "duracion_recorrido", "fecha_origen_recorrido", 
                     "id_estacion_origen", "nombre_estacion_origen", "direccion_estacion_origen",
                     "long_estacion_origen", "lat_estacion_origen", "fecha_destino_recorrido",
                     "id_estacion_destino", "nombre_estacion_destino", "direccion_estacion_destino",
                     "long_estacion_destino", "lat_estacion_destino", "id_usuario", "modelo_bicicleta", "género"],
        "normalize": {
            "": None,  # Drop
            "Id_recorrido": "id_recorrido",
            "género": "genero",
        },
        "has_baecobici": True,
        "has_comma_decimals": True,
        "has_concatenated_coords": True  # lat_estacion_destino has "lat,lon" format
    },
    2021: {
        "structure": ["", "Id_recorrido", "duracion_recorrido", "fecha_origen_recorrido",
                     "id_estacion_origen", "nombre_estacion_origen", "direccion_estacion_origen",
                     "long_estacion_origen", "lat_estacion_origen", "fecha_destino_recorrido",
                     "id_estacion_destino", "nombre_estacion_destino", "direccion_estacion_destino",
                     "long_estacion_destino", "lat_estacion_destino", "id_usuario", "modelo_bicicleta", 
                     "género", "Género"],
        "normalize": {
            "": None,  # Drop
            "Id_recorrido": "id_recorrido",
            "género": "genero",
            "Género": None,  # Drop this one - will merge it later
        },
        "has_baecobici": True,
        "has_comma_decimals": True,
        "has_concatenated_coords": True  # lat_estacion_destino has "lat,lon" format
    },
    2022: {
        "structure": ["", "X", "Id_recorrido", "duracion_recorrido", "fecha_origen_recorrido",
                     "id_estacion_origen", "nombre_estacion_origen", "direccion_estacion_origen",
                     "long_estacion_origen", "lat_estacion_origen", "fecha_destino_recorrido",
                     "id_estacion_destino", "nombre_estacion_destino", "direccion_estacion_destino",
                     "long_estacion_destino", "lat_estacion_destino", "id_usuario", "modelo_bicicleta", "Género"],
        "normalize": {
            "": None,  # Drop
            "X": None,  # Drop
            "Id_recorrido": "id_recorrido", 
            "Género": "genero",
        },
        "has_baecobici": True,
        "has_comma_decimals": True,
        "has_concatenated_coords": False  # No concatenated coordinates in 2022
    },
    2023: {
        "structure": ["", "Id_recorrido", "duracion_recorrido", "fecha_origen_recorrido",
                     "id_estacion_origen", "nombre_estacion_origen", "direccion_estacion_origen",
                     "long_estacion_origen", "lat_estacion_origen", "fecha_destino_recorrido",
                     "id_estacion_destino", "nombre_estacion_destino", "direccion_estacion_destino",
                     "long_estacion_destino", "lat_estacion_destino", "id_usuario", "modelo_bicicleta", "género"],
        "normalize": {
            "": None,  # Drop
            "Id_recorrido": "id_recorrido",
            "género": "genero",
        },
        "has_baecobici": True,
        "has_comma_decimals": True,
        "has_concatenated_coords": False  # No concatenated coordinates in 2023
    },
    2024: {
        "structure": ["id_recorrido", "duracion_recorrido", "fecha_origen_recorrido",
                     "id_estacion_origen", "nombre_estacion_origen", "direccion_estacion_origen",
                     "long_estacion_origen", "lat_estacion_origen", "fecha_destino_recorrido",
                     "id_estacion_destino", "nombre_estacion_destino", "direccion_estacion_destino",
                     "long_estacion_destino", "lat_estacion_destino", "id_usuario", "modelo_bicicleta", "genero"],
        "normalize": {},  # Already normalized
        "has_baecobici": False,
        "has_comma_decimals": False,
        "has_concatenated_coords": False
    }
}

USERS_COLUMN_MAPPINGS = {
    2020: {
        "structure": ["ID_usuario", "genero_usuario", "edad_usuario", "fecha_alta", "hora_alta", "Customer.Has.Dni..Yes...No."],
        "normalize": {
            "ID_usuario": "id_usuario",
            "Customer.Has.Dni..Yes...No.": None  # Drop
        }
    },
    2021: {
        "structure": ["ID_usuario", "genero_usuario", "edad_usuario", "fecha_alta", "hora_alta", "Customer.Has.Dni..Yes...No."],
        "normalize": {
            "ID_usuario": "id_usuario",
            "Customer.Has.Dni..Yes...No.": None  # Drop
        }
    },
    2022: {
        "structure": ["ID_usuario", "genero_usuario", "edad_usuario", "fecha_alta", "hora_alta", "Customer.Has.Dni..Yes...No."],
        "normalize": {
            "ID_usuario": "id_usuario",
            "Customer.Has.Dni..Yes...No.": None  # Drop
        }
    },
    2023: {
        "structure": ["ID_usuario", "genero_usuario", "edad_usuario", "fecha_alta", "hora_alta", "Customer.Has.Dni..Yes...No."],
        "normalize": {
            "ID_usuario": "id_usuario",
            "Customer.Has.Dni..Yes...No.": None  # Drop
        }
    },
    2024: {
        "structure": ["id_usuario", "genero_usuario", "edad_usuario", "fecha_alta", "hora_alta"],
        "normalize": {}  # Already normalized
    }
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_year_from_filename(file_path: Path) -> int:
    """Extract year from filename safely."""
    filename = str(file_path.name)
    for year in [2020, 2021, 2022, 2023, 2024]:
        if str(year) in filename:
            return year
    raise ValueError(f"Could not extract year from filename: {filename}")


def normalize_column_names(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Normalize column names using the provided mapping."""
    df_normalized = df.copy()
    
    # Remove quotes from all column names first
    df_normalized.columns = df_normalized.columns.str.strip('"')
    
    # Apply mappings
    columns_to_drop = []
    for old_name, new_name in mapping.items():
        if old_name in df_normalized.columns:
            if new_name is None:
                columns_to_drop.append(old_name)
            else:
                df_normalized = df_normalized.rename(columns={old_name: new_name})
    
    # Drop unwanted columns
    if columns_to_drop:
        df_normalized = df_normalized.drop(columns=columns_to_drop, errors='ignore')
    
    return df_normalized


def clean_baecobici_suffixes(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove BAEcobici suffixes from specified columns."""
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            # Check if any values have the suffix
            sample_vals = df[col].dropna().astype(str).head(100)
            if len(sample_vals) > 0:
                # Convert to boolean explicitly to avoid ambiguity
                has_suffix_series = sample_vals.str.contains('BAEcobici', na=False)
                has_suffix = bool(has_suffix_series.any())
                if has_suffix:
                    df[col] = df[col].astype(str).str.replace('BAEcobici', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fix_comma_decimals(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Fix comma separators in duration columns for EcoBici data.
    
    In EcoBici data (2020-2023), commas in duracion_recorrido are thousands separators,
    not decimal separators. For example:
    - "1,582" means 1582 seconds (not 1.582 seconds)
    - "2,026" means 2026 seconds (not 2.026 seconds)
    
    This was a critical bug causing duration values to be divided by ~1000.
    """
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            # Remove commas (thousands separators) from duration values
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def separate_concatenated_coordinates(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Separate concatenated coordinates in lat/long columns."""
    
    coord_columns = ['lat_estacion_origen', 'lat_estacion_destino']
    
    for col in coord_columns:
        if col in df.columns:
            # create corresponding longitude column name
            lon_col = col.replace('lat_', 'long_')
            
            # check if any values in lat column contain commas (concatenated coords)
            lat_series = df[col].astype(str)
            has_comma = lat_series.str.contains(',', na=False).any()
            
            if has_comma:
                if verbose:
                    print(f"separating concatenated coordinates in {col}")
                
                # Split concatenated coordinates
                coords_split = lat_series.str.split(',', expand=True)
                
                # Update lat and lon columns
                df[col] = pd.to_numeric(coords_split[0], errors='coerce')
                if coords_split.shape[1] > 1:
                    df[lon_col] = pd.to_numeric(coords_split[1], errors='coerce')
    
    return df


def load_and_preprocess_csv(file_path: Path, data_type: str, verbose: bool = False) -> pd.DataFrame:
    """Load and preprocess a single CSV file based on its year and data type."""
    
    year = extract_year_from_filename(file_path)
    
    if verbose:
        print(f"loading {data_type} data for {year}: {file_path}")
    
    # load csv with specific settings based on year
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"error loading {file_path}: {e}")
        return pd.DataFrame()
    
    if data_type == "trips":
        df = preprocess_trips_by_year(df, year, verbose)
    elif data_type == "users":
        df = preprocess_users_by_year(df, year, verbose)
    
    return df


def preprocess_trips_by_year(df: pd.DataFrame, year: int, verbose: bool = False) -> pd.DataFrame:
    """Preprocess trips data based on year-specific requirements."""
    
    if year not in TRIPS_COLUMN_MAPPINGS:
        raise ValueError(f"unsupported year: {year}")
    
    config = TRIPS_COLUMN_MAPPINGS[year]
    
    # normalize column names
    df = normalize_column_names(df, config["normalize"])
    
    # clean BAEcobici suffixes if present
    if config["has_baecobici"]:
        cols_to_clean = ['id_estacion_origen', 'id_estacion_destino']
        df = clean_baecobici_suffixes(df, cols_to_clean)
    
    # fix comma decimals if present
    if config["has_comma_decimals"]:
        df = fix_comma_decimals(df, ['duracion_recorrido'])
    
    # separate concatenated coordinates if present
    if config["has_concatenated_coords"]:
        df = separate_concatenated_coordinates(df, verbose)
    
    if verbose:
        print(f"  processed {len(df)} trips for {year}")
    
    return df


def preprocess_users_by_year(df: pd.DataFrame, year: int, verbose: bool = False) -> pd.DataFrame:
    """Preprocess users data based on year-specific requirements."""
    
    if year not in USERS_COLUMN_MAPPINGS:
        raise ValueError(f"unsupported year: {year}")
    
    config = USERS_COLUMN_MAPPINGS[year]
    
    # normalize column names
    df = normalize_column_names(df, config["normalize"])
    
    if verbose:
        print(f"  processed {len(df)} users for {year}")
    
    return df


def load_csv_files(input_dir: Path, data_type: str, parse_dates: Optional[List[str]] = None, verbose: bool = False) -> pd.DataFrame:
    """Load and combine multiple CSV files of the same type."""
    
    if verbose:
        print(f"\nloading {data_type} files from: {input_dir}")
    
    # find all relevant CSV files
    pattern = f"{data_type}_*.csv"
    csv_files = list(input_dir.glob(pattern))
    
    if not csv_files:
        print(f"no {data_type} files found matching pattern: {pattern}")
        return pd.DataFrame()
    
    if verbose:
        print(f"found {len(csv_files)} {data_type} files")
        for file in sorted(csv_files):
            print(f"  - {file.name}")
    
    # load and combine all files
    all_dfs = []
    total_start_time = time.time()
    
    for file_path in sorted(csv_files):
        start_time = time.time()
        df = load_and_preprocess_csv(file_path, data_type, verbose)
        
        if not df.empty:
            all_dfs.append(df)
            if verbose:
                load_time = time.time() - start_time
                print(f"  loaded {len(df):,} rows in {load_time:.2f}s")
    
    if not all_dfs:
        print(f"no valid {data_type} data found")
        return pd.DataFrame()
    
    # combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # parse dates if specified
    if parse_dates:
        for date_col in parse_dates:
            if date_col in combined_df.columns:
                combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce')
    
    total_time = time.time() - total_start_time
    
    if verbose:
        print(f"\ncombined {data_type} dataset:")
        print(f"  total rows: {len(combined_df):,}")
        print(f"  total columns: {len(combined_df.columns)}")
        print(f"  total load time: {total_time:.2f}s")
        print(f"  columns: {list(combined_df.columns)}")
    
    return combined_df


def load_users(input_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load and combine all users CSV files."""
    return load_csv_files(
        input_dir=input_dir,
        data_type="users",
        parse_dates=['fecha_alta'],
        verbose=verbose
    )


def load_trips(input_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load and combine all trips CSV files."""
    return load_csv_files(
        input_dir=input_dir,
        data_type="trips", 
        parse_dates=['fecha_origen_recorrido', 'fecha_destino_recorrido'],
        verbose=verbose
    )


def load_all_raw_data(raw_dir: Path, save_dir: Optional[Path] = None, verbose: bool = False, 
                     load_parquet: bool = False, save_parquet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all raw CSV data or existing parquet files.
    
    Args:
        raw_dir: Directory containing raw CSV files
        save_dir: Directory to save parquet files (optional)
        verbose: Whether to print detailed information
        load_parquet: Whether to try loading existing parquet files first
        save_parquet: Whether to save data as parquet files
        
    Returns:
        Tuple of (users_df, trips_df)
    """
    
    if save_dir is None:
        save_dir = raw_dir.parent / "processed"
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    users_parquet_path = save_dir / "users_raw.parquet"
    trips_parquet_path = save_dir / "trips_raw.parquet"
    
    # try to load existing parquet files first
    if load_parquet and users_parquet_path.exists() and trips_parquet_path.exists():
        if verbose:
            print("loading existing parquet files...")
        try:
            users_df = pd.read_parquet(users_parquet_path)
            trips_df = pd.read_parquet(trips_parquet_path)
            
            if verbose:
                print(f"loaded users: {len(users_df):,} rows")
                print(f"loaded trips: {len(trips_df):,} rows")
            
            return users_df, trips_df
            
        except Exception as e:
            if verbose:
                print(f"failed to load parquet files: {e}")
                print("falling back to CSV loading...")
    
    # load from CSV files
    if verbose:
        print("loading raw CSV data...")
        print("="*50)
    
    # load users and trips
    users_df = load_users(raw_dir / "users", verbose=verbose)
    trips_df = load_trips(raw_dir / "trips", verbose=verbose)
    
    if verbose:
        print(f"\nfinal dataset summary:")
        print(f"  users: {len(users_df):,} rows, {len(users_df.columns)} columns")
        print(f"  trips: {len(trips_df):,} rows, {len(trips_df.columns)} columns")
    
    # save as parquet if requested
    if save_parquet and not users_df.empty and not trips_df.empty:
        if verbose:
            print(f"\nsaving parquet files to: {save_dir}")
        
        users_df.to_parquet(users_parquet_path, index=False)
        trips_df.to_parquet(trips_parquet_path, index=False)
        
        if verbose:
            users_size = users_parquet_path.stat().st_size / (1024*1024)
            trips_size = trips_parquet_path.stat().st_size / (1024*1024)
            print(f"  users.parquet: {users_size:.1f} MB")
            print(f"  trips.parquet: {trips_size:.1f} MB")
    
    return users_df, trips_df 