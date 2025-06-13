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
    
    # Only process lat_estacion_destino which has concatenated coords in 2020-2021
    if 'lat_estacion_destino' in df.columns and df['lat_estacion_destino'].dtype == 'object':
        # Check if this column has concatenated coordinates (format: "lat,lon")
        sample_vals = df['lat_estacion_destino'].dropna().astype(str).head(100)
        has_concatenated = False
        
        if len(sample_vals) > 0:
            # Check if values contain comma (indicating concatenated coordinates)
            has_concatenated = sample_vals.str.contains(',', na=False).any()
        
        if has_concatenated:
            if verbose:
                print("Found concatenated coordinates, separating...")
            
            # Store original concatenated values before splitting
            concatenated_coords = df['lat_estacion_destino'].astype(str)
            
            # Extract latitude (first part before comma)
            df['lat_estacion_destino'] = concatenated_coords.str.split(',').str[0]
            
            # Extract longitude (second part after comma) and store in long_estacion_destino
            df['long_estacion_destino'] = concatenated_coords.str.split(',').str[1]
    
    # Convert coordinate columns to numeric
    coord_columns = ['lat_estacion_destino', 'long_estacion_destino', 
                    'lat_estacion_origen', 'long_estacion_origen']
    
    for col in coord_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


# =============================================================================
# MAIN PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_trips_by_year(df: pd.DataFrame, year: int, verbose: bool = False) -> pd.DataFrame:
    """Preprocess trips data based on the specific year's characteristics."""
    if year not in TRIPS_COLUMN_MAPPINGS:
        raise ValueError(f"Year {year} not supported")
    
    config = TRIPS_COLUMN_MAPPINGS[year]
    
    # Step 1: Handle gender column merging for specific years BEFORE normalization
    if year == 2021:  # Has both "género" and "Género"
        # Remove quotes from both columns first
        gender_col1 = 'género'
        gender_col2 = 'Género'
        
        # Check if these columns exist (with quotes)
        if f'"{gender_col1}"' in df.columns:
            gender_col1 = f'"{gender_col1}"'
        if f'"{gender_col2}"' in df.columns:
            gender_col2 = f'"{gender_col2}"'
            
        if gender_col1 in df.columns and gender_col2 in df.columns:
            # Merge the two gender columns before normalization
            df[gender_col1] = df[gender_col1].fillna(df[gender_col2])
            df = df.drop(columns=[gender_col2])
    
    # Step 2: Normalize column names
    df = normalize_column_names(df, config["normalize"])
    
    # Step 3: Handle BAEcobici suffixes
    if config["has_baecobici"]:
        baecobici_columns = ['id_recorrido', 'id_estacion_origen', 'id_estacion_destino', 'id_usuario']
        df = clean_baecobici_suffixes(df, baecobici_columns)
    
    # Step 4: Fix comma decimal separators
    if config["has_comma_decimals"]:
        df = fix_comma_decimals(df, ['duracion_recorrido'])
    
    # Step 5: Handle concatenated coordinates
    if config["has_concatenated_coords"]:
        df = separate_concatenated_coordinates(df, verbose)
    
    # Step 6: Remove unwanted columns that appear across all years
    columns_to_remove = ['Unnamed: 0']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df


def preprocess_users_by_year(df: pd.DataFrame, year: int, verbose: bool = False) -> pd.DataFrame:
    """Preprocess users data based on the specific year's characteristics."""
    if year not in USERS_COLUMN_MAPPINGS:
        raise ValueError(f"Year {year} not supported")
    
    config = USERS_COLUMN_MAPPINGS[year]
    
    # Step 1: Normalize column names
    df = normalize_column_names(df, config["normalize"])
    
    return df


def load_and_preprocess_csv(file_path: Path, data_type: str, verbose: bool = False) -> pd.DataFrame:
    """Load and preprocess a single CSV file."""
    year = extract_year_from_filename(file_path)
    
    # Read CSV
    df = pd.read_csv(file_path, low_memory=False, engine="c")
    
    # Preprocess based on data type and year
    if data_type == "trips":
        df = preprocess_trips_by_year(df, year, verbose)
    elif data_type == "users":
        df = preprocess_users_by_year(df, year, verbose)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return df


def load_csv_files(input_dir: Path, data_type: str, parse_dates: Optional[List[str]] = None, verbose: bool = False) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate them."""
    files = sorted(input_dir.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    
    dfs = []
    total_rows = 0
    
    for file_path in files:
        if verbose:
            print(f"Cargando: {file_path.name}")
        start_time = time.time()
        
        # Load and preprocess
        df = load_and_preprocess_csv(file_path, data_type, verbose)
        
        # Handle date parsing if specified
        if parse_dates:
            for date_col in parse_dates:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Basic memory optimization
        for col in df.select_dtypes(include=['object']).columns:
            if col not in (parse_dates or []):
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        
        dfs.append(df)
        total_rows += len(df)
        
        if verbose:
            load_time = time.time() - start_time
            print(f"   {len(df):,} filas cargadas en {load_time:.2f}s")
    
    if verbose:
        print(f"Concatenando {len(dfs)} archivos con {total_rows:,} filas totales...")
    concat_start = time.time()
    
    # Concatenate - all DataFrames should now have consistent column names
    result_df = pd.concat(dfs, ignore_index=True, copy=False)
    
    if verbose:
        concat_time = time.time() - concat_start
        print(f"   Concatenacion completada en {concat_time:.2f}s")
    
    return result_df


def load_users(input_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load all user CSV files into a single DataFrame."""
    if verbose:
        print("Cargando datos de usuarios...")

    
    users_df = load_csv_files(input_dir, "users", verbose=verbose)
    
    # Additional optimizations for users data
    if 'edad_usuario' in users_df.columns:
        users_df['edad_usuario'] = pd.to_numeric(users_df['edad_usuario'], errors='coerce')
    

    
    return users_df


def load_trips(input_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load all trip CSV files into a single DataFrame."""
    if verbose:
        print("Cargando datos de viajes...")

    
    # Date columns for parsing
    date_columns = ["fecha_origen_recorrido", "fecha_destino_recorrido"]
    
    trips_df = load_csv_files(input_dir, "trips", parse_dates=date_columns, verbose=verbose)
    

    
    return trips_df


def load_all_raw_data(raw_dir: Path, save_dir: Optional[Path] = None, verbose: bool = False, load_parquet: bool = False, save_parquet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all raw datasets (users and trips) from data/raw directory."""
    if verbose:
        print("INICIANDO CARGA OPTIMIZADA DE DATOS RAW")
        print("=" * 50)
    users_dir = raw_dir / "users"
    trips_dir = raw_dir / "trips"
    
    if not users_dir.exists():
        raise FileNotFoundError(f"Directorio de usuarios no encontrado: {users_dir}")
    if not trips_dir.exists():
        raise FileNotFoundError(f"Directorio de viajes no encontrado: {trips_dir}")
    
    # Try to load from parquet if requested
    if load_parquet and save_dir is not None:
        users_parquet = save_dir / "users.parquet"
        trips_parquet = save_dir / "trips.parquet"
        
        if users_parquet.exists() and trips_parquet.exists():
            if verbose:
                print("Loading from parquet files...")
            users_df = pd.read_parquet(users_parquet)
            trips_df = pd.read_parquet(trips_parquet)
            return users_df, trips_df
    
    users_df = load_users(users_dir, verbose)
    
    if verbose:
        print(f"Total usuarios: {len(users_df):,} registros")
        print(f"Columnas usuarios: {list(users_df.columns)}")
        print()
    
    trips_df = load_trips(trips_dir, verbose)
    
    if verbose:
        print(f"Total viajes: {len(trips_df):,} registros")
        print(f"Columnas viajes: {list(trips_df.columns)}")
        print()
    
    # process station IDs to extract first 3 digits
    if verbose:
        print("Procesando IDs de estacion (extrayendo primeros 3 digitos)...")
    trips_df = process_station_ids_to_3_digits(trips_df, verbose)
    
    # ensure user IDs are integers in both dataframes
    if verbose:
        print("Procesando IDs de usuario (convirtiendo a enteros)...")
    
    if 'id_usuario' in users_df.columns:
        users_df['id_usuario'] = users_df['id_usuario'].astype('Int64')
        if verbose:
            print(f"  users_df id_usuario type: {users_df['id_usuario'].dtype}")
    
    if 'id_usuario' in trips_df.columns:
        trips_df['id_usuario'] = trips_df['id_usuario'].astype('Int64')
        if verbose:
            print(f"  trips_df id_usuario type: {trips_df['id_usuario'].dtype}")
    
    if verbose:
        print("Procesamiento de IDs completado")
        print()
    
    # Save to parquet if requested
    if save_parquet and save_dir is not None:
        if verbose:
            print("Saving to parquet files...")
        save_dir.mkdir(parents=True, exist_ok=True)
        users_df.to_parquet(save_dir / "users.parquet")
        trips_df.to_parquet(save_dir / "trips.parquet")
    
    return users_df, trips_df
# =============================================================================
# UTILITY FUNCTIONS FOR STATION ID PROCESSING
# =============================================================================

def extract_first_3_digits_station_id(station_id):
    """
    Extract the first 3 digits from station_id
    
    Args:
        station_id: Station ID (can be int, float, or string)
        
    Returns:
        int: First 3 digits of the station ID, or original if less than 3 digits
    """
    if pd.isna(station_id):
        return station_id
    
    # convert to string and remove decimal points if float
    station_str = str(int(float(station_id)))
    
    # extract first 3 digits
    if len(station_str) >= 3:
        return int(station_str[:3])
    else:
        return int(station_str)

def process_station_ids_to_3_digits(df, verbose: bool = False):
    """
    Process both origin and destination station IDs to extract first 3 digits
    
    Args:
        df: DataFrame with station ID columns
        verbose: Print processing information
        
    Returns:
        DataFrame with processed station IDs as integers
    """
    df_processed = df.copy()
    
    # process origin station IDs
    if 'id_estacion_origen' in df_processed.columns:
        if verbose:
            print("Processing id_estacion_origen...")
        original_count = df_processed['id_estacion_origen'].nunique()
        df_processed['id_estacion_origen'] = df_processed['id_estacion_origen'].apply(extract_first_3_digits_station_id)
        # ensure the column is integer type
        df_processed['id_estacion_origen'] = df_processed['id_estacion_origen'].astype('Int64')  # nullable integer
        new_count = df_processed['id_estacion_origen'].nunique()
        if verbose:
            print(f"  Original unique origin stations: {original_count}")
            print(f"  New unique origin stations: {new_count}")
            print(f"  Data type: {df_processed['id_estacion_origen'].dtype}")
            if original_count != new_count:
                print(f"  Reduction: {original_count - new_count} stations")
    
    # process destination station IDs
    if 'id_estacion_destino' in df_processed.columns:
        if verbose:
            print("Processing id_estacion_destino...")
        original_count = df_processed['id_estacion_destino'].nunique()
        df_processed['id_estacion_destino'] = df_processed['id_estacion_destino'].apply(extract_first_3_digits_station_id)
        # ensure the column is integer type
        df_processed['id_estacion_destino'] = df_processed['id_estacion_destino'].astype('Int64')  # nullable integer
        new_count = df_processed['id_estacion_destino'].nunique()
        if verbose:
            print(f"  Original unique destination stations: {original_count}")
            print(f"  New unique destination stations: {new_count}")
            print(f"  Data type: {df_processed['id_estacion_destino'].dtype}")
            if original_count != new_count:
                print(f"  Reduction: {original_count - new_count} stations")
    
    return df_processed