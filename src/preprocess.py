import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re


def analyze_users_for_visualization(users_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze users data for visualization purposes.
    
    Args:
        users_df: Users DataFrame
        
    Returns:
        Dict with processed data for visualization
    """
    processed_data = {}
    
    # Age analysis
    if 'edad_usuario' in users_df.columns:
        users_clean = users_df.copy()
        users_clean['edad_usuario'] = pd.to_numeric(users_clean['edad_usuario'], errors='coerce')
        edad_valid = users_clean['edad_usuario'].dropna()
        edad_valid = edad_valid[(edad_valid >= 0) & (edad_valid <= 150)]
        processed_data['edad_valid'] = edad_valid
        processed_data['edad_promedio'] = edad_valid.mean()
    
    # Gender analysis
    if 'genero_usuario' in users_df.columns:
        processed_data['genero_counts'] = users_df['genero_usuario'].value_counts()
    
    # Registration date analysis
    if 'fecha_alta' in users_df.columns:
        users_temp = users_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(users_temp['fecha_alta']):
            users_temp['fecha_alta'] = pd.to_datetime(users_temp['fecha_alta'], errors='coerce')
        
        processed_data['registros_por_año'] = users_temp['fecha_alta'].dt.year.value_counts().sort_index()
        processed_data['registros_por_mes'] = users_temp['fecha_alta'].dt.month.value_counts().sort_index()
    
    return processed_data


def analyze_trips_for_visualization(trips_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trips data for visualization purposes.
    
    Args:
        trips_df: Trips DataFrame
        
    Returns:
        Dict with processed data for visualization
    """
    processed_data = {}
    
    # Prepare temporal data
    trips_temp = trips_df.copy()
    fecha_col = 'fecha_origen_recorrido' if 'fecha_origen_recorrido' in trips_temp.columns else 'fecha_origen'
    
    if not pd.api.types.is_datetime64_any_dtype(trips_temp[fecha_col]):
        trips_temp[fecha_col] = pd.to_datetime(trips_temp[fecha_col], errors='coerce')
    
    trips_temp = trips_temp.dropna(subset=[fecha_col])
    processed_data['trips_clean'] = trips_temp
    processed_data['fecha_col'] = fecha_col
    
    # Temporal analysis
    processed_data['viajes_por_hora'] = trips_temp[fecha_col].dt.hour.value_counts().sort_index()
    processed_data['viajes_por_dia'] = trips_temp[fecha_col].dt.dayofweek.value_counts().sort_index()
    processed_data['viajes_por_mes'] = trips_temp[fecha_col].dt.month.value_counts().sort_index()
    processed_data['viajes_por_año'] = trips_temp[fecha_col].dt.year.value_counts().sort_index()
    
    # Station analysis
    if 'id_estacion_origen' in trips_temp.columns:
        processed_data['top_origen'] = trips_temp['id_estacion_origen'].value_counts().head(15)
        processed_data['viajes_por_estacion'] = trips_temp['id_estacion_origen'].value_counts()
    
    if 'id_estacion_destino' in trips_temp.columns:
        processed_data['top_destino'] = trips_temp['id_estacion_destino'].value_counts().head(15)
    
    # Duration analysis
    if 'duracion_recorrido' in trips_temp.columns:
        trips_temp['duracion_num'] = pd.to_numeric(
            trips_temp['duracion_recorrido'].astype(str).str.replace(',', ''), 
            errors='coerce'
        )
        duracion_valid = trips_temp['duracion_num'].dropna()
        duracion_valid = duracion_valid[(duracion_valid > 120) & (duracion_valid < 7200)]  # filter: >2 min, <2 hours
        processed_data['duracion_valid'] = duracion_valid
        processed_data['duracion_promedio'] = duracion_valid.mean()
    
    # Heatmap data
    processed_data['hora'] = trips_temp[fecha_col].dt.hour
    processed_data['dia_semana'] = trips_temp[fecha_col].dt.dayofweek
    processed_data['activity_matrix'] = trips_temp.groupby([
        trips_temp[fecha_col].dt.dayofweek, 
        trips_temp[fecha_col].dt.hour
    ]).size().unstack(fill_value=0)
    
    return processed_data


def clean_dataset_comprehensive(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Simplified comprehensive dataset cleaning since most normalization happens during loading.
    
    Operations performed:
    - Final cleanup and validation
    - Handle edge cases missed during loading
    
    Args:
        df: Input DataFrame to preprocess
        verbose: Whether to print progress information
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if verbose:
        print(f"Final comprehensive cleaning for {len(df):,} rows...")
    
    # Basic cleanup - ensure no completely empty rows or columns
    original_shape = df.shape
    df = df.dropna(how='all')  # Drop rows where all values are NaN
    df = df.loc[:, ~df.isnull().all()]  # Drop columns where all values are NaN
    
    if verbose and df.shape != original_shape:
        print(f"Removed empty rows/columns: {original_shape} → {df.shape}")
    
    if verbose:
        print(f"Final cleaning completed for {len(df):,} rows!")
    
    return df





def clean_coordinate_columns(df: pd.DataFrame, coordinate_cols: List[str], verbose: bool = False) -> pd.DataFrame:
    """
    Clean and validate coordinate columns.
    
    Args:
        df: DataFrame containing coordinate columns
        coordinate_cols: List of coordinate column names to clean
        verbose: Whether to print detailed progress
        
    Returns:
        pd.DataFrame: DataFrame with cleaned coordinates
    """
    if verbose:
        print("Batch coordinate cleaning")
        print(f"Processing {len(df):,} rows...")
    
    for col in coordinate_cols:
        if col not in df.columns:
            continue
            
        if verbose:
            original_nulls = df[col].isnull().sum()
            print(f"Cleaning coordinate column: {col}")
            print(f"Original nulls: {original_nulls:,}")
        
        # Clean coordinate values
        if df[col].dtype == 'object':
            # Handle string coordinates
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate coordinate ranges (rough validation for Buenos Aires area)
        if 'lat_' in col:
            # Latitude should be around -34.x for Buenos Aires
            mask = (df[col] < -35.0) | (df[col] > -33.0)
            if mask.sum() > 0:
                df.loc[mask, col] = np.nan
        elif 'long_' in col:
            # Longitude should be around -58.x for Buenos Aires
            mask = (df[col] < -59.0) | (df[col] > -57.0)
            if mask.sum() > 0:
                df.loc[mask, col] = np.nan
        
        if verbose:
            final_nulls = df[col].isnull().sum()
            valid_coords = len(df) - final_nulls
            sample_values = df[col].dropna().head(5).tolist()
            
            print(f"Valid coordinates: {valid_coords:,}")
            print(f"Final nulls: {final_nulls:,}")
            print(f"Sample cleaned values: {sample_values}")
    
    if verbose:
        print("Batch coordinate cleaning completed!")
    
    return df


def optimize_memory_usage(df: pd.DataFrame, preserve_datetime_cols: List[str] = None, verbose: bool = False) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types and using categories.
    
    Args:
        df: DataFrame to optimize
        preserve_datetime_cols: List of datetime columns to preserve
        verbose: Whether to print detailed progress
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    if preserve_datetime_cols is None:
        preserve_datetime_cols = []

    if verbose:
        print(f"Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Preserve datetime columns
    for col in preserve_datetime_cols:
        if col in df.columns:
            if verbose:
                print(f"Preserving datetime column: {col}")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in preserve_datetime_cols:
            original_dtype = df[col].dtype
            
            if 'int' in str(original_dtype):
                # Try to downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif 'float' in str(original_dtype):
                # Downcast floats to float32 if possible
                if df[col].dtype == 'float64':
                    # Check if we can safely convert to float32
                    max_val = df[col].max()
                    min_val = df[col].min()
                    if not pd.isna(max_val) and not pd.isna(min_val):
                        if (max_val < np.finfo(np.float32).max and 
                            min_val > np.finfo(np.float32).min):
                            df[col] = df[col].astype('float32')
            
            if verbose and df[col].dtype != original_dtype:
                if 'id_recorrido' in col:
                    print(f"{col}: regular {original_dtype}")
                else:
                    print(f"{col}: {original_dtype} → {df[col].dtype}")
    
    # Optimize object columns to categories
    for col in df.select_dtypes(include=['object']).columns:
        if col not in preserve_datetime_cols:
            # Convert to category if it has reasonable cardinality
            nunique = df[col].nunique()
            total_len = len(df)
            
            if nunique / total_len < 0.5:  # Less than 50% unique values

                df[col] = df[col].astype('category')
              
                
                if verbose:
                    print(f"{col}: object → category")

    
    return df


def preprocess_data(users_df: pd.DataFrame, trips_df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline for users and trips data.
    
    Args:
        users_df: Raw users DataFrame
        trips_df: Raw trips DataFrame  
        verbose: Whether to print detailed progress
        
    Returns:
        Tuple of preprocessed (users_df, trips_df)
    """
    if verbose:
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Initial state:")
        print(f"  - Users: {users_df.shape[0]:,} rows, {users_df.shape[1]} columns")
        print(f"  - Trips: {trips_df.shape[0]:,} rows, {trips_df.shape[1]} columns")
    
    # Step 1: Final comprehensive cleaning
    if verbose:
        print(f"\n1. Final comprehensive cleaning...")
    trips_df = clean_dataset_comprehensive(trips_df, verbose=verbose)
    users_df = clean_dataset_comprehensive(users_df, verbose=verbose)
    
    # Step 2: Coordinate cleaning
    if verbose:
        print(f"\n2. Geographic coordinates cleaning...")
    coordinate_cols = ['lat_estacion_destino', 'long_estacion_destino', 
                      'lat_estacion_origen', 'long_estacion_origen']
    trips_df = clean_coordinate_columns(trips_df, coordinate_cols, verbose=verbose)
    
    # Step 3: Date preprocessing (preserve datetime columns)
    if verbose:
        print(f"\n3. Date preprocessing validation...")
    
    # Verify date columns are properly formatted
    datetime_cols_trips = ['fecha_origen_recorrido', 'fecha_destino_recorrido']
    datetime_cols_users = ['fecha_alta']  # if it exists
    
    for col in datetime_cols_trips:
        if col in trips_df.columns and trips_df[col].dtype != 'datetime64[ns]':
            if verbose:
                print(f"Converting {col} to datetime...")
            trips_df[col] = pd.to_datetime(trips_df[col], errors='coerce')
    
    for col in datetime_cols_users:
        if col in users_df.columns and users_df[col].dtype != 'datetime64[ns]':
            if verbose:
                print(f"Converting {col} to datetime...")
            users_df[col] = pd.to_datetime(users_df[col], errors='coerce')
    
    if verbose:
        print("Date column types AFTER conversion:")
        for col in datetime_cols_users:
            if col in users_df.columns:
                print(f"  - users_df['{col}']: {users_df[col].dtype}")
        for col in datetime_cols_trips:
            if col in trips_df.columns:
                print(f"  - trips_df['{col}']: {trips_df[col].dtype}")
    
    # Step 4: Memory optimization (preserve datetime columns)
    if verbose:
        print(f"\n4. Memory optimization (with datetime preservation)...")
    
    trips_df = optimize_memory_usage(trips_df, preserve_datetime_cols=datetime_cols_trips, verbose=verbose)
    users_df = optimize_memory_usage(users_df, preserve_datetime_cols=datetime_cols_users, verbose=verbose)
    
    if verbose:
        print(f"\nPREPROCESSING COMPLETED!")
        print(f"Final state:")
        print(f"  - Users: {users_df.shape[0]:,} rows, {users_df.shape[1]} columns")
        print(f"  - Trips: {trips_df.shape[0]:,} rows, {trips_df.shape[1]} columns")
        
        # Show final column lists
        print(f"\nFinal column names:")
        print(f"  - Users: {list(users_df.columns)}")
        print(f"  - Trips: {list(trips_df.columns)}")
    
    return users_df, trips_df


