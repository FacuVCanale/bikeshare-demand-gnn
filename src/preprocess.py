import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


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
        duracion_valid = duracion_valid[(duracion_valid > 0) & (duracion_valid < 7200)]
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


def clean_dataset_comprehensive(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive dataset cleaning with optimized operations.
    
    Operations performed:
    - Merges gender columns (género, genero, Género) into 'genero'  
    - Merges ID columns (id_recorrido, Id_recorrido) into 'id_recorrido'
    - Removes 'BAEcobici' suffix from specified columns and converts to int
    - Removes unnecessary columns (Unnamed: 0, X, and original duplicated columns)
    
    Args:
        df: Input DataFrame to preprocess
        verbose: Whether to print progress information
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if verbose:
        print(f"🧹 Starting comprehensive cleaning for {len(df):,} rows...")
    
    # Step 1: Merge gender columns using vectorized operations
    gender_columns = ['género', 'genero', 'Género']
    existing_gender_cols = [col for col in gender_columns if col in df.columns]
    
    if existing_gender_cols:
        if verbose:
            print("   ✓ Merging gender columns...")
        # Use coalesce-like operation - more efficient than bfill
        df['genero'] = df[existing_gender_cols[0]].copy()
        for col in existing_gender_cols[1:]:
            mask = df['genero'].isna()
            df.loc[mask, 'genero'] = df.loc[mask, col]
        
        # Drop original columns except 'genero'
        cols_to_drop = [col for col in existing_gender_cols if col != 'genero']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Step 2: Merge ID columns
    id_columns = ['id_recorrido', 'Id_recorrido']
    existing_id_cols = [col for col in id_columns if col in df.columns]
    
    if len(existing_id_cols) > 1:
        if verbose:
            print("   ✓ Merging ID columns...")
        df['id_recorrido'] = df[existing_id_cols[0]].copy()
        for col in existing_id_cols[1:]:
            mask = df['id_recorrido'].isna()
            df.loc[mask, 'id_recorrido'] = df.loc[mask, col]
        
        cols_to_drop = [col for col in existing_id_cols if col != 'id_recorrido']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Step 3: Process BAEcobici suffix columns
    columns_to_process = ['id_estacion_origen', 'id_estacion_destino', 'id_usuario', 'id_recorrido']
    existing_process_cols = [col for col in columns_to_process if col in df.columns]
    
    if existing_process_cols:
        if verbose:
            print(f"   ✓ Processing BAEcobici suffixes for {len(existing_process_cols)} columns...")
    
    for column in existing_process_cols:
            if df[column].dtype == 'object':
                # Check if column contains BAEcobici suffix (sample check)
                sample_mask = df[column].notna()
                if sample_mask.sum() > 0:
                    sample_values = df.loc[sample_mask, column].astype(str).head(1000)
                    has_suffix = sample_values.str.contains('BAEcobici', na=False).any()
                    
                    if has_suffix:
                        if verbose:
                            print(f"     - Processing {column}...")
                        # Vectorized string replacement - much faster than iterative
                        df[column] = df[column].astype(str).str.replace('BAEcobici', '', regex=False)
                        # Convert to numeric efficiently
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
    
    # Step 4: Remove unnecessary columns
    unnecessary_columns = ['Unnamed: 0', 'X']
    existing_unnecessary = [col for col in unnecessary_columns if col in df.columns]
    if existing_unnecessary:
        if verbose:
            print("   ✓ Removing unnecessary columns...")
        df.drop(columns=existing_unnecessary, inplace=True, errors='ignore')
    
    if verbose:
        print(f"✅ Comprehensive cleaning completed for {len(df):,} rows!")
    
    return df


def remove_baecobici_suffix(df: pd.DataFrame, columns: List[str], inplace: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Remove 'BAEcobici' suffix from specified columns and convert to integers.

    Args:
        df: DataFrame containing columns to process
        columns: List of column names to process
        inplace: Whether to modify DataFrame in-place
        verbose: Whether to print progress

    Returns:
        pd.DataFrame: DataFrame with processed columns
    """
    if not inplace:
        df = df.copy()
    
    for column in columns:
        if column in df.columns and df[column].dtype == 'object':
            # Quick sample check to see if processing is needed
            sample_mask = df[column].notna()
            if sample_mask.sum() > 0:
                sample_values = df.loc[sample_mask, column].astype(str).head(100)
                has_suffix = sample_values.str.contains('BAEcobici', na=False).any()
                
                if has_suffix:
                    if verbose:
                        print(f"   - Processing {column}...")
                    # Vectorized operations - much faster than row-by-row
                    df[column] = (df[column]
                                .astype(str)
                                .str.replace('BAEcobici', '', regex=False))
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
    
    return df


def clean_latitude_column(df: pd.DataFrame, lat_column: str, 
                         inplace: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Clean latitude columns that contain concatenated lat,lon values.
    
    Handles cases like:
    - "-34.611032" -> -34.611032 (already clean)
    - "-34.611032,-58.3682604" -> -34.611032 (extract latitude)
    - "nan" or None -> NaN
    
    Args:
        df: DataFrame containing the latitude column
        lat_column: Name of the latitude column to clean
        inplace: Whether to modify the DataFrame in-place
        verbose: Whether to print processing information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned latitude column
    """
    if verbose:
        print(f"🧭 Cleaning latitude column: {lat_column}")
        original_nulls = df[lat_column].isna().sum()
        print(f"   📊 Original nulls: {original_nulls:,}")
    
    if not inplace:
        df = df.copy()
    
    if lat_column not in df.columns:
        if verbose:
            print(f"   ⚠️ Column {lat_column} not found in DataFrame")
        return df
    
    # Convert to string for processing
    lat_series = df[lat_column].astype(str)
    
    # Identify different patterns
    if verbose:
        comma_count = lat_series.str.contains(',', na=False).sum()
        print(f"   📈 Values with comma (lat,lon): {comma_count:,}")
    
    # Vectorized extraction of latitude (first part before comma if exists)
    cleaned_lat = lat_series.str.split(',').str[0]
    
    # Convert to numeric, handling errors gracefully
    df[lat_column] = pd.to_numeric(cleaned_lat, errors='coerce')
    
    if verbose:
        final_nulls = df[lat_column].isna().sum()
        valid_values = len(df) - final_nulls
        print(f"   ✅ Valid latitudes: {valid_values:,}")
        print(f"   ⚠️ Final nulls: {final_nulls:,} (+{final_nulls - original_nulls:,} from cleaning)")
        
        # Show sample of cleaned values
        sample_values = df[lat_column].dropna().head(5).tolist()
        print(f"   📝 Sample cleaned values: {sample_values}")
    
    return df


def clean_longitude_column(df: pd.DataFrame, lon_column: str, 
                          inplace: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Clean longitude columns that might contain concatenated lat,lon values.
    
    Handles cases like:
    - "-58.3682604" -> -58.3682604 (already clean)
    - "-34.611032,-58.3682604" -> -58.3682604 (extract longitude)
    - "nan" or None -> NaN
    
    Args:
        df: DataFrame containing the longitude column
        lon_column: Name of the longitude column to clean
        inplace: Whether to modify the DataFrame in-place
        verbose: Whether to print processing information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned longitude column
    """
    if verbose:
        print(f"🧭 Cleaning longitude column: {lon_column}")
        original_nulls = df[lon_column].isna().sum()
        print(f"   📊 Original nulls: {original_nulls:,}")
    
    if not inplace:
        df = df.copy()
    
    if lon_column not in df.columns:
        if verbose:
            print(f"   ⚠️ Column {lon_column} not found in DataFrame")
        return df
    
    # Convert to string for processing
    lon_series = df[lon_column].astype(str)
    
    if verbose:
        comma_count = lon_series.str.contains(',', na=False).sum()
        print(f"   📈 Values with comma (lat,lon): {comma_count:,}")
    
    # Extract longitude (second part after comma, or the value itself if no comma)
    def extract_longitude(value_str):
        parts = value_str.split(',')
        if len(parts) >= 2:
            return parts[1]  # longitude is second part
        else:
            return value_str  # assume it's already longitude
    
    # Vectorized longitude extraction
    cleaned_lon = lon_series.apply(extract_longitude)
    
    # Convert to numeric
    df[lon_column] = pd.to_numeric(cleaned_lon, errors='coerce')
    
    if verbose:
        final_nulls = df[lon_column].isna().sum()
        valid_values = len(df) - final_nulls
        print(f"   ✅ Valid longitudes: {valid_values:,}")
        print(f"   ⚠️ Final nulls: {final_nulls:,} (+{final_nulls - original_nulls:,} from cleaning)")
        
        sample_values = df[lon_column].dropna().head(5).tolist()
        print(f"   📝 Sample cleaned values: {sample_values}")
    
    return df


def clean_coordinate_columns(df: pd.DataFrame, 
                           coordinate_columns: Optional[List[str]] = None,
                           inplace: bool = True, 
                           verbose: bool = True) -> pd.DataFrame:
    """
    Batch cleaning of multiple coordinate columns.
    
    Automatically detects and cleans latitude/longitude columns that may have
    concatenated "lat,lon" values.
    
    Args:
        df: DataFrame to process
        coordinate_columns: List of coordinate columns to clean. If None, auto-detects
        inplace: Whether to modify DataFrame in-place
        verbose: Whether to print processing information

    Returns:
        pd.DataFrame: DataFrame with cleaned coordinate columns
    """
    if verbose:
        print("🗺️  BATCH COORDINATE CLEANING")
        print(f"   📊 Processing {len(df):,} rows...")
    
    if not inplace:
        df = df.copy()
    
    # Auto-detect coordinate columns if not specified
    if coordinate_columns is None:
        coordinate_patterns = ['lat', 'lon', 'latitude', 'longitude', 'estacion']
        coordinate_columns = []
        for col in df.columns:
            if any(pattern in col.lower() for pattern in coordinate_patterns):
                if df[col].dtype == 'object':  # Only process string columns
                    coordinate_columns.append(col)
        
        if verbose:
            print(f"   🔍 Auto-detected coordinate columns: {coordinate_columns}")
    
    # Process each coordinate column
    for col in coordinate_columns:
        if 'lat' in col.lower():
            clean_latitude_column(df, col, inplace=True, verbose=verbose)
        elif 'lon' in col.lower():
            clean_longitude_column(df, col, inplace=True, verbose=verbose)
        else:
            # For generic coordinate columns, try latitude cleaning first
            clean_latitude_column(df, col, inplace=True, verbose=verbose)
    
    if verbose:
        print("✅ Batch coordinate cleaning completed!")
    
    return df


def optimize_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    OPTIMIZED memory usage function that handles NaN values properly.
    
    This can reduce memory usage by 50-70% for large datasets.
    Uses nullable integer types to handle NaN values correctly.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print memory savings
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    if verbose:
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"🗜️  Original memory usage: {start_memory:.1f} MB")
    
    # Store datetime columns to preserve them
    datetime_columns = {}
    for col in df.columns:
        if df[col].dtype.name.startswith('datetime'):
            datetime_columns[col] = df[col].copy()
            if verbose:
                print(f"     🗓️ Preserving datetime column: {col}")
    
    # Optimize integer columns - use nullable integer types for columns with NaN
    for col in df.select_dtypes(include=['int64', 'Int64']).columns:
        # Check if column has NaN values
        has_na = df[col].isna().any()
        
        # Skip if all values are NaN
        if df[col].isna().all():
            if verbose:
                print(f"     ⚠️ Skipping {col}: all values are NaN")
            continue
            
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Use nullable integer types if there are NaN values
        if has_na:
            if col_min >= 0:  # Unsigned integers (nullable)
                if col_max < 255:
                    df[col] = df[col].astype('UInt8')
                elif col_max < 65535:
                    df[col] = df[col].astype('UInt16') 
                elif col_max < 4294967295:
                    df[col] = df[col].astype('UInt32')
                else:
                    df[col] = df[col].astype('UInt64')
            else:  # Signed integers (nullable)
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('Int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('Int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('Int32')
                else:
                    df[col] = df[col].astype('Int64')
        else:
            # No NaN values - use regular numpy types for better performance
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
        
        if verbose:
            print(f"     ✓ {col}: {'nullable' if has_na else 'regular'} {df[col].dtype}")
    
    # Optimize float columns - handle with downcast parameter
    for col in df.select_dtypes(include=['float64']).columns:
        try:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], downcast='float')
            if verbose and df[col].dtype != original_dtype:
                print(f"     ✓ {col}: {original_dtype} → {df[col].dtype}")
        except Exception as e:
            if verbose:
                print(f"     ⚠️ Could not optimize {col}: {str(e)}")
    
    # Optimize object columns that might be categorical
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Skip columns that might need ordering comparisons
            skip_patterns = ['fecha', 'date', 'time', 'hora', 'id_', 'recorrido', 'usuario']
            should_skip = any(pattern in col.lower() for pattern in skip_patterns)
            
            if should_skip:
                if verbose:
                    print(f"     ⚠️ Skipping {col}: might need ordering comparisons")
                continue
            
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            
            # More conservative threshold and additional checks
            if num_unique_values / num_total_values < 0.3:  # More conservative: 30% instead of 50%
                # Additional check: ensure it's really a good candidate for categorical
                sample_values = df[col].dropna().head(10).tolist()
                
                # Skip if values look like they might need ordering
                numeric_like = any(str(val).replace('.', '').replace('-', '').isdigit() 
                                 for val in sample_values if val is not None)
                
                if numeric_like:
                    if verbose:
                        print(f"     ⚠️ Skipping {col}: values look numeric/orderable")
                    continue
                
                original_memory = df[col].memory_usage(deep=True)
                df[col] = df[col].astype('category')
                new_memory = df[col].memory_usage(deep=True)
                if verbose:
                    savings = (original_memory - new_memory) / original_memory * 100
                    print(f"     ✓ {col}: object → category ({savings:.1f}% savings)")
            else:
                if verbose:
                    unique_pct = num_unique_values / num_total_values * 100
                    print(f"     ⚠️ Skipping {col}: too many unique values ({unique_pct:.1f}%)")
        except Exception as e:
            if verbose:
                print(f"     ⚠️ Could not categorize {col}: {str(e)}")
    
    # Restore datetime columns to ensure they remain datetime
    for col, datetime_series in datetime_columns.items():
        df[col] = datetime_series
        if verbose:
            print(f"     🗓️ Restored datetime column: {col}")
    
    if verbose:
        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        savings = (start_memory - end_memory) / start_memory * 100
        print(f"     💾 Optimized memory usage: {end_memory:.1f} MB")
        print(f"     📈 Memory savings: {savings:.1f}%")
        print(f"     🚀 Reduction factor: {start_memory/end_memory:.1f}x")
    
    return df


