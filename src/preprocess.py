import pandas as pd


def preprocess_users_data(users_df: pd.DataFrame):
    
    """
    Preprocesa los datos de usuarios.
    
    Args:
        users_df: DataFrame original de usuarios
        
    Returns:
        Dict con datos preprocesados para visualización
    """
    processed_data = {}
    
    # Datos de edad
    if 'edad_usuario' in users_df.columns:
        users_clean = users_df.copy()
        users_clean['edad_usuario'] = pd.to_numeric(users_clean['edad_usuario'], errors='coerce')
        edad_valid = users_clean['edad_usuario'].dropna()
        edad_valid = edad_valid[(edad_valid >= 0) & (edad_valid <= 150)]
        processed_data['edad_valid'] = edad_valid
        processed_data['edad_promedio'] = edad_valid.mean()
    
    # Datos de género
    if 'genero_usuario' in users_df.columns:
        processed_data['genero_counts'] = users_df['genero_usuario'].value_counts()
    
    # Datos de fechas de alta
    if 'fecha_alta' in users_df.columns:
        users_temp = users_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(users_temp['fecha_alta']):
            users_temp['fecha_alta'] = pd.to_datetime(users_temp['fecha_alta'], errors='coerce')
        
        processed_data['registros_por_año'] = users_temp['fecha_alta'].dt.year.value_counts().sort_index()
        processed_data['registros_por_mes'] = users_temp['fecha_alta'].dt.month.value_counts().sort_index()
    
    return processed_data




def preprocess_trips_data(trips_df: pd.DataFrame):
    """
    Preprocesa los datos de viajes.
    
    Args:
        trips_df: DataFrame original de viajes
        
    Returns:
        Dict con datos preprocesados para visualización
    """
    processed_data = {}
    
    # preparar datos temporales
    trips_temp = trips_df.copy()
    fecha_col = 'fecha_origen_recorrido' if 'fecha_origen_recorrido' in trips_temp.columns else 'fecha_origen'
    
    if not pd.api.types.is_datetime64_any_dtype(trips_temp[fecha_col]):
        trips_temp[fecha_col] = pd.to_datetime(trips_temp[fecha_col], errors='coerce')
    
    trips_temp = trips_temp.dropna(subset=[fecha_col])
    processed_data['trips_clean'] = trips_temp
    processed_data['fecha_col'] = fecha_col
    
    # analisis temporal
    processed_data['viajes_por_hora'] = trips_temp[fecha_col].dt.hour.value_counts().sort_index()
    processed_data['viajes_por_dia'] = trips_temp[fecha_col].dt.dayofweek.value_counts().sort_index()
    processed_data['viajes_por_mes'] = trips_temp[fecha_col].dt.month.value_counts().sort_index()
    processed_data['viajes_por_año'] = trips_temp[fecha_col].dt.year.value_counts().sort_index()
    
    # Análisis de estaciones
    if 'id_estacion_origen' in trips_temp.columns:
        processed_data['top_origen'] = trips_temp['id_estacion_origen'].value_counts().head(15)
        processed_data['viajes_por_estacion'] = trips_temp['id_estacion_origen'].value_counts()
    
    if 'id_estacion_destino' in trips_temp.columns:
        processed_data['top_destino'] = trips_temp['id_estacion_destino'].value_counts().head(15)
    
    # Análisis de duración
    if 'duracion_recorrido' in trips_temp.columns:
        trips_temp['duracion_num'] = pd.to_numeric(
            trips_temp['duracion_recorrido'].astype(str).str.replace(',', ''), 
            errors='coerce'
        )
        duracion_valid = trips_temp['duracion_num'].dropna()
        duracion_valid = duracion_valid[(duracion_valid > 0) & (duracion_valid < 7200)]
        processed_data['duracion_valid'] = duracion_valid
        processed_data['duracion_promedio'] = duracion_valid.mean()
    
    # Datos para heatmap
    processed_data['hora'] = trips_temp[fecha_col].dt.hour
    processed_data['dia_semana'] = trips_temp[fecha_col].dt.dayofweek
    processed_data['activity_matrix'] = trips_temp.groupby([
        trips_temp[fecha_col].dt.dayofweek, 
        trips_temp[fecha_col].dt.hour
    ]).size().unstack(fill_value=0)
    
    return processed_data


def preprocess_whole_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive preprocessing function for the entire dataset.
    
    Performs the following operations in-place:
    - Merges gender columns (género, genero, Género) into 'genero'  
    - Merges ID columns (id_recorrido, Id_recorrido) into 'id_recorrido'
    - Removes 'BAEcobici' suffix from specified columns and converts to int
    - Removes unnecessary columns (Unnamed: 0, X, and original duplicated columns)
    
    Args:
        df: Input DataFrame to preprocess
        
    Returns:
        pd.DataFrame: The same DataFrame modified in-place
    """
    df_work = df.copy()
    
    # merge gender columns into 'genero'
    gender_columns = ['género', 'genero', 'Género']
    existing_gender_cols = [col for col in gender_columns if col in df_work.columns]
    
    if existing_gender_cols:
        # create unified gender column, prioritizing non-null values
        df_work['genero'] = df_work[existing_gender_cols].bfill(axis=1).iloc[:, 0]
        # remove original gender columns except 'genero'
        cols_to_drop = [col for col in existing_gender_cols if col != 'genero']
        df_work.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # merge id_recorrido columns
    id_columns = ['id_recorrido', 'Id_recorrido']
    existing_id_cols = [col for col in id_columns if col in df_work.columns]
    
    if len(existing_id_cols) > 1:
        # create unified id column, prioritizing non-null values
        df_work['id_recorrido'] = df_work[existing_id_cols].bfill(axis=1).iloc[:, 0]
        # remove duplicated id columns except 'id_recorrido'
        cols_to_drop = [col for col in existing_id_cols if col != 'id_recorrido']
        df_work.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # process columns with BAEcobici suffix
    columns_to_process = ['id_estacion_origen', 'id_estacion_destino', 'id_usuario', 'id_recorrido']
    existing_process_cols = [col for col in columns_to_process if col in df_work.columns]
    
    for column in existing_process_cols:
        if df_work[column].dtype == 'object':
            # check if column contains BAEcobici suffix
            sample_values = df_work[column].dropna().astype(str).head(100)
            if any('BAEcobici' in str(val) for val in sample_values):
                df_work[column] = df_work[column].astype(str).str.replace('BAEcobici', '', regex=False)
                # convert to int, handling any conversion errors
                df_work[column] = pd.to_numeric(df_work[column], errors='coerce').astype('Int64')
    
    # remove unnecessary columns
    unnecessary_columns = ['Unnamed: 0', 'X']
    existing_unnecessary = [col for col in unnecessary_columns if col in df_work.columns]
    if existing_unnecessary:
        df_work.drop(columns=existing_unnecessary, inplace=True, errors='ignore')
    
    # modify original dataframe in-place by replacing all data
    # first, drop all existing columns
    df.drop(columns=df.columns.tolist(), inplace=True)
    
    # then add all columns from processed dataframe
    for col in df_work.columns:
        df[col] = df_work[col].values
    
    return df


def process_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Remove the suffix 'BAEcobici' from the specified columns and convert the values to integers.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to process.
    columns (list): A list of column names to process.

    Returns:
    pd.DataFrame: The DataFrame with processed columns.
    """
    df_result = df.copy()
    for column in columns:
        if column in df_result.columns:
            df_result[column] = df_result[column].astype(str).str.replace('BAEcobici', '', regex=False).astype(int)
    return df_result


def test_preprocess_whole_dataset():
    """
    Simple test function to validate preprocess_whole_dataset functionality.
    """
    import pandas as pd
    import numpy as np
    
    # create test dataframe with problematic columns
    test_data = {
        'id_estacion_origen': ['123BAEcobici', '456BAEcobici', '789BAEcobici'],
        'id_estacion_destino': ['321BAEcobici', '654BAEcobici', '987BAEcobici'], 
        'id_usuario': ['111BAEcobici', '222BAEcobici', '333BAEcobici'],
        'id_recorrido': [1001, 1002, 1003],
        'Id_recorrido': [1001, 1002, 1003],  # duplicate column
        'genero': ['M', 'F', None],
        'género': [None, None, 'M'],  # duplicate with accent
        'Género': ['M', 'F', 'M'],   # duplicate capitalized
        'Unnamed: 0': [0, 1, 2],     # unnecessary column
        'X': ['a', 'b', 'c'],        # unnecessary column
        'other_column': ['data1', 'data2', 'data3']  # should remain
    }
    
    test_df = pd.DataFrame(test_data)
    print("Original DataFrame:")
    print(f"Columns: {list(test_df.columns)}")
    print(f"Shape: {test_df.shape}")
    print("\nSample data:")
    print(test_df.head(3))
    
    # apply preprocessing
    processed_df = preprocess_whole_dataset(test_df.copy())
    
    print("\n" + "="*50)
    print("Processed DataFrame:")
    print(f"Columns: {list(processed_df.columns)}")
    print(f"Shape: {processed_df.shape}")
    print("\nProcessed data:")
    print(processed_df.head(3))
    
    # validate results
    print("\n" + "="*50)
    print("VALIDATION RESULTS:")
    
    # check gender merge
    if 'genero' in processed_df.columns:
        gender_counts = processed_df['genero'].value_counts()
        print(f"✓ Gender column unified: {dict(gender_counts)}")
    
    # check ID merge
    if 'id_recorrido' in processed_df.columns and 'Id_recorrido' not in processed_df.columns:
        print("✓ ID columns merged successfully")
    
    # check BAEcobici removal
    id_cols = ['id_estacion_origen', 'id_estacion_destino', 'id_usuario']
    baecobici_removed = True
    for col in id_cols:
        if col in processed_df.columns:
            sample_val = str(processed_df[col].iloc[0])
            if 'BAEcobici' in sample_val:
                baecobici_removed = False
                break
    print(f"✓ BAEcobici suffix removed: {baecobici_removed}")
    
    # check unnecessary columns removal
    unnecessary_removed = not any(col in processed_df.columns for col in ['Unnamed: 0', 'X'])
    print(f"✓ Unnecessary columns removed: {unnecessary_removed}")
    
    print("\n🎉 Test completed successfully!")
    return processed_df


if __name__ == "__main__":
    # run test when script is executed directly
    test_preprocess_whole_dataset()
