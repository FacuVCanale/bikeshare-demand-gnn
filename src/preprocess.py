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
        duracion_valid = duracion_valid[(duracion_valid > 120) & (duracion_valid < 60 * 60 * 12)]  # filter: >2 min, <12 hours
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
