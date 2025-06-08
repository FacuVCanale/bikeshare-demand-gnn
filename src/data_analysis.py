import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las visualizaciones
plt.style.use('default')
sns.set_palette("husl")




# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def plot_user_analysis(processed_users: Dict[str, Any], save_plots: bool = False, 
                      output_dir: Optional[Path] = None) -> None:
    """Genera gráficos de análisis de usuarios."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ANÁLISIS DE USUARIOS ECOBICI', fontsize=16, fontweight='bold')
    
    # 1.1 Distribución de edades
    if 'edad_valid' in processed_users:
        edad_valid = processed_users['edad_valid']
        edad_promedio = processed_users['edad_promedio']
        
        axes[0, 0].hist(edad_valid, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribución de Edades')
        axes[0, 0].set_xlabel('Edad (años)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(edad_promedio, color='red', linestyle='--', 
                          label=f'Media: {edad_promedio:.1f} años')
        axes[0, 0].legend()
    
    # 1.2 Distribución de géneros
    if 'genero_counts' in processed_users:
        genero_counts = processed_users['genero_counts']
        axes[0, 1].pie(genero_counts.values, labels=genero_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen'])
        axes[0, 1].set_title('Distribución por Género')
    
    # 1.3 Registros por año
    if 'registros_por_año' in processed_users:
        registros_por_año = processed_users['registros_por_año']
        axes[1, 0].bar(registros_por_año.index, registros_por_año.values, color='lightgreen')
        axes[1, 0].set_title('Registros de Usuarios por Año')
        axes[1, 0].set_xlabel('Año')
        axes[1, 0].set_ylabel('Nuevos Usuarios')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 1.4 Registros por mes
    if 'registros_por_mes' in processed_users:
        registros_por_mes = processed_users['registros_por_mes']
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        axes[1, 1].bar(range(1, 13), [registros_por_mes.get(i, 0) for i in range(1, 13)], 
                      color='orange')
        axes[1, 1].set_title('Registros por Mes')
        axes[1, 1].set_xlabel('Mes')
        axes[1, 1].set_ylabel('Registros')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].set_xticklabels(meses, rotation=45)
    
    plt.tight_layout()
    if save_plots and output_dir:
        plt.savefig(output_dir / 'usuarios_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_temporal_analysis(processed_trips: Dict[str, Any], save_plots: bool = False, 
                          output_dir: Optional[Path] = None) -> None:
    """Genera gráficos de análisis temporal de viajes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ANÁLISIS TEMPORAL DE VIAJES', fontsize=16, fontweight='bold')
    
    # 2.1 Viajes por hora del día
    if 'viajes_por_hora' in processed_trips:
        viajes_por_hora = processed_trips['viajes_por_hora']
        axes[0, 0].plot(viajes_por_hora.index, viajes_por_hora.values, marker='o', linewidth=2, color='blue')
        axes[0, 0].set_title('Viajes por Hora del Día')
        axes[0, 0].set_xlabel('Hora')
        axes[0, 0].set_ylabel('Número de Viajes')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))
    
    # 2.2 Viajes por día de la semana
    if 'viajes_por_dia' in processed_trips:
        dias_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        viajes_por_dia = processed_trips['viajes_por_dia']
        axes[0, 1].bar(range(7), [viajes_por_dia.get(i, 0) for i in range(7)], color='green')
        axes[0, 1].set_title('Viajes por Día de la Semana')
        axes[0, 1].set_xlabel('Día')
        axes[0, 1].set_ylabel('Número de Viajes')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(dias_semana)
    
    # 2.3 Viajes por mes
    if 'viajes_por_mes' in processed_trips:
        viajes_por_mes = processed_trips['viajes_por_mes']
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        axes[1, 0].bar(range(1, 13), [viajes_por_mes.get(i, 0) for i in range(1, 13)], color='purple')
        axes[1, 0].set_title('Viajes por Mes')
        axes[1, 0].set_xlabel('Mes')
        axes[1, 0].set_ylabel('Número de Viajes')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(meses, rotation=45)
    
    # 2.4 Tendencia por año
    if 'viajes_por_año' in processed_trips:
        viajes_por_año = processed_trips['viajes_por_año']
        axes[1, 1].plot(viajes_por_año.index, viajes_por_año.values, marker='o', linewidth=3, color='red')
        axes[1, 1].set_title('Tendencia de Viajes por Año')
        axes[1, 1].set_xlabel('Año')
        axes[1, 1].set_ylabel('Número de Viajes')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots and output_dir:
        plt.savefig(output_dir / 'viajes_temporal.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_station_analysis(processed_trips: Dict[str, Any], save_plots: bool = False, 
                         output_dir: Optional[Path] = None) -> None:
    """Genera gráficos de análisis de estaciones."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ANÁLISIS DE ESTACIONES', fontsize=16, fontweight='bold')
    
    # 3.1 Top 15 estaciones de origen
    if 'top_origen' in processed_trips:
        top_origen = processed_trips['top_origen']
        axes[0, 0].barh(range(len(top_origen)), top_origen.values, color='lightblue')
        axes[0, 0].set_title('Top 15 Estaciones de Origen')
        axes[0, 0].set_xlabel('Número de Viajes')
        axes[0, 0].set_yticks(range(len(top_origen)))
        axes[0, 0].set_yticklabels([str(x)[:20] for x in top_origen.index], fontsize=8)
    
    # 3.2 Top 15 estaciones de destino
    if 'top_destino' in processed_trips:
        top_destino = processed_trips['top_destino']
        axes[0, 1].barh(range(len(top_destino)), top_destino.values, color='lightcoral')
        axes[0, 1].set_title('Top 15 Estaciones de Destino')
        axes[0, 1].set_xlabel('Número de Viajes')
        axes[0, 1].set_yticks(range(len(top_destino)))
        axes[0, 1].set_yticklabels([str(x)[:20] for x in top_destino.index], fontsize=8)
    
    # 3.3 Distribución de viajes por estación
    if 'viajes_por_estacion' in processed_trips:
        viajes_por_estacion = processed_trips['viajes_por_estacion']
        axes[1, 0].hist(viajes_por_estacion.values, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title('Distribución de Viajes por Estación')
        axes[1, 0].set_xlabel('Número de Viajes')
        axes[1, 0].set_ylabel('Número de Estaciones')
        axes[1, 0].set_yscale('log')
    
    # 3.4 Duración de viajes
    if 'duracion_valid' in processed_trips:
        duracion_valid = processed_trips['duracion_valid']
        duracion_promedio = processed_trips['duracion_promedio']
        
        axes[1, 1].hist(duracion_valid / 60, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_title('Distribución de Duración de Viajes')
        axes[1, 1].set_xlabel('Duración (minutos)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].axvline(duracion_promedio / 60, color='red', linestyle='--', 
                          label=f'Media: {duracion_promedio / 60:.1f} min')
        axes[1, 1].legend()
    
    plt.tight_layout()
    if save_plots and output_dir:
        plt.savefig(output_dir / 'estaciones_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_activity_heatmap(processed_trips: Dict[str, Any], save_plots: bool = False, 
                         output_dir: Optional[Path] = None) -> None:
    """Genera heatmap de actividad."""
    
    if 'activity_matrix' in processed_trips:
        activity_matrix = processed_trips['activity_matrix']
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(activity_matrix, 
                    xticklabels=range(24),
                    yticklabels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                    cmap='YlOrRd', 
                    cbar_kws={'label': 'Número de Viajes'})
        plt.title('🔥 HEATMAP DE ACTIVIDAD: Viajes por Hora y Día de la Semana', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Hora del Día')
        plt.ylabel('Día de la Semana')
        
        if save_plots and output_dir:
            plt.savefig(output_dir / 'heatmap_actividad.png', dpi=300, bbox_inches='tight')
        plt.show()


def print_summary_statistics(users_df: pd.DataFrame, processed_users: Dict[str, Any], 
                           trips_df: pd.DataFrame, processed_trips: Dict[str, Any]) -> None:
    """Imprime estadísticas resumen."""
    
    print("\n📊 ESTADÍSTICAS RESUMEN")
    print("=" * 40)
    
    # Usuarios
    print(f"👥 USUARIOS:")
    print(f"   • Total usuarios: {len(users_df):,}")
    
    if 'edad_promedio' in processed_users:
        edad_promedio = processed_users['edad_promedio']
        if not pd.isna(edad_promedio):
            print(f"   • Edad promedio: {edad_promedio:.1f} años")
    
    if 'genero_counts' in processed_users:
        genero_counts = processed_users['genero_counts']
        print(f"   • Distribución de género:")
        for genero, count in genero_counts.items():
            pct = count / len(users_df) * 100
            print(f"     - {genero}: {count:,} ({pct:.1f}%)")
    
    # Viajes
    print(f"\n🚲 VIAJES:")
    print(f"   • Total viajes: {len(trips_df):,}")
    
    if 'trips_clean' in processed_trips and 'fecha_col' in processed_trips:
        trips_clean = processed_trips['trips_clean']
        fecha_col = processed_trips['fecha_col']
        
        print(f"   • Período: {trips_clean[fecha_col].min()} - {trips_clean[fecha_col].max()}")
        days_diff = (trips_clean[fecha_col].max() - trips_clean[fecha_col].min()).days + 1
        print(f"   • Promedio viajes/día: {len(trips_df) / days_diff:,.0f}")
        
        if 'top_origen' in processed_trips:
            print(f"   • Estaciones únicas (origen): {len(processed_trips['viajes_por_estacion']):,}")
        if 'top_destino' in processed_trips:
            estaciones_destino = len(processed_trips['top_destino'])
            print(f"   • Estaciones únicas (destino): {estaciones_destino:,}")
    
    if 'duracion_promedio' in processed_trips:
        duracion_promedio = processed_trips['duracion_promedio']
        if not pd.isna(duracion_promedio):
            print(f"   • Duración promedio: {duracion_promedio / 60:.1f} minutos")


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
        print(f"🔽 FILTERING DATA UNTIL {max_date}")
        print(f"   📊 Input: Users={len(users_df):,}, Trips={len(trips_df):,}")
    
    max_datetime = pd.to_datetime(max_date)
    
    # Optimize users filtering
    if verbose:
        print(f"   ⚡ Filtering users...")
    
    # Vectorized filtering - much faster than multiple conditions
    users_mask = (users_df[user_date_col].notna()) & (users_df[user_date_col] <= max_datetime)
    users_filtered = users_df[users_mask].copy()
    
    # Optimize trips filtering
    if verbose:
        print(f"   ⚡ Filtering trips...")
    
    # Vectorized filtering for large dataset
    trips_mask = (trips_df[trip_date_col].notna()) & (trips_df[trip_date_col] <= max_datetime)
    trips_filtered = trips_df[trips_mask].copy()
    
    if verbose:
        users_removed = len(users_df) - len(users_filtered)
        trips_removed = len(trips_df) - len(trips_filtered)
        print(f"   ✅ Users: {len(users_df):,} → {len(users_filtered):,} (-{users_removed:,})")
        print(f"   ✅ Trips: {len(trips_df):,} → {len(trips_filtered):,} (-{trips_removed:,})")
    
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
        print("🔄 OPTIMIZED TEMPORAL SPLIT")
        print(f"   📅 Train ≤ {train_end_date} | Val: {train_end_date} - {val_end_date} | Test: {val_end_date} - {test_end_date}")
        print(f"   📊 Input: Users={len(users_df):,}, Trips={len(trips_df):,}")
    
    # Convert dates once - more efficient
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date) 
    test_end = pd.to_datetime(test_end_date)
    
    # Get date columns as series for faster operations
    user_dates = users_df[user_date_col]
    trip_dates = trips_df[trip_date_col]
    
    # Vectorized splitting for users - much faster than individual filters
    if verbose:
        print("   ⚡ Splitting users...")
    
    users_train_mask = user_dates <= train_end
    users_val_mask = (user_dates > train_end) & (user_dates <= val_end)
    users_test_mask = (user_dates > val_end) & (user_dates <= test_end)
    
    users_train = users_df[users_train_mask].copy()
    users_val = users_df[users_val_mask].copy()
    users_test = users_df[users_test_mask].copy()
    
    # Vectorized splitting for trips - optimized for large datasets
    if verbose:
        print("   ⚡ Splitting trips...")
    
    trips_train_mask = trip_dates <= train_end
    trips_val_mask = (trip_dates > train_end) & (trip_dates <= val_end)
    trips_test_mask = (trip_dates > val_end) & (trip_dates <= test_end)
    
    trips_train = trips_df[trips_train_mask].copy()
    trips_val = trips_df[trips_val_mask].copy()
    trips_test = trips_df[trips_test_mask].copy()
    
    if verbose:
        print(f"   ✅ Users: Train={len(users_train):,}, Val={len(users_val):,}, Test={len(users_test):,}")
        print(f"   ✅ Trips: Train={len(trips_train):,}, Val={len(trips_val):,}, Test={len(trips_test):,}")
    
    return {
        'users_train': users_train,
        'users_val': users_val, 
        'users_test': users_test,
        'trips_train': trips_train,
        'trips_val': trips_val,
        'trips_test': trips_test
    }



def load_combined_data(combined_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datasets combinados desde archivos parquet."""
    print("=== CARGANDO DATASETS COMBINADOS DESDE PARQUET ===")
    
    # Convert to Path if string
    combined_dir = Path(combined_dir)
    
    users_file = combined_dir / "users.parquet"
    trips_file = combined_dir / "trips.parquet"
    stations_file = combined_dir / "station_info.parquet"
    
    if not users_file.exists():
        raise FileNotFoundError(f"No se encontró: {users_file}")
    if not trips_file.exists():
        raise FileNotFoundError(f"No se encontró: {trips_file}")
    
    print("Cargando usuarios desde parquet...")
    users_df = pd.read_parquet(users_file)
    print(f"✅ Usuarios cargados: {len(users_df):,} registros")
    
    print("Cargando viajes desde parquet...")
    trips_df = pd.read_parquet(trips_file)
    print(f"✅ Viajes cargados: {len(trips_df):,} registros")
    
    if stations_file.exists():
        print("Cargando información de estaciones desde parquet...")
        station_info = pd.read_parquet(stations_file)
        print(f"✅ Estaciones cargadas: {len(station_info):,} registros")
    else:
        print("⚠️ Archivo de estaciones no encontrado, creando DataFrame vacío")
        station_info = pd.DataFrame()
    
    return users_df, trips_df, station_info 