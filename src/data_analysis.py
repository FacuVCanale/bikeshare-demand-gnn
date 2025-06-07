import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
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
    fig.suptitle('👥 ANÁLISIS DE USUARIOS ECOBICI', fontsize=16, fontweight='bold')
    
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
    fig.suptitle('🚲 ANÁLISIS TEMPORAL DE VIAJES', fontsize=16, fontweight='bold')
    
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
    fig.suptitle('🚉 ANÁLISIS DE ESTACIONES', fontsize=16, fontweight='bold')
    
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

def split_data_until_august_2024(users_df: pd.DataFrame, trips_df: pd.DataFrame, 
                                train_end_date: str = "2023-06-30",
                                val_end_date: str = "2023-12-31",
                                test_end_date: str = "2024-08-31",
                                verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Splitea los datos de usuarios y viajes manteniendo integridad temporal.
    
    RESTRICCIÓN IMPORTANTE: Solo se utilizan datos hasta agosto 2024 inclusive.
    
    Args:
        users_df: DataFrame de usuarios completo
        trips_df: DataFrame de viajes completo  
        train_end_date: Fecha final para entrenamiento (formato YYYY-MM-DD)
        val_end_date: Fecha final para validación (formato YYYY-MM-DD)
        test_end_date: Fecha final para test (formato YYYY-MM-DD, máximo 2024-08-31)
        verbose: Si mostrar información del split
        
    Returns:
        Dict con las claves:
        - 'users_train', 'users_val', 'users_test': DataFrames de usuarios
        - 'trips_train', 'trips_val', 'trips_test': DataFrames de viajes
        - 'split_info': Información detallada del split
        
    Raises:
        ValueError: Si test_end_date es posterior a agosto 2024
    """
    
    # Validar restricción temporal
    max_allowed_date = pd.to_datetime("2024-08-31")
    test_date = pd.to_datetime(test_end_date)
    
    if test_date > max_allowed_date:
        raise ValueError(f"La fecha de test no puede ser posterior a agosto 2024. "
                        f"Recibido: {test_end_date}, máximo permitido: 2024-08-31")
    
    if verbose:
        print("🔄 INICIANDO SPLIT TEMPORAL DE DATOS ECOBICI")
        print("=" * 50)
        print(f"📅 Restricción: Solo datos hasta {max_allowed_date.strftime('%B %Y')}")
    
    # =========================================================================
    # PREPARACIÓN DE FECHAS
    # =========================================================================
    
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date) 
    test_end = pd.to_datetime(test_end_date)
    
    # =========================================================================
    # PROCESAMIENTO DE USUARIOS
    # =========================================================================
    
    users_work = users_df.copy()
    
    # Convertir fecha_alta a datetime
    if not pd.api.types.is_datetime64_any_dtype(users_work['fecha_alta']):
        users_work['fecha_alta'] = pd.to_datetime(users_work['fecha_alta'], errors='coerce')
    
    # Filtrar usuarios hasta la fecha máxima permitida
    users_filtered = users_work[
        (users_work['fecha_alta'].notna()) & 
        (users_work['fecha_alta'] <= max_allowed_date)
    ].copy()
    
    # Split de usuarios por fecha de alta
    users_train = users_filtered[users_filtered['fecha_alta'] <= train_end].copy()
    users_val = users_filtered[
        (users_filtered['fecha_alta'] > train_end) & 
        (users_filtered['fecha_alta'] <= val_end)
    ].copy()
    users_test = users_filtered[
        (users_filtered['fecha_alta'] > val_end) & 
        (users_filtered['fecha_alta'] <= test_end)
    ].copy()
    
    # =========================================================================
    # PROCESAMIENTO DE VIAJES  
    # =========================================================================
    
    trips_work = trips_df.copy()
    
    # Determinar columna de fecha principal
    fecha_col = 'fecha_origen_recorrido' if 'fecha_origen_recorrido' in trips_work.columns else 'fecha_origen'
    
    # Convertir fecha a datetime
    if not pd.api.types.is_datetime64_any_dtype(trips_work[fecha_col]):
        trips_work[fecha_col] = pd.to_datetime(trips_work[fecha_col], errors='coerce')
    
    # Filtrar viajes hasta la fecha máxima permitida
    trips_filtered = trips_work[
        (trips_work[fecha_col].notna()) & 
        (trips_work[fecha_col] <= max_allowed_date)
    ].copy()
    
    # Split de viajes por fecha de origen
    trips_train = trips_filtered[trips_filtered[fecha_col] <= train_end].copy()
    trips_val = trips_filtered[
        (trips_filtered[fecha_col] > train_end) & 
        (trips_filtered[fecha_col] <= val_end)
    ].copy()
    trips_test = trips_filtered[
        (trips_filtered[fecha_col] > val_end) & 
        (trips_filtered[fecha_col] <= test_end)
    ].copy()
    
    # =========================================================================
    # INFORMACIÓN DEL SPLIT
    # =========================================================================
    
    split_info = {
        'fechas': {
            'train_periodo': f"Inicio - {train_end_date}",
            'val_periodo': f"{(train_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} - {val_end_date}",
            'test_periodo': f"{(val_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} - {test_end_date}",
            'restriccion_maxima': "2024-08-31"
        },
        'usuarios': {
            'total_original': len(users_df),
            'total_filtrado': len(users_filtered),
            'descartados_por_fecha': len(users_df) - len(users_filtered),
            'train': len(users_train),
            'val': len(users_val), 
            'test': len(users_test)
        },
        'viajes': {
            'total_original': len(trips_df),
            'total_filtrado': len(trips_filtered),
            'descartados_por_fecha': len(trips_df) - len(trips_filtered),
            'train': len(trips_train),
            'val': len(trips_val),
            'test': len(trips_test)
        },
        'rangos_temporales': {
            'usuarios': {
                'train': f"{users_train['fecha_alta'].min()} - {users_train['fecha_alta'].max()}" if len(users_train) > 0 else "Sin datos",
                'val': f"{users_val['fecha_alta'].min()} - {users_val['fecha_alta'].max()}" if len(users_val) > 0 else "Sin datos", 
                'test': f"{users_test['fecha_alta'].min()} - {users_test['fecha_alta'].max()}" if len(users_test) > 0 else "Sin datos"
            },
            'viajes': {
                'train': f"{trips_train[fecha_col].min()} - {trips_train[fecha_col].max()}" if len(trips_train) > 0 else "Sin datos",
                'val': f"{trips_val[fecha_col].min()} - {trips_val[fecha_col].max()}" if len(trips_val) > 0 else "Sin datos",
                'test': f"{trips_test[fecha_col].min()} - {trips_test[fecha_col].max()}" if len(trips_test) > 0 else "Sin datos"
            }
        }
    }
    
    # =========================================================================
    # MOSTRAR INFORMACIÓN (SI VERBOSE=TRUE)
    # =========================================================================
    
    if verbose:
        print("\n📊 RESUMEN DEL SPLIT:")
        print("-" * 30)
        
        print(f"\n👥 USUARIOS:")
        print(f"   • Original: {split_info['usuarios']['total_original']:,}")
        print(f"   • Filtrado (≤ ago 2024): {split_info['usuarios']['total_filtrado']:,}")
        print(f"   • Descartados: {split_info['usuarios']['descartados_por_fecha']:,}")
        print(f"   • Train: {split_info['usuarios']['train']:,} ({split_info['usuarios']['train']/split_info['usuarios']['total_filtrado']*100:.1f}%)")
        print(f"   • Val: {split_info['usuarios']['val']:,} ({split_info['usuarios']['val']/split_info['usuarios']['total_filtrado']*100:.1f}%)")
        print(f"   • Test: {split_info['usuarios']['test']:,} ({split_info['usuarios']['test']/split_info['usuarios']['total_filtrado']*100:.1f}%)")
        
        print(f"\n🚲 VIAJES:")
        print(f"   • Original: {split_info['viajes']['total_original']:,}")
        print(f"   • Filtrado (≤ ago 2024): {split_info['viajes']['total_filtrado']:,}")
        print(f"   • Descartados: {split_info['viajes']['descartados_por_fecha']:,}")
        print(f"   • Train: {split_info['viajes']['train']:,} ({split_info['viajes']['train']/split_info['viajes']['total_filtrado']*100:.1f}%)")
        print(f"   • Val: {split_info['viajes']['val']:,} ({split_info['viajes']['val']/split_info['viajes']['total_filtrado']*100:.1f}%)")
        print(f"   • Test: {split_info['viajes']['test']:,} ({split_info['viajes']['test']/split_info['viajes']['total_filtrado']*100:.1f}%)")
        
        print(f"\n📅 PERÍODOS TEMPORALES:")
        print(f"   • Train: {split_info['fechas']['train_periodo']}")
        print(f"   • Val: {split_info['fechas']['val_periodo']}")
        print(f"   • Test: {split_info['fechas']['test_periodo']}")
        
        print("\n✅ Split completado exitosamente!")
    
    # =========================================================================
    # RETORNAR RESULTADOS
    # =========================================================================
    
    return {
        'users_train': users_train,
        'users_val': users_val, 
        'users_test': users_test,
        'trips_train': trips_train,
        'trips_val': trips_val,
        'trips_test': trips_test,
        'split_info': split_info
    }


# =============================================================================
# FUNCIÓN PRINCIPAL REFACTORIZADA
# =============================================================================

def analyze_data_graphically(users_df: pd.DataFrame, trips_df: pd.DataFrame, 
                           save_plots: bool = False, output_dir: Optional[Path] = None, processed_users: Optional[Dict[str, Any]] = None, processed_trips: Optional[Dict[str, Any]] = None) -> None:
    """
    Realiza un análisis gráfico completo de los datos de usuarios y viajes de EcoBici.
    
    Args:
        users_df: DataFrame con datos de usuarios
        trips_df: DataFrame con datos de viajes
        save_plots: Si guardar los gráficos como archivos
        output_dir: Directorio donde guardar los gráficos
    """
    
    if save_plots and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 INICIANDO ANÁLISIS GRÁFICO DE DATOS ECOBICI")
    print("=" * 60)
    
    # Preprocesar datos
    print("🔄 Preprocesando datos...")

    
    # Generar visualizaciones
    print("📊 Generando análisis de usuarios...")
    plot_user_analysis(processed_users, save_plots, output_dir)
    
    print("📈 Generando análisis temporal...")
    plot_temporal_analysis(processed_trips, save_plots, output_dir)
    
    print("🚉 Generando análisis de estaciones...")
    plot_station_analysis(processed_trips, save_plots, output_dir)
    
    print("🔥 Generando heatmap de actividad...")
    plot_activity_heatmap(processed_trips, save_plots, output_dir)
    
    # Mostrar estadísticas resumen
    print_summary_statistics(users_df, processed_users, trips_df, processed_trips)
    
    print(f"\n✅ Análisis gráfico completado!")
    if save_plots and output_dir:
        print(f"📁 Gráficos guardados en: {output_dir}")



def load_combined_data(combined_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets combinados desde archivos parquet."""
    print("=== CARGANDO DATASETS COMBINADOS DESDE PARQUET ===")
    
    users_file = combined_dir / "users.parquet"
    trips_file = combined_dir / "trips.parquet"
    
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
    
    return users_df, trips_df 