import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any
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
        # choose which duration range to display
        duration_range = 'extended'  # options: 'reasonable', 'extended', 'full'
        
        if duration_range == 'reasonable' and 'duracion_valid' in processed_trips:
            duracion_data = processed_trips['duracion_valid']
            title_suffix = "(Filtrado: 2min - 4h)"
        elif duration_range == 'extended' and 'duracion_extended' in processed_trips:
            duracion_data = processed_trips['duracion_extended'] 
            title_suffix = "(Extendido: 1min - 12h)"
        elif duration_range == 'full' and 'duracion_full' in processed_trips:
            duracion_data = processed_trips['duracion_full']
            title_suffix = "(Completo - hasta 1 semana)"
        else:
            duracion_data = processed_trips['duracion_valid']
            title_suffix = "(Filtrado: 2min - 4h)"
        
        duracion_promedio = processed_trips['duracion_promedio']
        
        # convert to minutes for better readability
        duracion_minutos = duracion_data / 60
        
        # calculate display range based on data characteristics
        if duration_range == 'full':
            # for full range, use 99th percentile to avoid extreme outliers
            display_max = duracion_minutos.quantile(0.99)
            bins = 100
        elif duration_range == 'extended':
            # for extended range, show most of the data but cap at 95th percentile if too spread
            q95 = duracion_minutos.quantile(0.95)
            display_max = min(duracion_minutos.max(), q95 * 1.2)  # allow some extra room
            bins = 60
        else:
            # for reasonable range, show all data with good resolution
            display_max = duracion_minutos.max()
            bins = 60
        
        axes[1, 1].hist(duracion_minutos, bins=bins, edgecolor='black', alpha=0.7, color='orange', range=(0, display_max))
        axes[1, 1].set_title(f'Distribución de Duración de Viajes {title_suffix}')
        axes[1, 1].set_xlabel('Duración (minutos)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].axvline(duracion_promedio / 60, color='red', linestyle='--', 
                          label=f'Media: {duracion_promedio / 60:.1f} min')
        
        # add statistics if available
        if 'duracion_stats' in processed_trips:
            stats = processed_trips['duracion_stats']
            textstr = f'Mediana: {stats["median"]/60:.1f}min\nP95: {stats["q95"]/60:.1f}min\nP99: {stats["q99"]/60:.1f}min'
            axes[1, 1].text(0.75, 0.95, textstr, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # add data info
        data_info = f'N = {len(duracion_data):,} viajes\nRango: {duracion_minutos.min():.1f} - {display_max:.1f} min'
        axes[1, 1].text(0.05, 0.95, data_info, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
        plt.title('HEATMAP DE ACTIVIDAD: Viajes por Hora y Día de la Semana', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Hora del Día')
        plt.ylabel('Día de la Semana')
        
        if save_plots and output_dir:
            plt.savefig(output_dir / 'heatmap_actividad.png', dpi=300, bbox_inches='tight')
        plt.show()


def print_summary_statistics(users_df: pd.DataFrame, processed_users: Dict[str, Any], 
                           trips_df: pd.DataFrame, processed_trips: Dict[str, Any]) -> None:
    """Imprime estadísticas resumen."""
    
    print("\nESTADÍSTICAS RESUMEN")
    print("=" * 40)
    
    # Usuarios
    print(f"USUARIOS:")
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
    print(f"\nVIAJES:")
    print(f"   • Total viajes: {len(trips_df):,}")
    
    if 'trips_clean' in processed_trips and 'fecha_col' in processed_trips:
        trips_clean = processed_trips['trips_clean']
        fecha_col = processed_trips['fecha_col']
        
        print(f"   • Período: {trips_clean[fecha_col].min()} - {trips_clean[fecha_col].max()}")
        days_diff = (trips_clean[fecha_col].max() - trips_clean[fecha_col].min()).days + 1
        print(f"   • Promedio viajes/día: {len(trips_df) / days_diff:,.0f}")
        
        if 'top_origen' in processed_trips:
            print(f"   • Estaciones únicas (origen): {len(trips_df['id_estacion_origen'].unique()):,}")
        
        if 'top_destino' in processed_trips:
            print(f"   • Estaciones únicas (destino): {len(trips_df['id_estacion_destino'].unique()):,}")
        
    
    if 'duracion_promedio' in processed_trips:
        duracion_promedio = processed_trips['duracion_promedio']
        if not pd.isna(duracion_promedio):
            print(f"   • Duración promedio: {duracion_promedio / 60:.1f} minutos")
