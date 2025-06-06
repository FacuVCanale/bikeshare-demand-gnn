import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las visualizaciones
plt.style.use('default')
sns.set_palette("husl")


def analyze_data_graphically(users_df: pd.DataFrame, trips_df: pd.DataFrame, 
                           save_plots: bool = False, output_dir: Optional[Path] = None) -> None:
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
    
    # ====================================
    # 1. ANÁLISIS DE USUARIOS
    # ====================================
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('👥 ANÁLISIS DE USUARIOS ECOBICI', fontsize=16, fontweight='bold')
    
    # 1.1 Distribución de edades
    if 'edad_usuario' in users_df.columns:
        users_df_clean = users_df.copy()
        users_df_clean['edad_usuario'] = pd.to_numeric(users_df_clean['edad_usuario'], errors='coerce')
        edad_valid = users_df_clean['edad_usuario'].dropna()
        edad_valid = edad_valid[(edad_valid >= 15) & (edad_valid <= 80)]  # Filtrar edades razonables
        
        axes[0, 0].hist(edad_valid, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribución de Edades')
        axes[0, 0].set_xlabel('Edad (años)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(edad_valid.mean(), color='red', linestyle='--', 
                          label=f'Media: {edad_valid.mean():.1f} años')
        axes[0, 0].legend()
    
    # 1.2 Distribución de géneros
    if 'genero_usuario' in users_df.columns:
        genero_counts = users_df['genero_usuario'].value_counts()
        axes[0, 1].pie(genero_counts.values, labels=genero_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen'])
        axes[0, 1].set_title('Distribución por Género')
    
    # 1.3 Registros por año
    if 'fecha_alta' in users_df.columns:
        users_df_temp = users_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(users_df_temp['fecha_alta']):
            users_df_temp['fecha_alta'] = pd.to_datetime(users_df_temp['fecha_alta'], errors='coerce')
        
        registros_por_año = users_df_temp['fecha_alta'].dt.year.value_counts().sort_index()
        axes[1, 0].bar(registros_por_año.index, registros_por_año.values, color='lightgreen')
        axes[1, 0].set_title('Registros de Usuarios por Año')
        axes[1, 0].set_xlabel('Año')
        axes[1, 0].set_ylabel('Nuevos Usuarios')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 1.4 Registros por mes
    if 'fecha_alta' in users_df.columns:
        registros_por_mes = users_df_temp['fecha_alta'].dt.month.value_counts().sort_index()
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
    
    # ====================================
    # 2. ANÁLISIS TEMPORAL DE VIAJES
    # ====================================
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🚲 ANÁLISIS TEMPORAL DE VIAJES', fontsize=16, fontweight='bold')
    
    # Preparar datos temporales
    trips_temp = trips_df.copy()
    fecha_col = 'fecha_origen_recorrido' if 'fecha_origen_recorrido' in trips_temp.columns else 'fecha_origen'
    
    if not pd.api.types.is_datetime64_any_dtype(trips_temp[fecha_col]):
        trips_temp[fecha_col] = pd.to_datetime(trips_temp[fecha_col], errors='coerce')
    
    trips_temp = trips_temp.dropna(subset=[fecha_col])
    
    # 2.1 Viajes por hora del día
    viajes_por_hora = trips_temp[fecha_col].dt.hour.value_counts().sort_index()
    axes[0, 0].plot(viajes_por_hora.index, viajes_por_hora.values, marker='o', linewidth=2, color='blue')
    axes[0, 0].set_title('Viajes por Hora del Día')
    axes[0, 0].set_xlabel('Hora')
    axes[0, 0].set_ylabel('Número de Viajes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # 2.2 Viajes por día de la semana
    dias_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    viajes_por_dia = trips_temp[fecha_col].dt.dayofweek.value_counts().sort_index()
    axes[0, 1].bar(range(7), [viajes_por_dia.get(i, 0) for i in range(7)], color='green')
    axes[0, 1].set_title('Viajes por Día de la Semana')
    axes[0, 1].set_xlabel('Día')
    axes[0, 1].set_ylabel('Número de Viajes')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(dias_semana)
    
    # 2.3 Viajes por mes
    viajes_por_mes = trips_temp[fecha_col].dt.month.value_counts().sort_index()
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    axes[1, 0].bar(range(1, 13), [viajes_por_mes.get(i, 0) for i in range(1, 13)], color='purple')
    axes[1, 0].set_title('Viajes por Mes')
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('Número de Viajes')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(meses, rotation=45)
    
    # 2.4 Tendencia por año
    viajes_por_año = trips_temp[fecha_col].dt.year.value_counts().sort_index()
    axes[1, 1].plot(viajes_por_año.index, viajes_por_año.values, marker='o', linewidth=3, color='red')
    axes[1, 1].set_title('Tendencia de Viajes por Año')
    axes[1, 1].set_xlabel('Año')
    axes[1, 1].set_ylabel('Número de Viajes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots and output_dir:
        plt.savefig(output_dir / 'viajes_temporal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ====================================
    # 3. ANÁLISIS DE ESTACIONES
    # ====================================
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🚉 ANÁLISIS DE ESTACIONES', fontsize=16, fontweight='bold')
    
    # 3.1 Top 15 estaciones de origen
    if 'id_estacion_origen' in trips_temp.columns:
        top_origen = trips_temp['id_estacion_origen'].value_counts().head(15)
        axes[0, 0].barh(range(len(top_origen)), top_origen.values, color='lightblue')
        axes[0, 0].set_title('Top 15 Estaciones de Origen')
        axes[0, 0].set_xlabel('Número de Viajes')
        axes[0, 0].set_yticks(range(len(top_origen)))
        axes[0, 0].set_yticklabels([str(x)[:20] for x in top_origen.index], fontsize=8)
    
    # 3.2 Top 15 estaciones de destino
    if 'id_estacion_destino' in trips_temp.columns:
        top_destino = trips_temp['id_estacion_destino'].value_counts().head(15)
        axes[0, 1].barh(range(len(top_destino)), top_destino.values, color='lightcoral')
        axes[0, 1].set_title('Top 15 Estaciones de Destino')
        axes[0, 1].set_xlabel('Número de Viajes')
        axes[0, 1].set_yticks(range(len(top_destino)))
        axes[0, 1].set_yticklabels([str(x)[:20] for x in top_destino.index], fontsize=8)
    
    # 3.3 Distribución de viajes por estación
    if 'id_estacion_origen' in trips_temp.columns:
        viajes_por_estacion = trips_temp['id_estacion_origen'].value_counts()
        axes[1, 0].hist(viajes_por_estacion.values, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title('Distribución de Viajes por Estación')
        axes[1, 0].set_xlabel('Número de Viajes')
        axes[1, 0].set_ylabel('Número de Estaciones')
        axes[1, 0].set_yscale('log')
    
    # 3.4 Duración de viajes (si existe la columna)
    if 'duracion_recorrido' in trips_temp.columns:
        trips_temp['duracion_num'] = pd.to_numeric(trips_temp['duracion_recorrido'].astype(str).str.replace(',', ''), errors='coerce')
        duracion_valid = trips_temp['duracion_num'].dropna()
        duracion_valid = duracion_valid[(duracion_valid > 0) & (duracion_valid < 7200)]  # < 2 horas
        
        axes[1, 1].hist(duracion_valid / 60, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_title('Distribución de Duración de Viajes')
        axes[1, 1].set_xlabel('Duración (minutos)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].axvline(duracion_valid.mean() / 60, color='red', linestyle='--', 
                          label=f'Media: {duracion_valid.mean() / 60:.1f} min')
        axes[1, 1].legend()
    
    plt.tight_layout()
    if save_plots and output_dir:
        plt.savefig(output_dir / 'estaciones_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ====================================
    # 4. HEATMAP DE ACTIVIDAD
    # ====================================
    
    # Crear matriz hora vs día de la semana
    trips_temp['hora'] = trips_temp[fecha_col].dt.hour
    trips_temp['dia_semana'] = trips_temp[fecha_col].dt.dayofweek
    
    activity_matrix = trips_temp.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(activity_matrix, 
                xticklabels=range(24),
                yticklabels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                cmap='YlOrRd', 
                cbar_kws={'label': 'Número de Viajes'})
    plt.title('🔥 HEATMAP DE ACTIVIDAD: Viajes por Hora y Día de la Semana', fontsize=14, fontweight='bold')
    plt.xlabel('Hora del Día')
    plt.ylabel('Día de la Semana')
    
    if save_plots and output_dir:
        plt.savefig(output_dir / 'heatmap_actividad.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ====================================
    # 5. ESTADÍSTICAS RESUMEN
    # ====================================
    
    print("\n📊 ESTADÍSTICAS RESUMEN")
    print("=" * 40)
    
    # Usuarios
    print(f"👥 USUARIOS:")
    print(f"   • Total usuarios: {len(users_df):,}")
    if 'edad_usuario' in users_df.columns:
        users_clean = users_df.copy()
        users_clean['edad_usuario'] = pd.to_numeric(users_clean['edad_usuario'], errors='coerce')
        edad_promedio = users_clean['edad_usuario'].mean()
        if not pd.isna(edad_promedio):
            print(f"   • Edad promedio: {edad_promedio:.1f} años")
    
    if 'genero_usuario' in users_df.columns:
        genero_counts = users_df['genero_usuario'].value_counts()
        print(f"   • Distribución de género:")
        for genero, count in genero_counts.items():
            pct = count / len(users_df) * 100
            print(f"     - {genero}: {count:,} ({pct:.1f}%)")
    
    # Viajes
    print(f"\n🚲 VIAJES:")
    print(f"   • Total viajes: {len(trips_df):,}")
    print(f"   • Período: {trips_temp[fecha_col].min()} - {trips_temp[fecha_col].max()}")
    print(f"   • Promedio viajes/día: {len(trips_df) / ((trips_temp[fecha_col].max() - trips_temp[fecha_col].min()).days + 1):,.0f}")
    
    if 'id_estacion_origen' in trips_temp.columns:
        print(f"   • Estaciones únicas (origen): {trips_temp['id_estacion_origen'].nunique():,}")
    if 'id_estacion_destino' in trips_temp.columns:
        print(f"   • Estaciones únicas (destino): {trips_temp['id_estacion_destino'].nunique():,}")
    
    if 'duracion_recorrido' in trips_temp.columns and 'duracion_num' in trips_temp.columns:
        duracion_promedio = trips_temp['duracion_num'].mean()
        if not pd.isna(duracion_promedio):
            print(f"   • Duración promedio: {duracion_promedio / 60:.1f} minutos")
    
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