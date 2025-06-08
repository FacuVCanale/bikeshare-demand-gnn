import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_duration_issues(trips_df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Analiza problemas específicos con las duraciones de los viajes
    
    Args:
        trips_df: DataFrame de viajes
        verbose: Mostrar información detallada
        
    Returns:
        Diccionario con el análisis de duraciones
    """
    results = {}
    
    if verbose:
        print("🔍 ANÁLISIS DETALLADO DE DURACIONES DE VIAJES")
        print("=" * 50)
    
    # Convertir duracion_recorrido a numérico
    duration_col = 'duracion_recorrido'
    if duration_col in trips_df.columns:
        # Primero, analizar el tipo de datos original
        original_dtype = trips_df[duration_col].dtype
        results['original_dtype'] = str(original_dtype)
        
        if verbose:
            print(f"Tipo de dato original: {original_dtype}")
        
        # Contar valores nulos originales
        null_count = trips_df[duration_col].isnull().sum()
        results['null_count'] = null_count
        
        if verbose:
            print(f"Valores nulos: {null_count:,} ({null_count/len(trips_df)*100:.2f}%)")
        
        # Convertir a string para analizar formato
        duration_str = trips_df[duration_col].dropna().astype(str)
        
        # Analizar formatos encontrados
        sample_values = duration_str.head(20).tolist()
        results['sample_values'] = sample_values
        
        if verbose:
            print(f"\nPrimeros 20 valores de ejemplo:")
            for i, val in enumerate(sample_values):
                print(f"  {i+1:2d}: '{val}' (tipo: {type(val).__name__})")
        
        # Detectar si hay comas decimales
        has_comma_decimals = duration_str.str.contains(',', na=False).any()
        results['has_comma_decimals'] = has_comma_decimals
        
        if verbose and has_comma_decimals:
            print(f"\n⚠️  ENCONTRADAS COMAS DECIMALES")
            comma_examples = duration_str[duration_str.str.contains(',', na=False)].head(10)
            print("Ejemplos con comas:")
            for val in comma_examples:
                print(f"  '{val}'")
        
        # Convertir a numérico (reemplazando comas por puntos)
        duration_clean = duration_str.str.replace(',', '.', regex=False)
        duration_numeric = pd.to_numeric(duration_clean, errors='coerce')
        
        # Contar conversiones fallidas
        conversion_failed = duration_numeric.isnull().sum()
        results['conversion_failed'] = conversion_failed
        
        if verbose:
            print(f"\nConversión a numérico:")
            print(f"  Valores que no pudieron convertirse: {conversion_failed:,}")
        
        # Estadísticas básicas
        duration_valid = duration_numeric.dropna()
        results['total_valid'] = len(duration_valid)
        
        if len(duration_valid) > 0:
            stats = {
                'count': len(duration_valid),
                'mean': duration_valid.mean(),
                'median': duration_valid.median(),
                'min': duration_valid.min(),
                'max': duration_valid.max(),
                'std': duration_valid.std(),
                'q25': duration_valid.quantile(0.25),
                'q75': duration_valid.quantile(0.75),
                'q95': duration_valid.quantile(0.95),
                'q99': duration_valid.quantile(0.99)
            }
            results['stats'] = stats
            
            if verbose:
                print(f"\nEstadísticas de duración (segundos):")
                print(f"  Válidos: {stats['count']:,}")
                print(f"  Media: {stats['mean']:.1f} seg ({stats['mean']/60:.1f} min)")
                print(f"  Mediana: {stats['median']:.1f} seg ({stats['median']/60:.1f} min)")
                print(f"  Min: {stats['min']:.1f} seg")
                print(f"  Max: {stats['max']:.1f} seg ({stats['max']/(60*60*24):.1f} días)")
                print(f"  Std: {stats['std']:.1f} seg")
                print(f"  Q25: {stats['q25']:.1f} seg ({stats['q25']/60:.1f} min)")
                print(f"  Q75: {stats['q75']:.1f} seg ({stats['q75']/60:.1f} min)")
                print(f"  Q95: {stats['q95']:.1f} seg ({stats['q95']/60:.1f} min)")
                print(f"  Q99: {stats['q99']:.1f} seg ({stats['q99']/60:.1f} min)")
            
            # Analizar casos problemáticos
            problems = {}
            
            # Duraciones = 0
            zero_duration = (duration_valid == 0).sum()
            problems['zero_duration'] = zero_duration
            
            # Duraciones muy cortas (< 30 segundos)
            very_short = (duration_valid < 30).sum()
            problems['very_short'] = very_short
            
            # Duraciones muy largas (> 4 horas)
            very_long = (duration_valid > 4*60*60).sum()
            problems['very_long'] = very_long
            
            # Duraciones extremadamente largas (> 24 horas)
            extremely_long = (duration_valid > 24*60*60).sum()
            problems['extremely_long'] = extremely_long
            
            results['problems'] = problems
            
            if verbose:
                print(f"\n⚠️  CASOS PROBLEMÁTICOS:")
                print(f"  Duración = 0: {zero_duration:,} ({zero_duration/len(duration_valid)*100:.2f}%)")
                print(f"  Muy cortas (< 30 seg): {very_short:,} ({very_short/len(duration_valid)*100:.2f}%)")
                print(f"  Muy largas (> 4 horas): {very_long:,} ({very_long/len(duration_valid)*100:.2f}%)")
                print(f"  Extremas (> 24 horas): {extremely_long:,} ({extremely_long/len(duration_valid)*100:.2f}%)")
                
                # Mostrar ejemplos de casos extremos
                if zero_duration > 0:
                    print(f"\n📍 Ejemplos de duraciones = 0:")
                    zero_mask = duration_valid == 0
                    zero_examples = trips_df[trips_df[duration_col].notna()][zero_mask].head(5)
                    for idx, row in zero_examples.iterrows():
                        print(f"  ID: {row['id_recorrido']}, Origen: {row['fecha_origen_recorrido']}, Destino: {row['fecha_destino_recorrido']}")
                
                if extremely_long > 0:
                    print(f"\n📍 Ejemplos de duraciones > 24 horas:")
                    extreme_mask = duration_valid > 24*60*60
                    extreme_examples = trips_df[trips_df[duration_col].notna()][extreme_mask].head(5)
                    for idx, row in extreme_examples.iterrows():
                        duration_days = duration_valid.iloc[idx] / (60*60*24)
                        print(f"  ID: {row['id_recorrido']}, Duración: {duration_days:.1f} días")
            
            # Calcular duración basada en fechas para validar
            if 'fecha_origen_recorrido' in trips_df.columns and 'fecha_destino_recorrido' in trips_df.columns:
                fecha_origen = pd.to_datetime(trips_df['fecha_origen_recorrido'], errors='coerce')
                fecha_destino = pd.to_datetime(trips_df['fecha_destino_recorrido'], errors='coerce')
                
                # Calcular duración real basada en fechas
                duration_calculated = (fecha_destino - fecha_origen).dt.total_seconds()
                duration_calc_valid = duration_calculated.dropna()
                
                results['calculated_duration_stats'] = {
                    'count': len(duration_calc_valid),
                    'mean': duration_calc_valid.mean(),
                    'median': duration_calc_valid.median(),
                    'min': duration_calc_valid.min(),
                    'max': duration_calc_valid.max()
                }
                
                if verbose:
                    print(f"\n🔄 DURACIÓN CALCULADA DESDE FECHAS:")
                    print(f"  Media calculada: {duration_calc_valid.mean():.1f} seg ({duration_calc_valid.mean()/60:.1f} min)")
                    print(f"  Mediana calculada: {duration_calc_valid.median():.1f} seg")
                    print(f"  Min calculada: {duration_calc_valid.min():.1f} seg")
                    print(f"  Max calculada: {duration_calc_valid.max():.1f} seg ({duration_calc_valid.max()/(60*60*24):.1f} días)")
                
                # Comparar duración registrada vs calculada
                comparison_mask = duration_valid.notna() & duration_calc_valid.notna()
                if comparison_mask.sum() > 0:
                    # Obtener índices válidos para ambas series
                    valid_indices = trips_df.index[trips_df[duration_col].notna()].intersection(
                        trips_df.index[duration_calculated.notna()]
                    )
                    
                    if len(valid_indices) > 0:
                        duration_recorded = duration_valid.reindex(valid_indices).dropna()
                        duration_calculated_aligned = duration_calc_valid.reindex(valid_indices).dropna()
                        
                        # Encontrar discrepancias significativas
                        diff = abs(duration_recorded - duration_calculated_aligned)
                        large_diff = diff > 60  # diferencias > 1 minuto
                        
                        results['discrepancies'] = {
                            'count': large_diff.sum(),
                            'percentage': large_diff.sum() / len(diff) * 100 if len(diff) > 0 else 0
                        }
                        
                        if verbose and large_diff.sum() > 0:
                            print(f"\n⚠️  DISCREPANCIAS ENCONTRADAS:")
                            print(f"  Viajes con diferencia > 1 min: {large_diff.sum():,} ({large_diff.sum()/len(diff)*100:.2f}%)")
                            
                            # Mostrar ejemplos de discrepancias
                            print(f"\n📍 Ejemplos de discrepancias:")
                            discrepant_indices = diff[large_diff].head(5).index
                            for idx in discrepant_indices:
                                recorded = duration_recorded.loc[idx]
                                calculated = duration_calculated_aligned.loc[idx]
                                difference = abs(recorded - calculated)
                                print(f"  ID: {trips_df.loc[idx, 'id_recorrido']}")
                                print(f"    Registrada: {recorded:.1f} seg ({recorded/60:.1f} min)")
                                print(f"    Calculada: {calculated:.1f} seg ({calculated/60:.1f} min)")
                                print(f"    Diferencia: {difference:.1f} seg ({difference/60:.1f} min)")
    
    return results


def plot_duration_analysis(results: Dict[str, Any], save_path: Optional[Path] = None):
    """
    Crea visualizaciones del análisis de duraciones
    """
    if 'stats' not in results:
        print("No hay estadísticas válidas para graficar")
        return
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ANÁLISIS DE DURACIONES DE VIAJES', fontsize=16, fontweight='bold')
    
    stats = results['stats']
    
    # Subplot 1: Estadísticas básicas
    ax1 = axes[0, 0]
    metrics = ['Media', 'Mediana', 'Q75', 'Q95']
    values = [stats['mean']/60, stats['median']/60, stats['q75']/60, stats['q95']/60]
    ax1.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('Estadísticas de Duración')
    ax1.set_ylabel('Minutos')
    ax1.tick_params(axis='x', rotation=45)
    
    # Subplot 2: Problemas identificados
    if 'problems' in results:
        ax2 = axes[0, 1]
        problems = results['problems']
        prob_names = ['Duración = 0', 'Muy cortas\n(< 30s)', 'Muy largas\n(> 4h)', 'Extremas\n(> 24h)']
        prob_values = [problems['zero_duration'], problems['very_short'], 
                      problems['very_long'], problems['extremely_long']]
        
        bars = ax2.bar(prob_names, prob_values, color=['red', 'orange', 'yellow', 'darkred'])
        ax2.set_title('Casos Problemáticos')
        ax2.set_ylabel('Número de Casos')
        ax2.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, prob_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom')
    
    # Subplot 3: Comparación con duración calculada
    if 'calculated_duration_stats' in results:
        ax3 = axes[1, 0]
        calc_stats = results['calculated_duration_stats']
        
        comparison_data = {
            'Registrada': [stats['mean']/60, stats['median']/60],
            'Calculada': [calc_stats['mean']/60, calc_stats['median']/60]
        }
        
        x = ['Media', 'Mediana']
        width = 0.35
        x_pos = np.arange(len(x))
        
        ax3.bar(x_pos - width/2, comparison_data['Registrada'], width, 
               label='Registrada', color='blue', alpha=0.7)
        ax3.bar(x_pos + width/2, comparison_data['Calculada'], width,
               label='Calculada', color='red', alpha=0.7)
        
        ax3.set_title('Duración: Registrada vs Calculada')
        ax3.set_ylabel('Minutos')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x)
        ax3.legend()
    
    # Subplot 4: Información de discrepancias
    if 'discrepancies' in results:
        ax4 = axes[1, 1]
        disc = results['discrepancies']
        
        # Gráfico de pie para mostrar proporción de discrepancias
        labels = ['Sin discrepancias', 'Con discrepancias']
        sizes = [100 - disc['percentage'], disc['percentage']]
        colors = ['lightgreen', 'red']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Discrepancias > 1 minuto')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def main():
    """Función principal para probar el análisis"""
    from data_pipeline import load_all_raw_data
    
    # Cargar datos
    raw_data_dir = Path('../data/raw')
    users_df, trips_df = load_all_raw_data(raw_data_dir, verbose=False)
    
    print(f"Datos cargados: {len(users_df):,} usuarios, {len(trips_df):,} viajes")
    
    # Analizar duraciones
    results = analyze_duration_issues(trips_df, verbose=True)
    
    # Crear visualizaciones
    plot_duration_analysis(results, save_path=Path('../reports/duration_analysis.png'))
    
    return results


if __name__ == "__main__":
    results = main() 