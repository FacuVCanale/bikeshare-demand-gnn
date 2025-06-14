import pandas as pd
import numpy as np
from collections import defaultdict
import json

def extract_first_3_digits_station_id(station_id):
    """Función que replica la lógica del pipeline"""
    if pd.isna(station_id):
        return station_id
    # Remove BAEcobici suffix if present
    station_str = str(station_id).replace('BAEcobici', '')
    station_str = str(int(float(station_str)))
    if len(station_str) >= 3:
        return int(station_str[:3])
    else:
        return int(station_str)

def clean_coordinate_pair(latitude, longitude):
    """Limpia y valida pares de coordenadas"""
    lat_str = str(latitude).strip()
    lon_str = str(longitude).strip()
    
    # Detectar si hay coordenadas malformadas (formato: "lat,lon")
    if ',' in lat_str and ',' not in lon_str:
        # Caso: latitud contiene "lat,lon" y longitud es solo un valor
        parts = lat_str.split(',')
        if len(parts) == 2:
            clean_lat = parts[0].strip()
            clean_lon = parts[1].strip()
            return f"({clean_lat}, {clean_lon})", "FIXED_FROM_MALFORMED"
    
    # Detectar si longitud y latitud están intercambiadas o duplicadas
    if lat_str == lon_str:
        return f"({lat_str}, {lon_str})", "DUPLICATE_COORDS"
    
    # Formato normal
    return f"({lat_str}, {lon_str})", "NORMAL"

print("=== ANÁLISIS DE CONSISTENCIA DE ESTACIONES ===")
print("Verificando coordenadas malformadas y duplicadas...\n")

# Diccionarios para almacenar información
station_data = defaultdict(lambda: {
    'total_trips': 0,
    'years_data': defaultdict(lambda: {
        'names': set(),
        'coord_pairs': set(),
        'coord_issues': set(),  # Tipos de problemas detectados
        'trip_count': 0
    }),
    'original_ids': set()
})

years = [2020, 2021, 2022, 2023, 2024]

# Recopilar datos de todos los años
for year in years:
    try:
        print(f"Procesando {year}...")
        df = pd.read_csv(f'data/raw/trips/trips_{year}.csv')
        
        # Procesar estaciones de origen
        for _, row in df[['id_estacion_origen', 'nombre_estacion_origen', 'long_estacion_origen', 'lat_estacion_origen']].drop_duplicates().iterrows():
            original_id = row['id_estacion_origen']
            name = row['nombre_estacion_origen']
            longitude = row['long_estacion_origen']
            latitude = row['lat_estacion_origen']
            
            # Aplicar la función de limpieza
            clean_id = extract_first_3_digits_station_id(original_id)
            
            if not pd.isna(clean_id):
                # Contar trips para esta estación en este año
                trips_count = df[df['id_estacion_origen'] == original_id].shape[0]
                
                # Limpiar coordenadas y detectar problemas
                coord_pair, issue_type = clean_coordinate_pair(latitude, longitude)
                
                station_data[clean_id]['total_trips'] += trips_count
                station_data[clean_id]['years_data'][year]['trip_count'] += trips_count
                station_data[clean_id]['years_data'][year]['names'].add(str(name))
                station_data[clean_id]['years_data'][year]['coord_pairs'].add(coord_pair)
                station_data[clean_id]['years_data'][year]['coord_issues'].add(issue_type)
                station_data[clean_id]['original_ids'].add(str(original_id))
        
        # Procesar estaciones de destino
        for _, row in df[['id_estacion_destino', 'nombre_estacion_destino', 'long_estacion_destino', 'lat_estacion_destino']].drop_duplicates().iterrows():
            original_id = row['id_estacion_destino']
            name = row['nombre_estacion_destino']
            longitude = row['long_estacion_destino']
            latitude = row['lat_estacion_destino']
            
            # Aplicar la función de limpieza
            clean_id = extract_first_3_digits_station_id(original_id)
            
            if not pd.isna(clean_id):
                # Contar trips para esta estación en este año (como destino)
                trips_count = df[df['id_estacion_destino'] == original_id].shape[0]
                
                # Limpiar coordenadas y detectar problemas
                coord_pair, issue_type = clean_coordinate_pair(latitude, longitude)
                
                # Solo agregar si no se ha procesado ya como origen
                if clean_id not in station_data or year not in station_data[clean_id]['years_data']:
                    station_data[clean_id]['total_trips'] += trips_count
                    station_data[clean_id]['years_data'][year]['trip_count'] += trips_count
                
                station_data[clean_id]['years_data'][year]['names'].add(str(name))
                station_data[clean_id]['years_data'][year]['coord_pairs'].add(coord_pair)
                station_data[clean_id]['years_data'][year]['coord_issues'].add(issue_type)
                station_data[clean_id]['original_ids'].add(str(original_id))
                
    except Exception as e:
        print(f"Error procesando {year}: {e}")

print("\nCreando DataFrame de resultados...")

# Crear DataFrame de resultados
results = []
for station_id, data in station_data.items():
    # Obtener todos los años activos
    years_active = sorted(list(data['years_data'].keys()))
    
    # Recopilar información por año
    years_info = {}
    year_coord_dict = {}  # Diccionario año -> coordenada principal
    all_names = set()
    all_coord_pairs = set()
    all_issues = set()
    
    for year in years_active:
        year_data = data['years_data'][year]
        
        # Seleccionar la coordenada principal (la más "limpia")
        coord_pairs = list(year_data['coord_pairs'])
        issues = list(year_data['coord_issues'])
        
        # Priorizar coordenadas normales sobre las problemáticas
        main_coord = coord_pairs[0]  # Por defecto
        if len(coord_pairs) > 1:
            # Si hay múltiples, preferir las normales
            normal_coords = [cp for cp, issue in zip(coord_pairs, issues) if 'NORMAL' in str(issue)]
            if normal_coords:
                main_coord = normal_coords[0]
        
        year_coord_dict[year] = main_coord
        
        years_info[year] = {
            'names': list(year_data['names']),
            'coord_pairs': coord_pairs,
            'coord_issues': list(year_data['coord_issues']),
            'main_coord': main_coord,
            'trip_count': year_data['trip_count']
        }
        
        all_names.update(year_data['names'])
        all_coord_pairs.update(year_data['coord_pairs'])
        all_issues.update(year_data['coord_issues'])
    
    # Detectar inconsistencias
    multiple_names = len(all_names) > 1
    multiple_coords = len(all_coord_pairs) > 1
    has_coord_issues = any(issue != 'NORMAL' for issue in all_issues)
    
    results.append({
        'station_id_clean': station_id,
        'total_trips': data['total_trips'],
        'years_active': years_active,
        'num_years': len(years_active),
        'year_coord_dict': year_coord_dict,  # Diccionario año -> coordenada
        'years_info': years_info,
        'all_names': list(all_names),
        'all_coord_pairs': list(all_coord_pairs),
        'all_coord_issues': list(all_issues),
        'num_different_names': len(all_names),
        'num_different_coords': len(all_coord_pairs),
        'original_ids': list(data['original_ids']),
        'num_original_ids': len(data['original_ids']),
        'has_multiple_names': multiple_names,
        'has_multiple_coords': multiple_coords,
        'has_coord_issues': has_coord_issues
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('total_trips', ascending=False)

print(f"\n=== RESUMEN GENERAL ===")
print(f"Total estaciones únicas (después de limpieza): {len(df_results)}")
print(f"Estaciones con múltiples nombres: {len(df_results[df_results['has_multiple_names']])}")
print(f"Estaciones con múltiples coordenadas: {len(df_results[df_results['has_multiple_coords']])}")
print(f"Estaciones con problemas de coordenadas: {len(df_results[df_results['has_coord_issues']])}")
print(f"Estaciones con múltiples IDs originales: {len(df_results[df_results['num_original_ids'] > 1])}")

# Análisis de problemas de coordenadas por año
print(f"\n=== ANÁLISIS DE PROBLEMAS DE COORDENADAS POR AÑO ===")
issue_stats = {}
for year in years:
    issue_count = 0
    for _, row in df_results.iterrows():
        if year in row['years_info']:
            year_issues = row['years_info'][year]['coord_issues']
            if any(issue != 'NORMAL' for issue in year_issues):
                issue_count += 1
    issue_stats[year] = issue_count
    print(f"{year}: {issue_count} estaciones con problemas de coordenadas")

# Mostrar estaciones problemáticas
print(f"\n=== ESTACIONES CON PROBLEMAS DE COORDENADAS ===")
problematic_coords = df_results[df_results['has_coord_issues']].head(10)
for _, row in problematic_coords.iterrows():
    print(f"\nID {row['station_id_clean']} - {row['all_names'][0] if row['all_names'] else 'Sin nombre'}:")
    print(f"  Problemas detectados: {row['all_coord_issues']}")
    print(f"  Diccionario año-coordenada:")
    for year, coord in row['year_coord_dict'].items():
        issues = row['years_info'][year]['coord_issues']
        issue_str = f" [{', '.join(issues)}]" if any(i != 'NORMAL' for i in issues) else ""
        print(f"    {year}: {coord}{issue_str}")

# Crear un DataFrame expandido para análisis detallado
detailed_results = []
for _, row in df_results.iterrows():
    station_id = row['station_id_clean']
    for year, info in row['years_info'].items():
        detailed_results.append({
            'station_id_clean': station_id,
            'year': year,
            'names': '; '.join(info['names']),
            'main_coord': info['main_coord'],
            'all_coord_pairs': '; '.join(info['coord_pairs']),
            'coord_issues': '; '.join(info['coord_issues']),
            'trip_count': info['trip_count'],
            'num_names_this_year': len(info['names']),
            'num_coords_this_year': len(info['coord_pairs']),
            'has_issues_this_year': any(issue != 'NORMAL' for issue in info['coord_issues'])
        })

df_detailed = pd.DataFrame(detailed_results)

# Crear DataFrame de resumen con diccionario año-coordenada
summary_results = []
for _, row in df_results.iterrows():
    # Convertir diccionario a string JSON para CSV
    year_coord_json = json.dumps(row['year_coord_dict'], ensure_ascii=False)
    
    summary_results.append({
        'station_id_clean': row['station_id_clean'],
        'total_trips': row['total_trips'],
        'years_active': '; '.join(map(str, row['years_active'])),
        'num_years': row['num_years'],
        'year_coord_dict': year_coord_json,
        'all_names': '; '.join(row['all_names']),
        'num_different_names': row['num_different_names'],
        'num_different_coords': row['num_different_coords'],
        'has_multiple_names': row['has_multiple_names'],
        'has_multiple_coords': row['has_multiple_coords'],
        'has_coord_issues': row['has_coord_issues'],
        'coord_issues_summary': '; '.join(row['all_coord_issues'])
    })

df_summary = pd.DataFrame(summary_results)

# Guardar resultados
df_summary.to_csv('station_consistency_summary.csv', index=False)
df_detailed.to_csv('station_consistency_detailed.csv', index=False)

print(f"\nResultados guardados en:")
print(f"  - station_consistency_summary.csv (resumen con diccionario año-coordenada)")
print(f"  - station_consistency_detailed.csv (detalle por estación-año)")

# Mostrar top 10 estaciones más activas
print(f"\n=== TOP 10 ESTACIONES MÁS ACTIVAS ===")
top_stations = df_summary[['station_id_clean', 'total_trips', 'num_years', 'has_coord_issues', 'coord_issues_summary']].head(10)
print(top_stations.to_string(index=False))