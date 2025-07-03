#!/usr/bin/env python3
"""
Plot R² por estación usando el CSV generado por predict_gnn_to_csv.py.

El script calcula R² para cada estación (comparando `arrivals` vs `pred_arrivals`) y
muestra un gráfico con el R² individual.  El eje X son los `station_id`
ordenados ascendentemente (0..max).  El eje Y es el valor de R².

Ejemplo de uso:
    python scripts/plot_r2_por_estacion.py \
        --csv predictions/test_predictions.csv \
        --output plots/r2_por_estacion.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.metrics import r2_score


def calcular_r2_por_estacion(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.Series:
    """Devuelve una serie con R² por station_id."""
    r2_por_estacion = {}
    for station_id, grupo in df.groupby('station_id'):
        if len(grupo) < 2:
            # R² indefinido con un solo punto → NaN
            r2_por_estacion[station_id] = np.nan
        else:
            r2_por_estacion[station_id] = r2_score(grupo[target_col], grupo[pred_col])
    return pd.Series(r2_por_estacion).sort_index()


def main():
    parser = argparse.ArgumentParser(description='Plot R² por estación a partir de CSV de predicciones')
    parser.add_argument('--csv', required=True, help='Ruta al CSV con columnas datetime, station_id, pred_arrivals, arrivals')
    parser.add_argument('--output', default=None, help='Ruta para guardar el gráfico (PNG). Si se omite, solo se muestra.')
    parser.add_argument('--pred_col', default='pred_arrivals', help='Nombre de la columna de predicciones')
    parser.add_argument('--target_col', default='arrivals', help='Nombre de la columna target real')
    parser.add_argument('--dpi', type=int, default=150, help='Resolución del gráfico')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV no encontrado: {csv_path}')

    df = pd.read_csv(csv_path)
    if args.pred_col not in df.columns or args.target_col not in df.columns:
        raise ValueError(f'El CSV debe contener las columnas "{args.pred_col}" y "{args.target_col}"')

    # calcular R² por estación
    serie_r2 = calcular_r2_por_estacion(df, args.target_col, args.pred_col)

    # plot
    plt.figure(figsize=(12, 6), dpi=args.dpi)
    plt.scatter(serie_r2.index, serie_r2.values, color='steelblue')
    plt.plot(serie_r2.index, serie_r2.values, color='steelblue', linewidth=1)
    plt.xlabel('Estación', fontsize=16)
    plt.ylabel('R²', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=args.dpi)
        print(f'Gráfico guardado en {output_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main() 