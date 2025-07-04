#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un heatmap 4K de la densidad de viajes Ecobici por comuna
en la Ciudad de Buenos Aires.

Ejecutar:
    python heatmap_trips_por_comuna.py \
        --trips  ./data/processed/trips_with_weather.parquet \
        --shp    ./data/comunas/comunas.shp \
        --out    ./heatmap_trips_comuna_4k.png
"""

import argparse
from pathlib import Path

import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Basemap support -------------------------------------------------------
# Se intenta importar `contextily` para añadir un mapa de fondo. En caso de
# no estar disponible, el script seguirá funcionando pero mostrará una
# advertencia al usuario.
try:
    import contextily as ctx  # type: ignore
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("Advertencia: el paquete 'contextily' no está instalado; el mapa de fondo no se añadirá.")

# ---------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Heatmap de viajes por comuna (4K)")
parser.add_argument("--trips", type=Path, required=True, help="Archivo Parquet de viajes")
parser.add_argument("--shp", type=Path, required=True, help="Shapefile de comunas")
parser.add_argument("--out", type=Path, default=Path("heatmap_trips_comuna_4k.png"),
                    help="Ruta de salida para la imagen PNG 4K")
parser.add_argument("--metric", choices=["trips", "idle_hours"], default="trips",
                    help="trips: densidad de viajes; idle_hours: horas sin viajes por día")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Carpeta de depuración para imágenes intermedias
# ---------------------------------------------------------------------
images_dir = Path("images")
images_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Cargar datos
# ---------------------------------------------------------------------
print("Leyendo datos…")
df = pl.read_parquet(args.trips)
gdf_comunas = gpd.read_file(args.shp)

# ---------------------------------------------------------------------
# 1b. Informar valores min y max de columnas que comienzan con "fecha_"
# ---------------------------------------------------------------------
fecha_cols = [c for c in df.columns if c.startswith("fecha_")]

if fecha_cols:
    print("\nResumen de columnas fecha_* (min / max):")
    for col in fecha_cols:
        try:
            col_min = df[col].min()
            col_max = df[col].max()
            print(f"  {col}:  min = {col_min},   max = {col_max}")
        except Exception as e:
            print(f"  {col}: error al calcular min/max → {e}")
else:
    print("No se encontraron columnas que comiencen con 'fecha_'.")

# ---------------------------------------------------------------------
# 2. Preparar coordenadas (origen + destino)
# ---------------------------------------------------------------------
print("Procesando puntos de origen y destino…")

if args.metric == "trips":
    # ------------------------------ Viajes ---------------------------
    origins = (
        df
        .select([
            pl.col("lat_estacion_origen").alias("lat"),
            pl.col("long_estacion_origen").alias("lon")
        ])
        .drop_nulls()
    )

    destinations = (
        df
        .select([
            pl.col("lat_estacion_destino").alias("lat"),
            pl.col("long_estacion_destino").alias("lon")
        ])
        .drop_nulls()
    )

    all_coords = pl.concat([origins, destinations])

    location_counts = (
        all_coords
        .group_by(["lat", "lon"])
        .agg(pl.len().alias("trip_count"))
    )

elif args.metric == "idle_hours":
    # ----------------------- Horas sin viajes ------------------------
    print("Calculando horas sin viajes por día…")

    # Determinar columna de fecha
    datetime_candidates = [
        "fecha_origen_recorrido",
        "fecha_origen",
        "fecha_retiro",
        "fecha_inicio",
        "fecha"
    ]
    datetime_col = next((c for c in datetime_candidates if c in df.columns), None)

    if datetime_col is None:
        raise ValueError("No se encontró columna de fecha/hora esperada en el DataFrame.")

    # Seleccionar coordenadas y hora
    coords_time = (
        df
        .select([
            pl.col("lat_estacion_origen").alias("lat"),
            pl.col("long_estacion_origen").alias("lon"),
            pl.col(datetime_col).alias("dt")
        ])
        .drop_nulls()
        .with_columns([
            pl.col("dt").str.to_datetime(strict=False).alias("dt")
        ])
        .with_columns([
            pl.col("dt").dt.date().alias("date"),
            pl.col("dt").dt.hour().alias("hour")
        ])
        .select(["lat", "lon", "date", "hour"])
    )

    # Horas sin viajes por día (24 - horas con viajes)
    idle_per_day = (
        coords_time
        .group_by(["lat", "lon", "date"])
        .agg(pl.n_unique("hour").alias("hrs_with_trips"))
        .with_columns((24 - pl.col("hrs_with_trips")).alias("idle_hours"))
    )

    location_counts = (
        idle_per_day
        .group_by(["lat", "lon"])
        .agg(pl.mean("idle_hours").alias("trip_count"))
    )

    # -------------------------------------------------------------
    # Información adicional para el usuario
    # -------------------------------------------------------------
    stats = idle_per_day.select([
        pl.mean("idle_hours").alias("media_horas_sin_viajes"),
        pl.median("idle_hours").alias("mediana_horas_sin_viajes"),
        pl.min("idle_hours").alias("min_horas_sin_viajes"),
        pl.max("idle_hours").alias("max_horas_sin_viajes"),
    ])
    media, mediana, minimo, maximo = stats.row(0)

    print("\nEstadísticas globales de horas sin viajes por estación-día:")
    print(f"  Media   : {media:5.2f} h")
    print(f"  Mediana : {mediana:5.2f} h")
    print(f"  Mínimo  : {minimo:5.2f} h")
    print(f"  Máximo  : {maximo:5.2f} h")

    # Distribución rápida (percentiles)
    percentiles = idle_per_day.select([
        pl.quantile("idle_hours", 0.25).alias("P25"),
        pl.quantile("idle_hours", 0.75).alias("P75"),
    ])
    p25, p75 = percentiles.row(0)
    print(f"  P25     : {p25:5.2f} h")
    print(f"  P75     : {p75:5.2f} h\n")
else:
    raise ValueError("Métrica no reconocida")

print(f"→ {len(location_counts)} ubicaciones únicas detectadas")

# ---------------------------------------------------------------------
# 3. Spatial join: puntos → comuna
# ---------------------------------------------------------------------
coords_gdf = gpd.GeoDataFrame(
    location_counts.to_pandas(),
    geometry=[Point(lon, lat) for lat, lon in zip(location_counts['lat'],
                                                  location_counts['lon'])],
    crs="EPSG:4326",
)

if coords_gdf.crs != gdf_comunas.crs:
    coords_gdf = coords_gdf.to_crs(gdf_comunas.crs)

coords_with_comuna = gpd.sjoin(coords_gdf, gdf_comunas, how="left", predicate="within")

# ---------------------------------------------------------------------
# 4. Agregación por comuna
# ---------------------------------------------------------------------
trips_by_comuna = (
    coords_with_comuna
    .groupby("comuna", as_index=False)["trip_count"]
    .sum()
    .rename(columns={"trip_count": "total_trips"})
)

print("Resumen por comuna:")
print(trips_by_comuna.sort_values("total_trips", ascending=False).head(10))

# ---------------------------------------------------------------------
# 5. Preparar GeoDataFrame final para graficar
# ---------------------------------------------------------------------
gdf_plot = gdf_comunas.merge(trips_by_comuna, on="comuna", how="left")
gdf_plot["total_trips"] = gdf_plot["total_trips"].fillna(0)

# ---------------------------------------------------------------------
# 6. Configurar lienzo 4K y graficar
# ---------------------------------------------------------------------
dpi = 200                # 200 ppp  → 19.2×19.2 in ≈ 3840×3840 px (cuadrado)
fig, ax = plt.subplots(figsize=(19.2, 19.2), dpi=dpi)

# ---------------------------------------------------------------------
# 6. Generar heatmap de densidad estilo "blur" (similar a Folium) y añadir mapa base
# ---------------------------------------------------------------------

# 6.1 Reproyectar coordenadas y comunas a Web Mercator
coords_mercator = coords_gdf.to_crs(epsg=3857)
gdf_comunas_mercator = gdf_comunas.to_crs(epsg=3857)

# 6.2 Preparar datos para grilla regular y suavizado Gaussian
from matplotlib.colors import LinearSegmentedColormap

# Matriz de conteos en grilla
x = coords_mercator.geometry.x.values
y = coords_mercator.geometry.y.values
weights = coords_mercator["trip_count"].values

# Extensión del mapa (bounding box) y resolución de la grilla
minx, miny, maxx, maxy = gdf_comunas_mercator.total_bounds

# Elegimos resolución de grilla acorde a la salida 4K (3840 px ancho).
GRID_SIZE = 3840
ratio = (maxy - miny) / (maxx - minx)
ny = int(GRID_SIZE * ratio)

heatmap, xedges, yedges = np.histogram2d(
    y,  # first dim corresponds to rows
    x,  # second dim columns
    bins=[ny, GRID_SIZE],
    range=[[miny, maxy], [minx, maxx]],
    weights=weights,
)

# Suavizado con filtro Gaussiano si SciPy está disponible
try:
    from scipy.ndimage import gaussian_filter  # type: ignore

    heatmap_blurred = gaussian_filter(heatmap, sigma=70)
except ImportError:
    print("Advertencia: SciPy no está instalado; el heatmap no se suavizará. Instale 'scipy' para un resultado más suave.")
    heatmap_blurred = heatmap

# Normalizar para rango 0–1
heatmap_norm = heatmap_blurred / (heatmap_blurred.max() if heatmap_blurred.max() > 0 else 1)

# Colormap base (sin transparencia)
if args.metric == "idle_hours":
    folium_gradient = plt.get_cmap("Purples")
else:
    folium_gradient = LinearSegmentedColormap.from_list(
        "folium_grad",
        ["blue", "lime", "orange", "red"],
        N=256,
    )

# Crear versión con transparencia en los valores bajos
from matplotlib.colors import ListedColormap

cmap_vals = folium_gradient(np.linspace(0, 1, 256))
# Hacemos que ~primer cuarto sea transparente
cmap_vals[:64, 3] = np.linspace(0, 1, 64)
transparent_cmap = ListedColormap(cmap_vals)

# ---------------------------------------------------------------------
# (Depuración) Guardar la matriz de calor sola para verificar contenido
# ---------------------------------------------------------------------
debug_heatmap_path = images_dir / "heatmap_array.png"
plt.figure(figsize=(8, 6))
plt.imshow(np.flipud(heatmap_norm), cmap=folium_gradient)
# plt.title("Heatmap (array) – depuración")
plt.axis("off")
plt.tight_layout()
plt.savefig(debug_heatmap_path, dpi=150)
plt.close()
print(f"[Debug] Imagen guardada: {debug_heatmap_path.resolve()}")

# 6.3 Definir extensión y dibujar basemap (zorder bajo)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
# ax.set_aspect('equal')  # Comentado para que el plot use toda la imagen rectangular

# Ocultar ejes desde el principio para todas las capturas
ax.set_axis_off()

if HAS_CTX:
    ctx.add_basemap(
        ax,
        crs="EPSG:3857",
        source=ctx.providers.CartoDB.Voyager,
        zoom=13,
        reset_extent=False,  # Mantener límites definidos
        alpha=1.0,
        zorder=1,
    )

# (Depuración) Basemap solo
fig.savefig(images_dir / "step1_basemap.png", dpi=dpi, bbox_inches="tight")

# 6.4 Dibujar heatmap difuso con imshow
im = ax.imshow(
    np.flipud(heatmap_norm),  # imshow origin lower-> flip
    extent=(minx, maxx, miny, maxy),
    cmap=transparent_cmap,
    alpha=0.8,
    interpolation="bilinear",
    zorder=2,
)

# (Depuración) Basemap + heatmap
fig.savefig(images_dir / "step2_heatmap_overlay.png", dpi=dpi, bbox_inches="tight")

# 6.45 Rellenar comunas con gris muy claro y alta transparencia
gdf_comunas_mercator.plot(
    ax=ax,
    facecolor="#9d9d9d",  # light gray
    edgecolor="none",
    alpha=0.15,  # very transparent
    zorder=2.5,
)

# 6.5 Superponer bordes de comunas
gdf_comunas_mercator.boundary.plot(ax=ax, linewidth=1.0, color="black", alpha=0.8, zorder=3)

# 6.6 Añadir etiquetas con el nombre/número de la comuna en su centroide
for _, row in gdf_comunas_mercator.iterrows():
    if row.geometry.is_empty or row.geometry.centroid.is_empty:
        continue
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    # Intenta encontrar una columna de nombre
    label = (
        str(row.get("comuna"))
        if "comuna" in row else str(row.get("COMUNA", ""))
    )
    ax.text(
        x,
        y,
        label,
        fontsize=36,
        ha="center",
        va="center",
        color="black",
        zorder=4,
        path_effects=[pe.withStroke(linewidth=6, foreground="white")],
    )

# (Depuración) Basemap + heatmap + comunas + labels
fig.savefig(images_dir / "step4_with_labels.png", dpi=dpi, bbox_inches="tight")

# 6.7 Añadir barra de color superpuesta (vertical a la derecha) con mayor tamaño
cax = ax.inset_axes([0.90, 0.10, 0.05, 0.80])  # [x0, y0, width, height] fraction of axis
cb = fig.colorbar(im, cax=cax, orientation="vertical")
cb.ax.tick_params(labelsize=32)
cb.outline.set_visible(False)

# Eliminar márgenes para imagen completa
fig.subplots_adjust(0, 0, 1, 1)

# ---------------------------------------------------------------------
# 7. Guardar imagen
# ---------------------------------------------------------------------
plt.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=dpi)
plt.close(fig)

print(f"\nImagen generada: {args.out} ({args.out.resolve()})")
