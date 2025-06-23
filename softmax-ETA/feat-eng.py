# trips_feature_engineering.py
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta
import holidays   # pip install holidays

# ──────────────────── 1. Carga de datos ────────────────────
TRIPS_PATH  = Path("data/processed/trips_with_weather.parquet")
USERS_PATH  = Path("data/raw/combined/users.csv")
OUTPUT_PATH = Path("data/processed/trips_feat_eng.parquet")

# Check if input files exist
if not TRIPS_PATH.exists():
    raise FileNotFoundError(f"Trips file not found: {TRIPS_PATH}")
if not USERS_PATH.exists():
    raise FileNotFoundError(f"Users file not found: {USERS_PATH}")

# Create output directory if it doesn't exist
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

try:
    df = pl.read_parquet(TRIPS_PATH)
    print(f"Successfully loaded trips data: {len(df)} rows")
except Exception as e:
    raise ValueError(f"Error loading trips data: {e}")

try:
    users = pl.read_csv(
        USERS_PATH,
        schema_overrides={
            'ID_usuario': pl.Utf8,
            'id_usuario': pl.Utf8,
            'edad_usuario': pl.Utf8  # load as string, we will clean later
        },
        infer_schema_length=10000  # give Polars some extra rows for better inference
    )
    print(f"Successfully loaded users data: {len(users)} rows")
except Exception as e:
    raise ValueError(f"Error loading users data: {e}")

# Handle potentially duplicate user ID columns by coalescing them
if 'ID_usuario' in users.columns and 'id_usuario' in users.columns:
    print("Found both 'ID_usuario' and 'id_usuario'. Coalescing into a single column.")
    users = users.with_columns([
        pl.coalesce([pl.col('ID_usuario'), pl.col('id_usuario')]).alias('ID_usuario')
    ]).drop('id_usuario')

# Rename columns for consistency
users = users.rename({
    "ID_usuario": "id_usuario",
    "fecha_alta": "fecha_alta",
    "genero_usuario": "genero_usuario",   # evita colisión con 'genero' de trip
    "edad_usuario": "edad_alta",
    "hora_alta": "hora_alta"
})

# Convert user ID to numeric, removing non-numeric IDs
print(f"Converting 'id_usuario' in users df to integer. Current user count: {len(users)}")
users = users.with_columns([
    # Remove any non-digit characters then cast. This mimics pandas `to_numeric(errors='coerce')` behaviour.
    pl.col('id_usuario')
      .str.replace_all(r"[^0-9]", "")
      .cast(pl.Int64, strict=False)
      .alias('id_usuario')
]).filter(pl.col('id_usuario').is_not_null())
print(f"User count after converting IDs to integer and dropping invalid: {len(users)}")

# Handle duplicate user rows by keeping the one with the latest registration date
users_original_count = len(users)
print(f"Users before deduplication: {users_original_count}")
users = users.with_columns([
    pl.col('fecha_alta').str.to_datetime(strict=False).alias('fecha_alta')
]).filter(pl.col('fecha_alta').is_not_null() & pl.col('id_usuario').is_not_null())

users = users.sort('fecha_alta').unique(subset=['id_usuario'], keep='last')
print(f"Users after deduplication: {len(users)}")
print(f"Removed {users_original_count - len(users)} duplicate or invalid user entries.")

# Convert datetime columns and validate
df = df.with_columns([
    pl.col("fecha_origen_recorrido").str.to_datetime(strict=False),
    pl.col("fecha_destino_recorrido").str.to_datetime(strict=False),
    pl.col("id_usuario").cast(pl.Int64, strict=False)
]).filter(
    pl.col("fecha_origen_recorrido").is_not_null() &
    pl.col("fecha_destino_recorrido").is_not_null() &
    pl.col("id_usuario").is_not_null()
)

print(f"Trip count after datetime conversion and filtering: {len(df)}")
df = df.sort("fecha_origen_recorrido")

# ──────────────────── 2. Join con tabla de usuarios ────────────────────
print(f"Original trip count: {len(df)}")
df = df.join(users, how="left", on="id_usuario")

# ──────────────────── 3. Features temporales ────────────────────
df = df.with_columns([
    # Use default integer sizes to avoid dtype mismatch issues
    pl.col("fecha_origen_recorrido").dt.hour().alias("hora"),
    pl.col("fecha_origen_recorrido").dt.weekday().alias("dia_semana"),  # 0 = Monday
    pl.col("fecha_origen_recorrido").dt.month().alias("mes"),
    pl.col("fecha_origen_recorrido").dt.date().alias("fecha_date"),
    
    # Weekend flag right away
    pl.col("fecha_origen_recorrido").dt.weekday().is_in([5, 6]).cast(pl.Int8).alias("es_fin_de_semana")
])

# Feriados Argentina - convert to list for better performance
ar_holidays = holidays.AR()
holiday_dates = set(ar_holidays.keys())

def is_holiday(date_series):
    return [1 if d in holiday_dates else 0 for d in date_series]

# Apply holidays using map_elements (more efficient than apply)
df = df.with_columns([
    pl.col("fecha_date").map_elements(lambda x: 1 if x in holiday_dates else 0, return_dtype=pl.Int64).alias("es_feriado_ar")
])

# Codificación cíclica
df = df.with_columns([
    (2 * np.pi * pl.col("hora") / 24).sin().alias("sin_hora"),
    (2 * np.pi * pl.col("hora") / 24).cos().alias("cos_hora"),
    (2 * np.pi * pl.col("dia_semana") / 7).sin().alias("sin_dow"),
    (2 * np.pi * pl.col("dia_semana") / 7).cos().alias("cos_dow"),
    (2 * np.pi * pl.col("mes") / 12).sin().alias("sin_month"),
    (2 * np.pi * pl.col("mes") / 12).cos().alias("cos_month")
])

# Wind direction cyclical encoding (if available)
if "weather_wind_direction_10m" in df.columns:
    df = df.with_columns([
        (2 * np.pi * pl.col("weather_wind_direction_10m") / 360).sin().alias("wind_dir_sin"),
        (2 * np.pi * pl.col("weather_wind_direction_10m") / 360).cos().alias("wind_dir_cos")
    ])

# Enhanced temporal patterns
df = df.with_columns([
    ((pl.col("hora") >= 7) & (pl.col("hora") <= 9) | 
     (pl.col("hora") >= 17) & (pl.col("hora") <= 19)).cast(pl.Int8).alias("peak_commute"),
    pl.col("fecha_origen_recorrido").dt.day().is_in([1, 15, 30, 31]).cast(pl.Int8).alias("payday_flag"),
    pl.col("mes").is_in([12, 1, 2]).cast(pl.Int8).alias("vacation_season")
])

# ──────────────────── 4. Features de usuario ────────────────────
# Handle user features more efficiently
df = df.with_columns([
    pl.col('fecha_alta').cast(pl.Datetime).alias('fecha_alta'),
    pl.col('edad_alta').cast(pl.Float64, strict=False).alias('edad_alta')
])

# Create user features only where data exists
df = df.with_columns([
    pl.when(pl.col('fecha_alta').is_not_null() & pl.col('edad_alta').is_not_null())
    .then(pl.col('edad_alta') + (pl.col("fecha_origen_recorrido").dt.year() - pl.col('fecha_alta').dt.year()))
    .otherwise(None)
    .alias('user_age_at_trip'),
    
    pl.when(pl.col('fecha_alta').is_not_null())
    .then((pl.col("fecha_origen_recorrido") - pl.col('fecha_alta')).dt.total_days().clip(lower_bound=0))
    .otherwise(None)
    .alias('user_days_since_signup')
])

valid_user_count = df.filter(pl.col('user_age_at_trip').is_not_null()).height
print(f"Found {valid_user_count} trips with valid user data out of {len(df)} total trips.")

# Rolling windows (usuario) - Much more memory efficient with Polars
print("Processing user rolling windows...")

# Sort by user and date for rolling operations
df_sorted = df.sort(["id_usuario", "fecha_origen_recorrido"])

# Simplified approach: calculate rolling windows using basic aggregations
for window_days, label in [(7, "7d"), (30, "30d"), (90, "90d")]:
    print(f"Processing user trips for {window_days} day window...")
    
    # Calculate window start for each trip
    df_sorted = df_sorted.with_columns([
        (pl.col("fecha_origen_recorrido") - pl.duration(days=window_days)).alias(f"window_start_{label}")
    ])
    
    # Self-join to count trips within the window
    trip_counts = (
        df_sorted
        .select(["id_usuario", "fecha_origen_recorrido", f"window_start_{label}"])
        .join(
            df_sorted.select(["id_usuario", "fecha_origen_recorrido"]).rename({"fecha_origen_recorrido": "trip_date"}),
            on="id_usuario",
            how="left"
        )
        # Count trips where trip_date is within the window (after window_start, before current time)
        .filter(
            (pl.col("trip_date") >= pl.col(f"window_start_{label}")) &
            (pl.col("trip_date") < pl.col("fecha_origen_recorrido"))
        )
        .group_by(["id_usuario", "fecha_origen_recorrido"])
        .agg([
            pl.len().alias(f"user_trips_{label}")
        ])
    )
    
    # Join back to main dataframe
    df_sorted = df_sorted.join(
        trip_counts,
        on=["id_usuario", "fecha_origen_recorrido"],
        how="left"
    ).drop(f"window_start_{label}")
    
    # Fill null values with 0 (no previous trips)
    df_sorted = df_sorted.with_columns([
        pl.col(f"user_trips_{label}").fill_null(0)
    ])
    
    non_null_count = df_sorted.filter(pl.col(f"user_trips_{label}").is_not_null()).height
    print(f"Added feature 'user_trips_{label}', non-null values: {non_null_count}")

print("Processing user duration statistics for 90 day window...")
# Calculate 90-day window duration statistics
df_sorted = df_sorted.with_columns([
    (pl.col("fecha_origen_recorrido") - pl.duration(days=90)).alias("window_start_90d")
])

# Self-join to get durations within the 90-day window
user_dur_stats = (
    df_sorted
    .select(["id_usuario", "fecha_origen_recorrido", "window_start_90d"])
    .join(
        df_sorted.select(["id_usuario", "fecha_origen_recorrido", "duracion_recorrido"]).rename({
            "fecha_origen_recorrido": "trip_date", 
            "duracion_recorrido": "trip_duration"
        }),
        on="id_usuario",
        how="left"
    )
    # Filter trips within the 90-day window
    .filter(
        (pl.col("trip_date") >= pl.col("window_start_90d")) &
        (pl.col("trip_date") < pl.col("fecha_origen_recorrido"))
    )
    .group_by(["id_usuario", "fecha_origen_recorrido"])
    .agg([
        pl.col("trip_duration").mean().alias("user_avg_dur_90d"),
        pl.col("trip_duration").std().alias("user_std_dur_90d")
    ])
)

df_sorted = df_sorted.join(
    user_dur_stats, 
    on=["id_usuario", "fecha_origen_recorrido"], 
    how="left"
).drop("window_start_90d")

# Fill null values with 0
df_sorted = df_sorted.with_columns([
    pl.col("user_avg_dur_90d").fill_null(0),
    pl.col("user_std_dur_90d").fill_null(0)
])

# Calculate preferred stations (mode)
print("Calculating user preferred stations...")
user_prefs = df_sorted.group_by("id_usuario").agg([
    pl.col("id_estacion_origen").mode().first().alias("user_pref_start"),
    pl.col("id_estacion_destino").mode().first().alias("user_pref_dest")
])

df_sorted = df_sorted.join(user_prefs, on="id_usuario", how="left")

# ──────────────────── 5. Features de estación ────────────────────
print("Processing station features...")

# Station rolling operations using simplified approach
for col_station, prefix in [("id_estacion_origen", "orig"), ("id_estacion_destino", "dest")]:
    print(f"Processing {prefix} station rolling counts...")
    
    df_sorted = df_sorted.sort([col_station, "fecha_origen_recorrido"])
    
    # Calculate 1-hour window
    df_sorted = df_sorted.with_columns([
        (pl.col("fecha_origen_recorrido") - pl.duration(hours=1)).alias(f"window_start_1h_{prefix}"),
        (pl.col("fecha_origen_recorrido") - pl.duration(hours=24)).alias(f"window_start_24h_{prefix}")
    ])
    
    # Self-join for 1-hour and 24-hour windows
    for window_hours, window_label in [(1, "1h"), (24, "24h")]:
        station_counts = (
            df_sorted
            .select([col_station, "fecha_origen_recorrido", f"window_start_{window_label}_{prefix}"])
            .join(
                df_sorted.select([col_station, "fecha_origen_recorrido"]).rename({"fecha_origen_recorrido": "trip_date"}),
                on=col_station,
                how="left"
            )
            .filter(
                (pl.col("trip_date") >= pl.col(f"window_start_{window_label}_{prefix}")) &
                (pl.col("trip_date") < pl.col("fecha_origen_recorrido"))
            )
            .group_by([col_station, "fecha_origen_recorrido"])
            .agg([
                pl.len().alias(f"station_{prefix}_trips_{window_label}")
            ])
        )
        
        df_sorted = df_sorted.join(
            station_counts,
            on=[col_station, "fecha_origen_recorrido"],
            how="left"
        ).with_columns([
            pl.col(f"station_{prefix}_trips_{window_label}").fill_null(0)
        ])
    
    # Clean up temporary columns
    df_sorted = df_sorted.drop([f"window_start_1h_{prefix}", f"window_start_24h_{prefix}"])

# Net flow calculation
df_sorted = df_sorted.with_columns([
    (pl.col("station_dest_trips_1h") - pl.col("station_orig_trips_1h")).alias("station_net_flow_1h")
])

# Station duration statistics
print("Calculating station origin average duration...")
df_sorted = df_sorted.sort(["id_estacion_origen", "fecha_origen_recorrido"])

# Calculate 30-day window for station duration
df_sorted = df_sorted.with_columns([
    (pl.col("fecha_origen_recorrido") - pl.duration(days=30)).alias("window_start_30d_dur")
])

station_dur_stats = (
    df_sorted
    .select(["id_estacion_origen", "fecha_origen_recorrido", "window_start_30d_dur"])
    .join(
        df_sorted.select(["id_estacion_origen", "fecha_origen_recorrido", "duracion_recorrido"]).rename({
            "fecha_origen_recorrido": "trip_date",
            "duracion_recorrido": "trip_duration"
        }),
        on="id_estacion_origen",
        how="left"
    )
    .filter(
        (pl.col("trip_date") >= pl.col("window_start_30d_dur")) &
        (pl.col("trip_date") < pl.col("fecha_origen_recorrido"))
    )
    .group_by(["id_estacion_origen", "fecha_origen_recorrido"])
    .agg([
        pl.col("trip_duration").mean().alias("station_orig_avg_dur_30d")
    ])
)

df_sorted = df_sorted.join(
    station_dur_stats, 
    on=["id_estacion_origen", "fecha_origen_recorrido"], 
    how="left"
).drop("window_start_30d_dur").with_columns([
    pl.col("station_orig_avg_dur_30d").fill_null(0)
])

# Entropy calculation using a more efficient approach
print("Calculating destination entropy...")

# Custom entropy function that works with Polars
def calculate_entropy(dest_ids):
    """Calculate entropy of destination IDs"""
    if len(dest_ids) == 0:
        return None
    
    # Count frequencies
    unique, counts = np.unique(dest_ids, return_counts=True)
    if len(unique) <= 1:
        return 0.0
    
    # Calculate probabilities and entropy
    probs = counts / counts.sum()
    return -(probs * np.log2(probs)).sum()

# Calculate rolling entropy - this is complex in Polars, so we'll do a simpler version
df_sorted = df_sorted.sort(["id_estacion_origen", "fecha_origen_recorrido"])

# For now, calculate a simpler diversity measure (unique destinations in window)
df_sorted = df_sorted.with_columns([
    (pl.col("fecha_origen_recorrido") - pl.duration(days=90)).alias("window_start_90d_dest")
])

station_dest_count = (
    df_sorted
    .select(["id_estacion_origen", "fecha_origen_recorrido", "window_start_90d_dest"])
    .join(
        df_sorted.select(["id_estacion_origen", "fecha_origen_recorrido"]).rename({"fecha_origen_recorrido": "trip_date"}),
        on="id_estacion_origen",
        how="left"
    )
    .filter(
        (pl.col("trip_date") >= pl.col("window_start_90d_dest")) &
        (pl.col("trip_date") < pl.col("fecha_origen_recorrido"))
    )
    .group_by(["id_estacion_origen", "fecha_origen_recorrido"])
    .agg([
        pl.len().alias("station_dest_count_90d")
    ])
)

df_sorted = df_sorted.join(
    station_dest_count, 
    on=["id_estacion_origen", "fecha_origen_recorrido"], 
    how="left"
).drop("window_start_90d_dest").with_columns([
    pl.col("station_dest_count_90d").fill_null(0)
])

# ──────────────────── 6. Features de ruta (origen-destino) ────────────────────
print("Processing route features...")
df_sorted = df_sorted.sort(["id_estacion_origen", "id_estacion_destino", "fecha_origen_recorrido"])

# Calculate 90-day window for route statistics
df_sorted = df_sorted.with_columns([
    (pl.col("fecha_origen_recorrido") - pl.duration(days=90)).alias("window_start_90d_route")
])

route_stats = (
    df_sorted
    .select(["id_estacion_origen", "id_estacion_destino", "fecha_origen_recorrido", "window_start_90d_route"])
    .join(
        df_sorted.select(["id_estacion_origen", "id_estacion_destino", "fecha_origen_recorrido", "duracion_recorrido"]).rename({
            "fecha_origen_recorrido": "trip_date",
            "duracion_recorrido": "trip_duration"
        }),
        on=["id_estacion_origen", "id_estacion_destino"],
        how="left"
    )
    .filter(
        (pl.col("trip_date") >= pl.col("window_start_90d_route")) &
        (pl.col("trip_date") < pl.col("fecha_origen_recorrido"))
    )
    .group_by(["id_estacion_origen", "id_estacion_destino", "fecha_origen_recorrido"])
    .agg([
        pl.len().alias("route_trip_count_90d"),
        pl.col("trip_duration").mean().alias("route_avg_dur_90d"),
        pl.col("trip_duration").std().alias("route_std_dur_90d")
    ])
)

df_sorted = df_sorted.join(
    route_stats, 
    on=["id_estacion_origen", "id_estacion_destino", "fecha_origen_recorrido"], 
    how="left"
).drop("window_start_90d_route").with_columns([
    pl.col("route_trip_count_90d").fill_null(0),
    pl.col("route_avg_dur_90d").fill_null(0),
    pl.col("route_std_dur_90d").fill_null(0)
])

# ──────────────────── 7. Clima derivado e interacciones ────────────────────
df_sorted = df_sorted.with_columns([
    (pl.col("weather_apparent_temperature") - pl.col("weather_temperature_2m")).alias("feels_like_diff"),
    (pl.col("weather_rain") > 0).cast(pl.Int8).alias("rain_flag"),
    (pl.col("weather_precipitation") > 0).cast(pl.Int8).alias("precip_flag")
])

# Weather lag features - simplified approach
for col in ["weather_precipitation", "weather_rain"]:
    if col in df_sorted.columns:
        col_name = col.replace('weather_', '')
        df_sorted = df_sorted.sort("fecha_origen_recorrido")
        
        # Simple lag by shifting the entire column (this assumes data is sorted by time)
        df_sorted = df_sorted.with_columns([
            pl.col(col).shift(1).alias(f"{col_name}_lag_1h")
        ])

# Weather code categories
if "weather_weather_code" in df_sorted.columns:
    weather_code_mapping = {
        0: 'Clear', 1: 'Clear', 2: 'Clear', 3: 'Clear',
        45: 'Clouds', 48: 'Clouds', 51: 'Drizzle', 53: 'Drizzle', 
        55: 'Drizzle', 61: 'Rain', 63: 'Rain', 65: 'Rain'
    }
    
    df_sorted = df_sorted.with_columns([
        pl.col("weather_weather_code").replace(weather_code_mapping, default="null").alias("weather_code_cat")
    ])
    
    # One-hot encoding
    df_sorted = df_sorted.to_dummies("weather_code_cat", separator="_")

# Temperature and wind buckets
if "weather_temperature_2m" in df_sorted.columns:
    df_sorted = df_sorted.with_columns([
        pl.col("weather_temperature_2m")
        .cut([-np.inf, 5, 15, 25, 35, np.inf], 
             labels=["mucha_frio", "frio", "templado", "calor", "mucha_calor"])
        .alias("temp_bucket")
    ])
else:
    df_sorted = df_sorted.with_columns([pl.lit("unknown").alias("temp_bucket")])

if "weather_wind_speed_10m" in df_sorted.columns:
    df_sorted = df_sorted.with_columns([
        pl.col("weather_wind_speed_10m")
        .cut([-np.inf, 5, 15, 25, np.inf],
             labels=["calma", "brisa", "ventoso", "muy_ventoso"])
        .alias("wind_bucket")
    ])
else:
    df_sorted = df_sorted.with_columns([pl.lit("unknown").alias("wind_bucket")])

# Peak hour interactions
df_sorted = df_sorted.with_columns([
    ((pl.col("hora") >= 7) & (pl.col("hora") <= 10)).cast(pl.Int8).alias("is_peak_morning"),
    ((pl.col("hora") >= 16) & (pl.col("hora") <= 19)).cast(pl.Int8).alias("is_peak_evening")
])

df_sorted = df_sorted.with_columns([
    (pl.col("rain_flag") * pl.col("is_peak_morning")).alias("rain_peak")
])

# ──────────────────── 8. Limpieza final y guardado ────────────────────
# Validate that we still have data after all transformations
if len(df_sorted) == 0:
    raise ValueError("No data remaining after feature engineering transformations")

# Fill NaN values in rolling columns with 0
rolling_cols = [c for c in df_sorted.columns if any(tag in c for tag in
                ["trips_", "avg_dur_", "std_dur_", "entropy", "flow", "route", "_lag_"])]

fill_exprs = [pl.col(col).fill_null(0).alias(col) for col in rolling_cols if col in df_sorted.columns]
if fill_exprs:
    df_sorted = df_sorted.with_columns(fill_exprs)

# Replace infinite values with null
numeric_cols = [col for col, dtype in zip(df_sorted.columns, df_sorted.dtypes) 
                if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]]

inf_replace_exprs = []
for col in numeric_cols:
    inf_replace_exprs.append(
        pl.when(pl.col(col).is_infinite())
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
    )

if inf_replace_exprs:
    df_sorted = df_sorted.with_columns(inf_replace_exprs)

# Report feature counts
total_features = len(df_sorted.columns)
weather_features = len([c for c in df_sorted.columns if 'weather_' in c])
rolling_features = len([c for c in df_sorted.columns if any(tag in c for tag in rolling_cols)])
temporal_features = len([c for c in df_sorted.columns if any(x in c for x in ['sin_', 'cos_', 'hora', 'dia_', 'mes', 'peak', 'feriado'])])

print(f"Total features created: {total_features}")
print(f"Weather features: {weather_features}")
print(f"Rolling/lag features: {rolling_features}")
print(f"Temporal features: {temporal_features}")

# ──────────────────── 9. Temporal Split (Train/Val/Test) ────────────────────
print("\n" + "="*60)
print("CREATING TEMPORAL SPLITS TO AVOID DATA LEAKAGE")
print("="*60)

# Define temporal cutoffs for splits
# Test: August 2023 to end
# Val: 2023 to August 2023 (January 2023 to July 2023)
# Train: Start to 2023 (before January 2023)

from datetime import datetime

test_start = datetime(2023, 8, 1)  # August 1, 2023
val_start = datetime(2023, 1, 1)   # January 1, 2023
train_end = datetime(2023, 1, 1)   # Before January 1, 2023

print(f"Split definitions:")
print(f"  Train:      Data before {train_end.strftime('%Y-%m-%d')}")
print(f"  Validation: {val_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}")
print(f"  Test:       {test_start.strftime('%Y-%m-%d')} onwards")

# Sort by date to ensure proper temporal order
df_sorted = df_sorted.sort("fecha_origen_recorrido")

# Check data range
min_date = df_sorted.select(pl.col("fecha_origen_recorrido").min()).item()
max_date = df_sorted.select(pl.col("fecha_origen_recorrido").max()).item()
print(f"\nData range: {min_date} to {max_date}")

# Create splits
train_data = df_sorted.filter(pl.col("fecha_origen_recorrido") < val_start)
val_data = df_sorted.filter(
    (pl.col("fecha_origen_recorrido") >= val_start) & 
    (pl.col("fecha_origen_recorrido") < test_start)
)
test_data = df_sorted.filter(pl.col("fecha_origen_recorrido") >= test_start)

# Validate splits
print(f"\nSplit sizes:")
print(f"  Train:      {len(train_data):,} trips")
print(f"  Validation: {len(val_data):,} trips")
print(f"  Test:       {len(test_data):,} trips")
print(f"  Total:      {len(train_data) + len(val_data) + len(test_data):,} trips")
print(f"  Original:   {len(df_sorted):,} trips")

# Verify no data leakage by checking date ranges
if len(train_data) > 0:
    train_min = train_data.select(pl.col("fecha_origen_recorrido").min()).item()
    train_max = train_data.select(pl.col("fecha_origen_recorrido").max()).item()
    print(f"\nTrain date range: {train_min} to {train_max}")

if len(val_data) > 0:
    val_min = val_data.select(pl.col("fecha_origen_recorrido").min()).item()
    val_max = val_data.select(pl.col("fecha_origen_recorrido").max()).item()
    print(f"Val date range:   {val_min} to {val_max}")

if len(test_data) > 0:
    test_min = test_data.select(pl.col("fecha_origen_recorrido").min()).item()
    test_max = test_data.select(pl.col("fecha_origen_recorrido").max()).item()
    print(f"Test date range:  {test_min} to {test_max}")

# Define output paths for each split
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_path = output_dir / "trips_train.parquet"
val_path = output_dir / "trips_val.parquet"
test_path = output_dir / "trips_test.parquet"

# Save each split
try:
    print(f"\nSaving datasets...")
    
    if len(train_data) > 0:
        train_data.write_parquet(train_path)
        print(f"  Train saved: {train_path.resolve()} ({len(train_data):,} trips)")
    else:
        print(f"  Warning: No train data to save!")
    
    if len(val_data) > 0:
        val_data.write_parquet(val_path)
        print(f"  Val saved:   {val_path.resolve()} ({len(val_data):,} trips)")
    else:
        print(f"  Warning: No validation data to save!")
        
    if len(test_data) > 0:
        test_data.write_parquet(test_path)
        print(f"  Test saved:  {test_path.resolve()} ({len(test_data):,} trips)")
    else:
        print(f"  Warning: No test data to save!")
    
    print(f"\nAll splits saved successfully!")
    print(f"Dataset shapes: Train{train_data.shape}, Val{val_data.shape}, Test{test_data.shape}")
    
    # Also save the complete dataset for reference
    df_sorted.write_parquet(OUTPUT_PATH)
    print(f"Complete dataset also saved: {OUTPUT_PATH.resolve()}")
    
except Exception as e:
    print(f"Error saving split files: {e}")
    raise
