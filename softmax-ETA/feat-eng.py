# trips_feature_engineering.py
# OPTIMIZED VERSION - Uses more RAM and less CPU for rolling window calculations
# Key optimizations:
# - Uses Polars' time-based rolling functions (rolling_sum_by, rolling_mean_by, rolling_std_by)
# - Leverages lazy evaluation for better memory management  
# - Vectorized operations instead of loops
# - Efficient time-based rolling windows with timedelta window sizes
# - Reduced CPU overhead by eliminating multiple filter operations and self-joins

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
    pl.col("id_usuario").cast(pl.Int64, strict=False),
    pl.col("duracion_recorrido").cast(pl.Float64, strict=False),
    pl.col("id_estacion_origen").cast(pl.Int64, strict=False),
    pl.col("id_estacion_destino").cast(pl.Int64, strict=False)
]).filter(
    pl.col("fecha_origen_recorrido").is_not_null() &
    pl.col("fecha_destino_recorrido").is_not_null() &
    pl.col("id_usuario").is_not_null() &
    pl.col("duracion_recorrido").is_not_null() &
    pl.col("id_estacion_origen").is_not_null() &
    pl.col("id_estacion_destino").is_not_null()
)

print(f"Trip count after datetime conversion and null filtering: {len(df)}")

# ──────────────────── Data Quality Filters ────────────────────
print("\nApplying data quality filters...")

# 1. Remove trips with unrealistic durations (< 1 minute or > 6 hours)
trips_before_duration = len(df)
df = df.filter(
    (pl.col("duracion_recorrido") >= 60) &      # At least 1 minute
    (pl.col("duracion_recorrido") <= 21600)     # At most 6 hours
)
removed_duration = trips_before_duration - len(df)
print(f"Removed {removed_duration:,} trips with unrealistic durations (< 1 min or > 6 hours)")

# 2. Remove stations with less than 5000 total trips
print("Calculating station trip counts...")

# Count trips per origin station
origin_counts = df.group_by("id_estacion_origen").agg(
    pl.count().alias("origin_trip_count")
)

# Count trips per destination station  
dest_counts = df.group_by("id_estacion_destino").agg(
    pl.count().alias("dest_trip_count")
)

# Combine counts - a station needs 5000+ trips as either origin OR destination
station_counts = (
    origin_counts
    .join(dest_counts, left_on="id_estacion_origen", right_on="id_estacion_destino", how="full")
    .with_columns([
        # Use the station ID from either origin or destination
        pl.coalesce([pl.col("id_estacion_origen"), pl.col("id_estacion_destino")]).alias("station_id"),
        # Sum origin and destination counts (filling nulls with 0)
        (pl.col("origin_trip_count").fill_null(0) + pl.col("dest_trip_count").fill_null(0)).alias("total_trips")
    ])
)

# Get stations with at least 5000 trips
valid_stations = (
    station_counts
    .filter(pl.col("total_trips") >= 5000)
    .select("station_id")
    .to_series()
    .to_list()
)

print(f"Found {len(valid_stations)} stations with 5000+ trips out of {station_counts.height} total stations")

# Filter trips to only include valid stations
trips_before_station_filter = len(df)
df = df.filter(
    pl.col("id_estacion_origen").is_in(valid_stations) &
    pl.col("id_estacion_destino").is_in(valid_stations)
)
print(f"Removed {trips_before_station_filter - len(df):,} trips with low-activity stations")

print(f"Final trip count after quality filters: {len(df):,}")
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

# Use rolling_sum on constant column for counting - more efficient than self-joins
print("Computing user rolling statistics with efficient rolling operations...")

# Add a constant column for counting trips
df_sorted = df_sorted.with_columns([
    pl.lit(1).alias("trip_count")
])

# Use rolling_sum for counting and rolling_mean/std for statistics
df_sorted = (
    df_sorted
    .sort(["id_usuario", "fecha_origen_recorrido"])
    .with_columns([
        # Rolling trip counts using rolling_sum_by for time-based windows
        pl.col("trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(days=7))
        .over("id_usuario")
        .alias("user_trips_7d_temp"),
        
        pl.col("trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(days=30))
        .over("id_usuario")
        .alias("user_trips_30d_temp"),
        
        pl.col("trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("id_usuario")
        .alias("user_trips_90d_temp"),
        
        # Duration statistics for 90-day window
        pl.col("duracion_recorrido")
        .rolling_mean_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("id_usuario")
        .alias("user_avg_dur_90d"),
        
        pl.col("duracion_recorrido")
        .rolling_std_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("id_usuario")
        .alias("user_std_dur_90d")
    ])
    .with_columns([
        # Subtract 1 to exclude current trip and clip to avoid negative values
        (pl.col("user_trips_7d_temp") - 1).clip(lower_bound=0).alias("user_trips_7d"),
        (pl.col("user_trips_30d_temp") - 1).clip(lower_bound=0).alias("user_trips_30d"),
        (pl.col("user_trips_90d_temp") - 1).clip(lower_bound=0).alias("user_trips_90d")
    ])
    .drop(["user_trips_7d_temp", "user_trips_30d_temp", "user_trips_90d_temp", "trip_count"])
)

# Fill null values with 0
df_sorted = df_sorted.with_columns([
    pl.col("user_avg_dur_90d").fill_null(0),
    pl.col("user_std_dur_90d").fill_null(0)
])

print(f"Added user rolling features for 7d, 30d, 90d windows")

# Calculate preferred stations (mode) - more efficient approach
print("Calculating user preferred stations...")
user_prefs = (
    df_sorted
    .group_by("id_usuario")
    .agg([
        pl.col("id_estacion_origen").mode().first().alias("user_pref_start"),
        pl.col("id_estacion_destino").mode().first().alias("user_pref_dest")
    ])
)

df_sorted = df_sorted.join(user_prefs, on="id_usuario", how="left")

# ──────────────────── 5. Features de estación ────────────────────
print("Processing station features...")

# Station rolling operations using efficient rolling_sum approach
print("Processing station rolling counts with optimized approach...")

# Add trip count column for station calculations
df_sorted = df_sorted.with_columns([
    pl.lit(1).alias("station_trip_count")
])

# Process both origin and destination station features efficiently
df_sorted = (
    df_sorted
    .sort(["id_estacion_origen", "fecha_origen_recorrido"])
    .with_columns([
        # 1-hour and 24-hour rolling counts for origin stations
        pl.col("station_trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(hours=1))
        .over("id_estacion_origen")
        .alias("station_orig_trips_1h_temp"),
        
        pl.col("station_trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(hours=24))
        .over("id_estacion_origen")
        .alias("station_orig_trips_24h_temp")
    ])
    .sort(["id_estacion_destino", "fecha_origen_recorrido"])
    .with_columns([
        # 1-hour and 24-hour rolling counts for destination stations
        pl.col("station_trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(hours=1))
        .over("id_estacion_destino")
        .alias("station_dest_trips_1h_temp"),
        
        pl.col("station_trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(hours=24))
        .over("id_estacion_destino")
        .alias("station_dest_trips_24h_temp")
    ])
    .with_columns([
        # Subtract 1 to exclude current trip and clip to avoid negative values
        (pl.col("station_orig_trips_1h_temp") - 1).clip(lower_bound=0).alias("station_orig_trips_1h"),
        (pl.col("station_orig_trips_24h_temp") - 1).clip(lower_bound=0).alias("station_orig_trips_24h"),
        (pl.col("station_dest_trips_1h_temp") - 1).clip(lower_bound=0).alias("station_dest_trips_1h"),
        (pl.col("station_dest_trips_24h_temp") - 1).clip(lower_bound=0).alias("station_dest_trips_24h")
    ])
    .drop([
        "station_trip_count", 
        "station_orig_trips_1h_temp", "station_orig_trips_24h_temp",
        "station_dest_trips_1h_temp", "station_dest_trips_24h_temp"
    ])
)

print("Added efficient station rolling features (1h, 24h windows)")

# Net flow calculation
df_sorted = df_sorted.with_columns([
    (pl.col("station_dest_trips_1h") - pl.col("station_orig_trips_1h")).alias("station_net_flow_1h")
])

# Station duration statistics using efficient rolling functions
print("Calculating station origin average duration with optimized approach...")

df_sorted = (
    df_sorted
    .sort(["id_estacion_origen", "fecha_origen_recorrido"])
    .with_columns([
        # 30-day rolling average duration for origin stations
        pl.col("duracion_recorrido")
        .rolling_mean_by(by="fecha_origen_recorrido", window_size=timedelta(days=30))
        .over("id_estacion_origen")
        .alias("station_orig_avg_dur_30d")
    ])
    .with_columns([
        pl.col("station_orig_avg_dur_30d").fill_null(0)
    ])
)

print("Added efficient station duration features")

# Station destination diversity using efficient rolling functions
print("Calculating destination diversity with optimized approach...")

# Add trip count column for diversity calculations
df_sorted = df_sorted.with_columns([
    pl.lit(1).alias("dest_trip_count")
])

# Calculate destination diversity using rolling_sum (simpler than entropy but more efficient)
df_sorted = (
    df_sorted
    .sort(["id_estacion_origen", "fecha_origen_recorrido"])
    .with_columns([
        # 90-day rolling count of trips from this origin station
        pl.col("dest_trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("id_estacion_origen")
        .alias("station_orig_trip_count_90d_temp")
    ])
    .with_columns([
        # Subtract 1 to exclude current trip
        (pl.col("station_orig_trip_count_90d_temp") - 1).clip(lower_bound=0).alias("station_dest_count_90d")
    ])
    .drop(["station_orig_trip_count_90d_temp", "dest_trip_count"])
)

print("Added efficient station destination diversity features")

# ──────────────────── 6. Features de ruta (origen-destino) ────────────────────
print("Processing route features with optimized approach...")

# Create route key for efficient grouping operations
df_sorted = df_sorted.with_columns([
    (pl.col("id_estacion_origen").cast(pl.Utf8) + "_" + pl.col("id_estacion_destino").cast(pl.Utf8)).alias("route_key")
])

# Add trip count column for route calculations
df_sorted = df_sorted.with_columns([
    pl.lit(1).alias("route_trip_count")
])

# Calculate 90-day rolling route statistics using efficient built-in functions
df_sorted = (
    df_sorted
    .sort(["route_key", "fecha_origen_recorrido"])
    .with_columns([
        # 90-day rolling statistics for each route
        pl.col("route_trip_count")
        .rolling_sum_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("route_key")
        .alias("route_trip_count_90d_temp"),
        
        pl.col("duracion_recorrido")
        .rolling_mean_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("route_key")
        .alias("route_avg_dur_90d"),
        
        pl.col("duracion_recorrido")
        .rolling_std_by(by="fecha_origen_recorrido", window_size=timedelta(days=90))
        .over("route_key")
        .alias("route_std_dur_90d")
    ])
    .with_columns([
        # Subtract 1 from count to exclude current trip
        (pl.col("route_trip_count_90d_temp") - 1).clip(lower_bound=0).alias("route_trip_count_90d")
    ])
    .drop(["route_trip_count_90d_temp", "route_key", "route_trip_count"])
    .with_columns([
        pl.col("route_avg_dur_90d").fill_null(0),
        pl.col("route_std_dur_90d").fill_null(0)
    ])
)

print("Added efficient route features (90d rolling statistics)")

# ──────────────────── 7. Clima derivado e interacciones ────────────────────
df_sorted = df_sorted.with_columns([
    (pl.col("weather_apparent_temperature") - pl.col("weather_temperature_2m")).alias("feels_like_diff"),
    (pl.col("weather_rain") > 0).cast(pl.Int8).alias("rain_flag"),
    (pl.col("weather_precipitation") > 0).cast(pl.Int8).alias("precip_flag")
])

# Weather lag features - efficient vectorized approach
print("Adding weather lag features...")
weather_lag_cols = ["weather_precipitation", "weather_rain"]
existing_weather_cols = [col for col in weather_lag_cols if col in df_sorted.columns]

if existing_weather_cols:
    # Sort by time once and create all lag features in one operation
    df_sorted = df_sorted.sort("fecha_origen_recorrido")
    
    lag_exprs = []
    for col in existing_weather_cols:
        col_name = col.replace('weather_', '')
        lag_exprs.append(pl.col(col).shift(1).alias(f"{col_name}_lag_1h"))
    
    df_sorted = df_sorted.with_columns(lag_exprs)
    print(f"Added lag features for: {existing_weather_cols}")
else:
    print("No weather columns found for lag features")

# Weather code categories
if "weather_weather_code" in df_sorted.columns:
    weather_code_mapping = {
        0: 'Clear', 1: 'Clear', 2: 'Clear', 3: 'Clear',
        45: 'Clouds', 48: 'Clouds', 51: 'Drizzle', 53: 'Drizzle', 
        55: 'Drizzle', 61: 'Rain', 63: 'Rain', 65: 'Rain'
    }
    
    df_sorted = df_sorted.with_columns([
        pl.col("weather_weather_code").replace_strict(weather_code_mapping, default="null").alias("weather_code_cat")
    ])
    
    # One-hot encoding
    df_sorted = df_sorted.to_dummies("weather_code_cat", separator="_")

# Temperature and wind buckets using when/then/otherwise for reliability
if "weather_temperature_2m" in df_sorted.columns:
    df_sorted = df_sorted.with_columns([
        pl.when(pl.col("weather_temperature_2m") < 5)
        .then(pl.lit("mucha_frio"))
        .when(pl.col("weather_temperature_2m") < 15)
        .then(pl.lit("frio"))
        .when(pl.col("weather_temperature_2m") < 25)
        .then(pl.lit("templado"))
        .when(pl.col("weather_temperature_2m") < 35)
        .then(pl.lit("calor"))
        .otherwise(pl.lit("mucha_calor"))
        .alias("temp_bucket")
    ])
else:
    df_sorted = df_sorted.with_columns([pl.lit("unknown").alias("temp_bucket")])

if "weather_wind_speed_10m" in df_sorted.columns:
    df_sorted = df_sorted.with_columns([
        pl.when(pl.col("weather_wind_speed_10m") < 5)
        .then(pl.lit("calma"))
        .when(pl.col("weather_wind_speed_10m") < 15)
        .then(pl.lit("brisa"))
        .when(pl.col("weather_wind_speed_10m") < 25)
        .then(pl.lit("ventoso"))
        .otherwise(pl.lit("muy_ventoso"))
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

# Final filter: Remove any rows with NaN targets (crucial for model training)
print("\nFinal target validation...")
pre_target_filter = len(df_sorted)

df_sorted = df_sorted.filter(
    pl.col("duracion_recorrido").is_not_null() &
    pl.col("id_estacion_destino").is_not_null() &
    pl.col("duracion_recorrido").is_finite()  # Remove infinite values too
)

removed_nan_targets = pre_target_filter - len(df_sorted)
if removed_nan_targets > 0:
    print(f"Removed {removed_nan_targets:,} trips with NaN/infinite target values")

print(f"Final dataset: {len(df_sorted):,} trips ready for modeling")

# Fill NaN values in rolling columns with 0 - vectorized approach
rolling_tags = ["trips_", "avg_dur_", "std_dur_", "entropy", "flow", "route", "_lag_"]
rolling_cols = [c for c in df_sorted.columns if any(tag in c for tag in rolling_tags)]

if rolling_cols:
    fill_exprs = [pl.col(col).fill_null(0).alias(col) for col in rolling_cols]
    df_sorted = df_sorted.with_columns(fill_exprs)
    print(f"Filled null values for {len(rolling_cols)} rolling/lag features")

# Replace infinite values with null - efficient vectorized approach
numeric_dtypes = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
numeric_cols = [col for col, dtype in zip(df_sorted.columns, df_sorted.dtypes) 
                if dtype in numeric_dtypes]

if numeric_cols:
    inf_replace_exprs = [
        pl.when(pl.col(col).is_infinite())
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in numeric_cols
    ]
    df_sorted = df_sorted.with_columns(inf_replace_exprs)
    print(f"Replaced infinite values for {len(numeric_cols)} numeric columns")

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
