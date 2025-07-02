import pandas as pd

# Load the parquet file
df = pd.read_parquet('data/processed/trips_with_weather.parquet')

# Print all columns that start with 'weather'
weather_columns = [col for col in df.columns if col.startswith('weather')]
print('Weather columns:', weather_columns)

# Check for specific columns
required_columns = ['weather_temperature_2m', 'weather_relative_humidity_2m', 'weather_apparent_temperature']
for col in required_columns:
    if col in df.columns:
        print(f"Column '{col}' is present.")
    else:
        print(f"Column '{col}' is missing.") 