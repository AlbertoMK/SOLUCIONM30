import pandas as pd
from pathlib import Path
from src.data_loader import load_csv_data, load_metadata
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW

def debug_sensor_data(sensor_id, target_date_str):
    """
    Detailed inspection of raw vs processed data for a specific sensor and date.
    """
    file_path = DATA_PATH_RAW / "trafico" / "01-2019" / "01-2019.csv"
    meta_path = DATA_PATH_RAW / "meta" / "pmed_ubicacion_10_2018.csv"
    
    print(f"🔍 Loading data from {file_path}...")
    df = load_csv_data(file_path)
    
    print(f"📊 Filtering for Sensor {sensor_id}...")
    df_sensor = df[df['id'] == sensor_id].copy()
    
    if df_sensor.empty:
        print(f"❌ No data found for sensor {sensor_id}")
        return

    # Check date parsing
    print(f"🕒 Raw 'fecha' sample: {df_sensor['fecha'].head(3).values}")
    df_sensor['fecha'] = pd.to_datetime(df_sensor['fecha'], errors='coerce')
    
    # Filter for target date
    mask = df_sensor['fecha'].dt.date == pd.to_datetime(target_date_str).date()
    df_day = df_sensor[mask].sort_values('fecha')
    
    if df_day.empty:
        print(f"❌ No data found for date {target_date_str}")
        return

    print(f"\n📅 Data for {target_date_str} (First 10 rows):")
    print(df_day[['fecha', 'vmed', 'intensidad', 'error']].head(10))
    
    # Focus on Morning Rush (07:00 - 10:00)
    morning_mask = (df_day['fecha'].dt.hour >= 6) & (df_day['fecha'].dt.hour <= 10)
    df_morning = df_day[morning_mask]
    
    print(f"\n🌅 Morning Rush Hour Stats (06:00 - 10:00):")
    print(df_morning[['vmed', 'intensidad']].describe())
    print("\nSample Morning Rows:")
    print(df_morning[['fecha', 'vmed', 'intensidad', 'error']].head(20))
    
    # Run Preprocessor
    print("\n⚙️ Running Preprocessor...")
    preprocessor = DataPreprocessor() 
    df_clean = preprocessor.clean_data(df)
    df_clean = preprocessor.create_features(df_clean)
    
    # Filter processed data
    df_clean_sensor = df_clean[
        (df_clean['id'] == sensor_id) & 
        (df_clean['fecha'].dt.date == pd.to_datetime(target_date_str).date()) &
        (df_clean['fecha'].dt.hour >= 6) & (df_clean['fecha'].dt.hour <= 10)
    ]
    
    print(f"\n✅ Processed Morning Data ({len(df_clean_sensor)} rows):")
    print(df_clean_sensor[['fecha', 'vmed', 'intensidad', 'density']].head(20))

    # Metadata Check
    print("\n📍 Checking Metadata...")
    meta_df = load_metadata(meta_path)
    if not meta_df.empty:
        if sensor_id in meta_df['id'].values:
            print(meta_df[meta_df['id'] == sensor_id].to_string())
        else:
            print(f"⚠️ Sensor {sensor_id} not found in metadata file.")

if __name__ == "__main__":
    debug_sensor_data(10210, "2019-01-16")
