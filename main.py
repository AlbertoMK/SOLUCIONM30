"""
Main entry point for the TFG Traffic Optimization project.
Processing pipeline execution.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW, DATA_PATH_PROCESSED, M30_EAST_SENSORS
from src.physics import TrafficPhysics
from src.optimizer import TrafficOptimizer

def main():
    """
    Main execution function. runs the pipeline on a sample.
    """
    print("🚦 Starting Traffic Flow Optimization System (M-30)...")
    
    # 1. Load Sample
    sample_file = DATA_PATH_RAW / "trafico" / "01-2019" / "01-2019.csv"
    if not sample_file.exists():
        print(f"❌ Sample file not found at {sample_file}")
        return

    print(f"📥 Loading data from: {sample_file.name}...")
    df = load_csv_data(sample_file)
    
    # Load Limits
    limits_file = DATA_PATH_PROCESSED / "realvlimit" / "sensor_limits.csv"
    limits_df = pd.DataFrame()
    if limits_file.exists():
        limits_df = pd.read_csv(limits_file)
    
    # 2. Preprocess
    print("🧹 Cleaning and Feature Engineering...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    # Show Quality Report
    quality = preprocessor.get_quality_report()
    print("📊 Data Quality Report:")
    for k, v in quality.items():
        print(f"   - {k}: {v}")

    df_features = preprocessor.create_features(df_clean)
    
    if df_features.empty:
        print("⚠️ No data remaining after filtering. Check sensor config.")
        return
        
    print(f"✅ Processed {len(df_features)} records for {len(df_features['id'].unique())} sensors.")

    # 3. Physics & Optimization
    print("🧠 Running Optimization Engine...")
    
    # Pick one sensor for demo
    sensor_id = df_features['id'].iloc[0]
    sensor_data = df_features[df_features['id'] == sensor_id].copy()
    
    # Determine base limit
    base_limit = 90
    if not limits_df.empty:
        l_row = limits_df[limits_df['id'] == sensor_id]
        if not l_row.empty:
            base_limit = int(l_row.iloc[0]['inferred_limit'])
    
    k_crit = TrafficPhysics.calculate_critical_density(sensor_data)
    q_max = TrafficPhysics.calculate_max_capacity(sensor_data)
    
    print(f"   - Detected Critical Density for {sensor_id}: {k_crit:.2f} veh/km")
    print(f"   - Detected Max Capacity (Q_max) for {sensor_id}: {q_max:.0f} veh/h")
    print(f"   - Detected Base Speed Limit for {sensor_id}: {base_limit} km/h")
    
    optimizer = TrafficOptimizer(critical_density_override=k_crit, 
                                 max_capacity_override=q_max,
                                 base_speed_limit=base_limit)
    df_opt = optimizer.optimize_traffic(sensor_data)
    
    # Stats
    n_optimized = df_opt[df_opt['limite_dinamico'] < base_limit].shape[0]
    avg_speed_real = df_opt['vmed'].mean()
    avg_speed_sim = df_opt['velocidad_opt'].mean()
    
    print(f"   - Time intervals with active VSL (70km/h): {n_optimized} / {len(df_opt)}")
    print(f"   - Average Speed (Real): {avg_speed_real:.2f} km/h")
    print(f"   - Average Speed (Simulated): {avg_speed_sim:.2f} km/h")
    print(f"   - Improvement: {((avg_speed_sim - avg_speed_real)/avg_speed_real)*100:.1f}%")
    
    print("\n🚀 Pipeline finished successfully.")
    print("\nTo view the Digital Twin Dashboard, run:")
    print("   streamlit run frontend/app.py")

if __name__ == "__main__":
    main()
