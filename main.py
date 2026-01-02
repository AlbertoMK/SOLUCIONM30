"""
Main entry point for the TFG Traffic Optimization project.
Processing pipeline execution.
"""
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW, M30_EAST_SENSORS
from src.physics import TrafficPhysics
from src.optimizer import TrafficOptimizer

def main():
    """
    Main execution function. runs the pipeline on a sample.
    """
    print("🚦 Starting Traffic Flow Optimization System (TFG M-30)...")
    
    # 1. Load Sample
    sample_file = DATA_PATH_RAW / "trafico" / "01-2019" / "01-2019.csv"
    if not sample_file.exists():
        print(f"❌ Sample file not found at {sample_file}")
        return

    print(f"📥 Loading data from: {sample_file.name}...")
    df = load_csv_data(sample_file)
    
    # 2. Preprocess
    print("🧹 Cleaning and Feature Engineering...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
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
    
    k_crit = TrafficPhysics.calculate_critical_density(sensor_data)
    print(f"   - Detected Critical Density for {sensor_id}: {k_crit:.2f} veh/km")
    
    optimizer = TrafficOptimizer(critical_density_override=k_crit)
    df_opt = optimizer.optimize_traffic(sensor_data)
    
    # Stats
    n_optimized = df_opt[df_opt['optimal_speed_limit'] < 90].shape[0]
    avg_speed_real = df_opt['vmed'].mean()
    avg_speed_sim = df_opt['simulated_speed'].mean()
    
    print(f"   - Time intervals with active VSL (70km/h): {n_optimized} / {len(df_opt)}")
    print(f"   - Average Speed (Real): {avg_speed_real:.2f} km/h")
    print(f"   - Average Speed (Simulated): {avg_speed_sim:.2f} km/h")
    print(f"   - Improvement: {((avg_speed_sim - avg_speed_real)/avg_speed_real)*100:.1f}%")
    
    print("\n🚀 Pipeline finished successfully.")
    print("\nTo view the Digital Twin Dashboard, run:")
    print("   streamlit run frontend/app.py")

if __name__ == "__main__":
    main()
