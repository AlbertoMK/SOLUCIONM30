import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW, DATA_PATH_PROCESSED
from src.physics import TrafficPhysics
from src.optimizer import TrafficOptimizer

def main():
    print("🔎 Analyzing VSL Improvement across sensors...")
    
    # 1. Load Data
    sample_file = DATA_PATH_RAW / "trafico" / "01-2019" / "01-2019.csv"
    if not sample_file.exists():
        print(f"❌ Data file not found: {sample_file}")
        return
        
    print(f"📥 Loading {sample_file.name}...")
    df = load_csv_data(sample_file)
    
    # Load Limits
    limits_file = DATA_PATH_PROCESSED / "realvlimit" / "sensor_limits.csv"
    limits_df = pd.DataFrame()
    if limits_file.exists():
        limits_df = pd.read_csv(limits_file)
        
    # 2. Preprocess (Global)
    print("🧹 Preprocessing complete dataset...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_features(df_clean)
    
    # 3. Analyze per Sensor
    unique_sensors = df_features['id'].unique()
    print(f"🧪 Testing {len(unique_sensors)} sensors...")
    
    results = []
    
    for sensor_id in tqdm(unique_sensors):
        sensor_data = df_features[df_features['id'] == sensor_id].copy()
        
        # Skip if too little data
        if len(sensor_data) < 100:
            continue
            
        # Get Base Limit
        base_limit = 90
        if not limits_df.empty:
            l_row = limits_df[limits_df['id'] == sensor_id]
            if not l_row.empty:
                base_limit = int(l_row.iloc[0]['inferred_limit'])
        
        # Calc Physics
        k_crit = TrafficPhysics.calculate_critical_density(sensor_data)
        q_max = TrafficPhysics.calculate_max_capacity(sensor_data)
        
        # Optimize
        optimizer = TrafficOptimizer(
            critical_density_override=k_crit,
            max_capacity_override=q_max,
            base_speed_limit=base_limit
        )
        df_opt = optimizer.optimize_traffic(sensor_data)
        
        # Calculate Metrics
        # Mask where VSL was active (Dynamic Limit < Base Limit)
        # Note: In optimizer, we set dynamic limit to target (e.g. 70) if active.
        # But if not active, it returns base_limit.
        vsl_active_mask = df_opt['limite_dinamico'] < base_limit
        
        df_active = df_opt[vsl_active_mask]
        n_active = len(df_active)
        pct_active = (n_active / len(df_opt)) * 100
        
        if n_active > 0:
            avg_speed_real = df_active['vmed'].mean()
            avg_speed_sim = df_active['velocidad_opt'].mean()
            
            # Absolute diff
            diff_speed = avg_speed_sim - avg_speed_real
            # Pct Improvement
            pct_improvement = (diff_speed / avg_speed_real) * 100 if avg_speed_real > 0 else 0
        else:
            diff_speed = 0
            pct_improvement = 0
            
        results.append({
            'sensor_id': sensor_id,
            'k_crit': k_crit,
            'base_limit': base_limit,
            'pct_active': pct_active,
            'n_active_intervals': n_active,
            'avg_speed_improvement_kmh': diff_speed,
            'pct_speed_improvement': pct_improvement
        })
        
    # 4. Results
    results_df = pd.DataFrame(results)
    
    # Sort by Percent Improvement (descending)
    # But only consider sensors where VSL was active at least 1% of the time to avoid outliers
    relevant_df = results_df[results_df['pct_active'] > 0.1].copy()
    top_df = relevant_df.sort_values(by='pct_speed_improvement', ascending=False)
    
    print("\n🏆 TOP 10 SENSORS BY IMPROVEMENT (during congestion):")
    print(top_df[['sensor_id', 'base_limit', 'pct_active', 'avg_speed_improvement_kmh', 'pct_speed_improvement']].head(10).to_string(index=False))
    
    # Save to CSV
    out_file = "analysis_improvements.csv"
    results_df.sort_values(by='pct_speed_improvement', ascending=False).to_csv(out_file, index=False)
    print(f"\n📂 Full results saved to {out_file}")

if __name__ == "__main__":
    main()
