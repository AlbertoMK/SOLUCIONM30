"""
Preprocessor Module.
Handles data cleaning, feature engineering, and preparation for ML models.
Includes lag generation and date parsing.
"""
import pandas as pd
import numpy as np
from src.config import M30_EAST_SENSORS

class DataPreprocessor:
    """
    Class to handle preprocessing of traffic data.
    """
    
    def __init__(self, sensor_ids=None):
        self.sensor_ids = sensor_ids
        self.quality_report = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs robust cleaning: filtering sensors, handling NaNs, logic checks, and detecting outliers.
        
        Args:
            df (pd.DataFrame): Raw data.
            
        Returns:
            pd.DataFrame: Cleaned data.
        """
        if df.empty:
            return df

        initial_rows = len(df)
        
        # 1. Filter by Sensor ID (if specified)
        if self.sensor_ids is not None:
            valid_ids = set(self.sensor_ids)
            mask = df['id'].isin(valid_ids)
            df_clean = df[mask].copy()
        else:
            df_clean = df.copy()
        
        # 2. Parse Dates
        if 'fecha' in df_clean.columns:
            df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')
        
        # 3. Sort by time
        df_clean = df_clean.sort_values(by=['id', 'fecha'])

        # 4. Filter Invalid Values (Outliers)
        # Valid Speed: 0 to 150 km/h (M-30 limit is usually 90/70, 150 is a safe physical cap)
        # Valid Intensity: 0 to ~12000 veh/h (3-4 lanes max capacity ~8000-9000, 12000 safety margin)
        mask_valid = (
            (df_clean['vmed'] >= 0) & (df_clean['vmed'] <= 150) &
            (df_clean['intensidad'] >= 0) & (df_clean['intensidad'] <= 12000)
        )
        n_outliers = (~mask_valid).sum()
        df_clean = df_clean[mask_valid].copy()

        # 5. Fix Logical Inconsistencies
        # If Speed (vmed) == 0, then Intensity MUST be 0.
        # Sometimes sensors report v=0 but q>0 (stopped cars but counting? or error)
        # Or v>0 but q=0 (single car detected but low flow? valid).
        # We correct v=0, q>0 case -> likely detector stuck or error. Setting q=0.
        mask_zero_speed = (df_clean['vmed'] == 0) & (df_clean['intensidad'] > 0)
        n_logic_fix = mask_zero_speed.sum()
        df_clean.loc[mask_zero_speed, 'intensidad'] = 0
        
        # 6. Handle Missing Values / Interpolation
        # Resample to ensure 15-min intervals if missing rows exist
        # This requires setting index temporarily or using groupby/resample
        # For simplicity in this structure: we use linear interpolate but LIMIT the gap.
        
        # Count NaNs before
        n_nans_before = df_clean[['vmed', 'intensidad']].isna().sum().sum()
        
        df_clean['vmed'] = df_clean.groupby('id')['vmed'].transform(
            lambda x: x.interpolate(method='linear', limit=2) # Limit 2 * 15m = 30 mins
        )
        df_clean['intensidad'] = df_clean.groupby('id')['intensidad'].transform(
            lambda x: x.interpolate(method='linear', limit=2)
        )
        
        # Remaining NaNs are likely large gaps -> Fill with 0 or drop?
        # For traffic flow, if we don't know, dropping/0 is risky. 
        # Let's ffill/bfill for edges, but keep internal large gaps as NaN if critical for physics?
        # The user wants "Better data treatment". Large gaps = Missing Data.
        # We will forward fill strictly 1 step to be safe, then fillna(0) assuming outage = potential low traffic or just unknown.
        # Actually, let's keep it robust: ffill then fillna(0) might bias "zero traffic".
        # Let's Drop rows that couldn't be interpolated to ensure "High Quality Data" only.
        df_clean = df_clean.dropna(subset=['vmed', 'intensidad'])

        # Record Quality Metrics
        self.quality_report = {
            "initial_rows": initial_rows,
            "final_rows": len(df_clean),
            "outliers_removed": int(n_outliers),
            "logic_errors_fixed": int(n_logic_fix),
            "pct_data_kept": round((len(df_clean)/initial_rows)*100, 2) if initial_rows > 0 else 0
        }
        
        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates advanced features: Density, Traffic States, Rolling Trends.
        
        Args:
            df (pd.DataFrame): Cleaned data.
            
        Returns:
            pd.DataFrame: Data with new features.
        """
        df = df.copy()
        
        # --- 1. Robust Density Calculation ---
        # Hybrid Approach:
        # If Speed > 10 km/h: K = Q / V
        # If Speed <= 10 km/h: Use Occupancy if available, else Fallback.
        
        # Helper for vectorization
        condition_normal = (df['vmed'] > 10)
        
        # Case A: Normal Flow
        # Clip speed to avoid div/0 just in case, though cleaned.
        safe_speed = df['vmed'].clip(lower=10) 
        k_qv = df['intensidad'] / safe_speed
        
        # Case B: Congestion / Stop
        # If occupancy exists, use it. M30 usually has 'ocupacion' (0-100)
        if 'ocupacion' in df.columns:
            # Factor ~3.5 to 4.0 converts Occupancy% to Veh/km (approx)
            k_occ = df['ocupacion'] * 3.5 
        else:
            # Fallback for congestion without occupancy: Max Density assumption?
            # Or just Q/V with very low V (high density).
            k_occ = k_qv 
            
        df['density'] = np.where(condition_normal, k_qv, k_occ)
        
        # Cap outliers (Jam density usually maxes ~150-200 per lane. M30 whole section ~400-500)
        df['density'] = df['density'].clip(upper=500.0)

        # --- 2. Traffic States (Categorical) ---
        # Level of Service (LOS) Approximation
        # A/B (Free): V > 70
        # C/D (Dense): 40 < V <= 70
        # E/F (Congested): V <= 40
        def categorize_state(row):
            v = row['vmed']
            if v > 70: return "Fluido"
            elif v > 40: return "Denso"
            else: return "Congestion"
            
        df['traffic_state'] = df.apply(categorize_state, axis=1)
        
        # --- 3. Time Features & Rush Hour ---
        if 'fecha' in df.columns:
            df['hour'] = df['fecha'].dt.hour
            df['day_of_week'] = df['fecha'].dt.dayofweek
            
            # Rush Hour Flag (Madrid typical: 7-10 AM, 17-20 PM Workdays)
            is_workday = df['day_of_week'] < 5
            is_morning_rush = (df['hour'] >= 7) & (df['hour'] < 10)
            is_afternoon_rush = (df['hour'] >= 17) & (df['hour'] < 20)
            
            df['is_rush_hour'] = is_workday & (is_morning_rush | is_afternoon_rush)

        # --- 4. Rolling Trends (Volatility/Shockwaves) ---
        # Need to sort by ID and Date first
        df = df.sort_values(by=['id', 'fecha'])
        
        # Rolling window of 1 hour (4 periods of 15 min)
        # We group by ID to avoid bleeding between sensors
        indexer = df.groupby('id').rolling(window=4, min_periods=1)
        
        # Reset index levels created by groupby rolling if necessary, or assign directly
        # Direct assignment with groupby transform is safer for preserving shape
        df['vmed_rolling_mean'] = df.groupby('id')['vmed'].transform(lambda x: x.rolling(4, min_periods=1).mean())
        df['vmed_volatility'] = df.groupby('id')['vmed'].transform(lambda x: x.rolling(4, min_periods=1).std()).fillna(0)
        
        # Flow Trend (is it increasing?)
        df['flow_trend'] = df.groupby('id')['intensidad'].transform(lambda x: x.diff())

        # --- 5. Prediction Target (Next 15 min) ---
        df['density_pred'] = df.groupby('id')['density'].shift(-1).fillna(method='ffill')
        
        return df

    def get_quality_report(self):
        """Returns the dictionary with quality metrics from the last clean_data run."""
        return self.quality_report

