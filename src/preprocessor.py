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
    
    def __init__(self):
        self.sensor_ids = M30_EAST_SENSORS

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic cleaning: filtering sensors, handling NaNs, logic checks.
        
        Args:
            df (pd.DataFrame): Raw data.
            
        Returns:
            pd.DataFrame: Cleaned data.
        """
        if df.empty:
            return df

        # 1. Filter by Sensor ID
        # Ensure ID column type matches config (handling string/int mismatch)
        # We try to convert both to string for robust comparison or keep mixed if needed.
        # Given our config has mixed types (int and str), let's ensure we match correctly.
        # For now, we assume the raw data IDs might be ints or strs.
        
        valid_ids = set(self.sensor_ids)
        mask = df['id'].isin(valid_ids)
        df_clean = df[mask].copy()
        
        # 2. Parse Dates
        # Format in csv: "YYYY-MM-DD HH:MM:SS"
        if 'fecha' in df_clean.columns:
            df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')
        
        # 3. Sort by time
        df_clean = df_clean.sort_values(by=['id', 'fecha'])

        # 4. Remove logical errors
        # Speed (vmed) < 0 or Intensity (intensidad) < 0
        df_clean = df_clean[
            (df_clean['vmed'] >= 0) & 
            (df_clean['intensidad'] >= 0)
        ]
        
        # 5. Handle Missing Values
        # Interpolate linear per group (sensor)
        # We need to set index to time to interpolate properly, or just use linear on columns
        # Ideally, we should pivot or group by ID.
        # Simple approach: Group by ID, then apply interpolation.
        df_clean['vmed'] = df_clean.groupby('id')['vmed'].transform(lambda x: x.interpolate(method='linear'))
        df_clean['intensidad'] = df_clean.groupby('id')['intensidad'].transform(lambda x: x.interpolate(method='linear'))
        
        # Fill remaining NaNs (edges) with ffill/bfill or 0
        df_clean = df_clean.ffill().bfill()
        
        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features for the ML model: Density, Time features.
        
        Args:
            df (pd.DataFrame): Cleaned data.
            
        Returns:
            pd.DataFrame: Data with new features.
        """
        df = df.copy()
        
        # 1. Density (K) = Intensity (limit/hour) / Speed (km/h)
        # Result is vehicles/km.
        # Handle division by zero: if speed is 0, density is technically infinite (jam).
        # We replace 0 with a small epsilon.
        epsilon = 1e-5
        df['density'] = df['intensidad'] / (df['vmed'] + epsilon)
        
        # 2. Time Features
        if 'fecha' in df.columns:
            df['hour'] = df['fecha'].dt.hour
            df['day_of_week'] = df['fecha'].dt.dayofweek
            df['month'] = df['fecha'].dt.month
            
        # 3. Predicted Density (t + 15min) -> Shift -1
        # Assuming data is sorted by id, fecha and 15 min intervals
        df['density_pred'] = df.groupby('id')['density'].shift(-1)
        # Fill last value with current
        df['density_pred'] = df['density_pred'].fillna(df['density'])
        
        return df
