"""
Traffic Physics module.
Handles Fundamental Diagram calculations and traffic state estimation.
"""
import pandas as pd
import numpy as np

class TrafficPhysics:
    """
    Class for handling traffic flow physics calculations.
    Based on the Fundamental Diagram of Traffic Flow (Greenshields, etc.).
    """
    
    @staticmethod
    def calculate_density(intensity: float, speed: float) -> float:
        """
        Calculates traffic density based on flow (intensity) and speed.
        
        Formula: k = q / v
        
        Args:
            intensity (float): Traffic flow in vehicles/hour.
            speed (float): Average speed in km/h.
            
        Returns:
            float: Density in vehicles/km. Returns 0 if speed is 0 to avoid division by zero.
        """
        if speed <= 0:
            return 0.0
        return intensity / speed

    @staticmethod
    def calculate_critical_density(df: pd.DataFrame) -> float:
        """
        Calculates the Critical Density (Kc) where Flow (Intensity) is maximized.
        Uses binning to find the peak of the Fundamental Diagram.
        
        Args:
            df (pd.DataFrame): Data with 'density' and 'intensidad'.
            
        Returns:
            float: Critical Density value. Returns default from config if calculation fails.
        """
        try:
            # Drop NaNs
            data = df[['density', 'intensidad']].dropna()
            
            if data.empty:
                return 40.0 # Default fallback
            
            # Bin density to smooth out noise
            # Create bins of 5 veh/km
            data['density_bin'] = (data['density'] // 5) * 5
            
            # Calculate mean flow per bin
            avg_flow = data.groupby('density_bin')['intensidad'].mean()
            
            # Find bin with max flow
            critical_density = avg_flow.idxmax()
            
            return float(critical_density)
        except Exception as e:
            print(f"Error calculating critical density: {e}")
            print(f"Error calculating critical density: {e}")
            return 45.0 # Conservative default

    @staticmethod
    def calculate_max_capacity(df: pd.DataFrame) -> float:
        """
        Calculates Maximum Capacity (Q_max) of the road section.
        Suggested: 99th percentile of historical intensity.
        
        Args:
            df (pd.DataFrame): Data with 'intensidad'.
            
        Returns:
            float: Max capacity in veh/h.
        """
        try:
            if 'intensidad' not in df.columns:
                return 4000.0
            
            # Use 99th percentile to filter out sensor errors/spikes
            q_max = df['intensidad'].quantile(0.99)
            return float(q_max)
        except Exception:
            return 4000.0 # Default for M-30 (3-4 lanes)

    @staticmethod
    def get_fundamental_diagram(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for plotting the Fundamental Diagram (Flow vs Density).
        
        Args:
            df (pd.DataFrame): DataFrame containing 'intensity' and 'speed' columns.
            
        Returns:
            pd.DataFrame: DataFrame with an added 'density' column, ready for plotting.
        """
        if 'intensidad' not in df.columns or 'vmed' not in df.columns:
            # Attempt renaming or checking
             pass
        
        df_fd = df.copy()
        
        # Ensure density exists or recalc
        if 'density' not in df_fd.columns:
             # vmed is speed, intensidad is flow
             df_fd['density'] = df_fd.apply(
                lambda row: row['intensidad'] / row['vmed'] if row['vmed'] > 0 else 0, axis=1
             )
        
        return df_fd
