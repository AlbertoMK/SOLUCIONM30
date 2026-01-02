import pandas as pd
import numpy as np
from .config import CRITICAL_DENSITY_THRESHOLD

class TrafficOptimizer:
    """
    Optimizer engine for Variable Speed Limits (VSL).
    """
    
    def __init__(self, critical_density_override: float = None):
        self.critical_density = critical_density_override

    def optimize_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies logic to determine Optimal Speed Limit and calculates Simulated Speed.
        
        Args:
            df (pd.DataFrame): Cleaned data with 'density' and 'density_pred'.
            
        Returns:
            pd.DataFrame: Data with 'optimal_speed_limit' and 'simulated_speed'.
        """
        df_opt = df.copy()
        
        # 1. Determine Critical Density (if not manually set)
        if self.critical_density is None:
            # We use a default if we can't calculate per sensor, 
            # OR we could import physics to calc it. 
            # For this Phase, usage of global default or passed value is fine.
            threshold = CRITICAL_DENSITY_THRESHOLD # From config (supposedly ~18-40)
            # Let's assume CRITICAL_DENSITY_THRESHOLD in config is actually relevant.
            # However, typically it's around 30-50 veh/km.
            pass
        else:
            threshold = self.critical_density

        # Use 45 as a safe default if config value is weird (config said 18% occupancy, which is ~30-40 veh/km)
        # Let's ensure we have a sane numerical value for Density (veh/km)
        if threshold < 10: 
             threshold = 40.0 # Correction if config was percentage
        
        # 2. Apply VSL Logic
        # Condition: If Predicted Density > Critical -> Limit = 70 km/h
        # Else -> Limit = 90 km/h
        
        if 'density_pred' not in df_opt.columns:
             df_opt['density_pred'] = df_opt['density'] # Fallback
             
        df_opt['optimal_speed_limit'] = np.where(
            df_opt['density_pred'] > threshold, 
            70, 
            90
        )
        
        # 3. Simulate Improvement
        # Model:
        # If Limit applied (70) and traffic was unstable (High Density):
        # We assume the VSL smooths flow, slightly increasing AVG speed compared to the "Stop & Go" reality.
        # Improvement factor: +15% to speed, but capped at the Speed Limit.
        # If Limit applied (70) but traffic was fast (just above critical density):
        # We cap speed at 70 (Compliance).
        
        def simulate_speed(row):
            actual_v = row['vmed']
            limit = row['optimal_speed_limit']
            
            if limit == 70:
                # Logic: We are intervening.
                if actual_v < 60:
                    # Congested regime. VSL reduces shockwaves -> Speed improves.
                    return min(actual_v * 1.15, 70.0) 
                else:
                    # Free flow or near capacity. VSL forces slowdown/harmonization.
                    return min(actual_v, 70.0)
            else:
                # No intervention.
                return actual_v
                
        df_opt['simulated_speed'] = df_opt.apply(simulate_speed, axis=1)
        
        return df_opt

