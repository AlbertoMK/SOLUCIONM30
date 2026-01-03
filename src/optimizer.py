import pandas as pd
import numpy as np
from .physics import TrafficPhysics

class TrafficOptimizer:
    """
    Optimizer engine for Variable Speed Limits (VSL).
    Implements logic to recalculate Flow (Q) and Speed (V) under optimized conditions.
    """
    
    def __init__(self, critical_density_override: float = None, 
                 max_capacity_override: float = None, 
                 base_speed_limit: int = 90):
        self.critical_density = critical_density_override
        self.max_capacity = max_capacity_override
        self.base_speed_limit = base_speed_limit

    def _round_speed(self, speed: float) -> int:
        """
        Rounds speed to the nearest multiple of 10 or 5 for valid speed limits.
        """
        return int(round(speed / 10.0) * 10)

    def optimize_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies logic to determine Optimal Speed Limit and calculates Simulated Speed
        based on recovering capacity drop effects.
        """
        df_opt = df.copy()
        
        # 1. Determine Parameters
        k_crit = self.critical_density
        if k_crit is None:
            k_crit = TrafficPhysics.calculate_critical_density(df_opt)
            
        q_max = self.max_capacity
        if q_max is None:
            q_max = TrafficPhysics.calculate_max_capacity(df_opt)
            
        print(f"   [Optimizer] Using K_crit={k_crit:.2f} veh/km, Q_max={q_max:.0f} veh/h, BaseLimit={self.base_speed_limit} km/h")
        
        # Determine VSL Target based on Base Limit (Step Down logic)
        # If 90 -> 70. If 70 -> 50. If 50 -> 30.
        if self.base_speed_limit >= 90:
            v_vsl_target = 70
        elif self.base_speed_limit >= 70:
            v_vsl_target = 50
        else:
            v_vsl_target = 30
            
        # 2. Define Simulation Function
        def simular_escenario(row):
            densidad_real = row.get('density', 0)
            flujo_real = row.get('intensidad', 0)
            velocidad_real = row.get('vmed', 0)
            
            # Param: VSL Activation Speed
            v_limite_activado = v_vsl_target
            
            # Logic: If density > 90% of Critical, Activate VSL
            algoritmo_activo = densidad_real > (k_crit * 0.9)
            
            if not algoritmo_activo:
                # No optimization needed/applied.
                # Return current state, but ensure "Limit" reflects the base limit of road.
                return flujo_real, velocidad_real, self.base_speed_limit
            else:
                # --- OPTIMIZATION LOGIC ---
                
                # Compliance Factor (User: 20% do not comply)
                compliance_rate = 0.8
                
                # 1. Recover Capacity Loop
                # Theoretical max improvement is 10-15%.
                # We scale this by compliance. If people speed, they cause micro-braking -> less recovery.
                base_improvement = 0.10
                real_improvement = base_improvement * compliance_rate
                
                factor_mejora_flujo = 1.0 + real_improvement
                
                # New Estimated Flow (cannot exceed physical Q_max)
                nuevo_flujo = min(flujo_real * factor_mejora_flujo, q_max)
                
                # New Estimated Speed
                # V = Q / K (Assuming K stays roughly constant)
                if densidad_real > 0:
                    nueva_velocidad = nuevo_flujo / densidad_real
                else:
                    nueva_velocidad = velocidad_real
                
                # Corrections
                # 1. Cap at the Dynamic Limit? 
                # If 20% speed, the average speed might slightly exceed the limit IF flow allows.
                # However, usually congestion limits speed more than the sign.
                # Let's enforce that the *harmonized* flow results in a speed that respects the physics.
                
                # Check against Limit:
                # If V_calc > VSL, it implies we *could* go faster relative to density.
                # But we are legally capped.
                # With non-compliance, average speed = 0.8 * VSL + 0.2 * V_speeding (e.g. VSL+10 or Base).
                # But let's simplify: we cap the *optimized* speed to the limit, 
                # allowing a small buffer for non-compliance if flow allows?
                # Actually, strictly, if we set 70, and V_calc comes out as 80, 
                # it means density is low enough to go 80.
                # But we have density > critical, so usually V_calc is LOW (e.g. 20-30).
                # So the Limit usually isn't the binding factor for speed, the Congestion is.
                # The Limit is the binding factor for *Flow Stability*.
                
                # If V_calc is huge (e.g. low density outlier), cap it.
                # Cap at Base Limit (absolute max) and VSL Target (strict max for compliant).
                # Let's cap at Base Speed Limit (physics/road max).
                nueva_velocidad = min(nueva_velocidad, self.base_speed_limit)
                
                # 2. Should not be worse than reality
                nueva_velocidad = max(nueva_velocidad, velocidad_real)
                
                return nuevo_flujo, nueva_velocidad, v_limite_activado

        # 3. Apply
        result = df_opt.apply(simular_escenario, axis=1, result_type='expand')
        df_opt[['intensidad_opt', 'velocidad_opt', 'limite_dinamico']] = result
        
        # 4. Rounding Logic
        df_opt['velocidad_opt'] = df_opt['velocidad_opt'].apply(self._round_speed)
        
        # Aliases
        df_opt['simulated_speed'] = df_opt['velocidad_opt']
        df_opt['optimal_speed_limit'] = df_opt['limite_dinamico']

        return df_opt
