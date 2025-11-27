import pandapower as pp
import numpy as np
import logging

class GridPhysics:
    def __init__(self):
        self.net = self._build_network()
        
        # --- THERMAL STATE VARIABLES (IEEE C57.91) ---
        # We assume the simulation starts at steady state with ambient temperature
        self.prev_top_oil_rise = 0.0  
        self.tau_oil = 180.0     # Oil time constant (minutes) - typical for 10MVA
        self.dt = 60.0           # Simulation timestep (minutes)
        
    def _build_network(self):
        """
        Builds a single-feeder distribution model.
        115 kV Source -> 10 MVA Transformer -> 13.8 kV Feeder
        """
        net = pp.create_empty_network()
        hv_bus = pp.create_bus(net, vn_kv=115, name="HV Substation")
        pp.create_ext_grid(net, bus=hv_bus, vm_pu=1.02)
        lv_bus = pp.create_bus(net, vn_kv=13.8, name="Feeder Head")
        
        pp.create_transformer_from_parameters(
            net, hv_bus=hv_bus, lv_bus=lv_bus, 
            sn_mva=10, vn_hv_kv=115, vn_lv_kv=13.8, 
            vkr_percent=0.5, vk_percent=10, 
            pfe_kw=10, i0_percent=0.1, name="Main Tx"
        )
        pp.create_load(net, bus=lv_bus, p_mw=0, q_mvar=0, name="Feeder Load")
        return net

    def solve_power_flow(self, load_mw):
        """Runs Newton-Raphson Power Flow."""
        q_mvar = load_mw * 0.33  # 0.95 PF assumption
        
        self.net.load.loc[0, 'p_mw'] = load_mw
        self.net.load.loc[0, 'q_mvar'] = q_mvar
        
        try:
            pp.runpp(self.net, algorithm='nr', max_iteration=50, init='flat')
            loading_percent = self.net.res_trafo.loading_percent[0]
            voltage_pu = self.net.res_bus.vm_pu[1]
            return loading_percent, voltage_pu
        except Exception:
            logging.warning(f"Grid Collapse at {load_mw:.2f} MW!")
            return 150.0, 0.85

    def calculate_thermal_aging(self, loading_percent, ambient_temp_c):
        """
        Calculates Hot Spot Temp (HST) using IEEE C57.91 Difference Equations.
        Tracks state from the previous timestep.
        """
        # Parameters for a typical ONAN Transformer
        rated_top_oil_rise = 50.0
        rated_hot_spot_rise = 15.0
        m = 0.8  # Oil exponent
        n = 1.6  # Winding exponent
        
        K = loading_percent / 100.0
        
        # --- 1. Top Oil Rise (Transient) ---
        # Target rise at this specific load (if held forever)
        target_oil_rise = rated_top_oil_rise * (K ** m)
        
        # Difference Equation: New = Old + (Target - Old) * (1 - e^(-dt/tau))
        change = (target_oil_rise - self.prev_top_oil_rise) * (1 - np.exp(-self.dt / self.tau_oil))
        current_top_oil_rise = self.prev_top_oil_rise + change
        
        # Update State for next loop
        self.prev_top_oil_rise = current_top_oil_rise
        
        # --- 2. Hot Spot Gradient (Instantaneous) ---
        # We assume winding heats much faster than oil, so we treat this as steady-state for 1-hour steps
        hot_spot_gradient = rated_hot_spot_rise * (K ** n)
        
        # --- 3. Total HST ---
        hst = ambient_temp_c + current_top_oil_rise + hot_spot_gradient
        
        # --- 4. Aging Factor (FAA) ---
        # IEEE standard aging equation
        if hst > 110:
            faa = np.exp(15000/383 - 15000/(hst + 273))
        else:
            faa = 1.0 # Standard aging
            
        return hst, faa