import pandapower as pp
import numpy as np
import logging

class GridPhysics:
    def __init__(self):
        self.net = self._build_network()
        
    def _build_network(self):
        """
        Builds a single-feeder distribution model.
        115 kV Source -> 10 MVA Transformer -> 13.8 kV Feeder
        """
        net = pp.create_empty_network()
        
        # 1. Grid Connection (Slack Bus)
        hv_bus = pp.create_bus(net, vn_kv=115, name="HV Substation")
        pp.create_ext_grid(net, bus=hv_bus, vm_pu=1.02)
        
        # 2. Distribution Bus
        lv_bus = pp.create_bus(net, vn_kv=13.8, name="Feeder Head")
        
        # 3. The Transformer (The Asset we are protecting)
        pp.create_transformer_from_parameters(
            net, hv_bus=hv_bus, lv_bus=lv_bus, 
            sn_mva=10, vn_hv_kv=115, vn_lv_kv=13.8, 
            vkr_percent=0.5, vk_percent=10, 
            pfe_kw=10, i0_percent=0.1, name="Main Tx"
        )
        
        # 4. Aggregated Load (Placeholders)
        pp.create_load(net, bus=lv_bus, p_mw=0, q_mvar=0, name="Feeder Load")
        
        return net

    def solve_power_flow(self, load_mw):
        """
        Runs Newton-Raphson Power Flow.
        Returns: Transformer Loading (%) and Voltage (p.u.)
        """
        # Calculate Reactive Power (Assuming 0.95 PF)
        q_mvar = load_mw * 0.33
        
        # FIX: Use .loc to avoid Pandas FutureWarnings
        self.net.load.loc[0, 'p_mw'] = load_mw
        self.net.load.loc[0, 'q_mvar'] = q_mvar
        
        try:
            # FIX: Use 'flat' initialization and more iterations to handle heavy overloads
            pp.runpp(self.net, algorithm='nr', max_iteration=50, init='flat')
            
            # Extract results
            loading_percent = self.net.res_trafo.loading_percent[0]
            voltage_pu = self.net.res_bus.vm_pu[1] # Index 1 is the LV bus
            return loading_percent, voltage_pu
            
        except Exception as e:
            # If it STILL crashes, it means the grid collapsed (Voltage collapse).
            # We return "Extreme" values to show on the plot that things are bad.
            logging.warning(f"Grid Collapse at {load_mw:.2f} MW! Returning max values.")
            return 150.0, 0.85 # 150% overload, 0.85 p.u. voltage (Brownout)

    def calculate_thermal_aging(self, loading_percent, ambient_temp_c):
        """
        Estimates Transformer Hot Spot Temperature (HST) using IEEE C57.91 logic.
        """
        rated_top_oil_rise = 50.0
        rated_hot_spot_rise = 15.0
        
        # Load Factor K
        K = loading_percent / 100.0
        
        # 1. Top Oil Rise
        top_oil_rise = rated_top_oil_rise * (K ** 1.6)
        
        # 2. Hot Spot Gradient
        hot_spot_gradient = rated_hot_spot_rise * (K ** 1.6)
        
        # 3. Total HST
        hst = ambient_temp_c + top_oil_rise + hot_spot_gradient
        
        # 4. Aging Factor (FAA)
        if hst > 110:
            faa = np.exp(15000/383 - 15000/(hst + 273))
        else:
            faa = 1.0
            
        return hst, faa