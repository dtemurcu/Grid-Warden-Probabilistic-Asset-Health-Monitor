import numpy as np
import pandas as pd

class EVSimulator:
    def __init__(self, num_evs=500, charging_power_kw=7.0, battery_capacity_kwh=60.0):
        self.num_evs = num_evs
        self.charging_power = charging_power_kw
        self.battery_capacity = battery_capacity_kwh
        
    def generate_load_profile(self, mode='uncontrolled'):
        """
        Generates a 24-hour aggregate load profile (MW).
        Modes:
        - 'uncontrolled': Plug in immediately (Evening Peak).
        - 'ulo_timer': Everyone starts at 11 PM (Timer Peak).
        - 'smart_managed': Utility spreads charging between 11 PM and 4 AM.
        """
        # 1. Simulate Arrival Times
        # Normal distribution around 6 PM
        arrivals = np.random.normal(loc=18.0, scale=2.0, size=self.num_evs)
        
        # 2. Simulate SOC (20-80%)
        arrival_soc = np.random.uniform(0.2, 0.8, size=self.num_evs)
        energy_needed = (1.0 - arrival_soc) * self.battery_capacity
        
        # 3. Charging Duration
        durations = energy_needed / self.charging_power
        
        # 4. Build Curve (Minute resolution)
        # 1440 mins + buffer
        load_curve_min = np.zeros(1440 + 600)
        
        for i in range(self.num_evs):
            arrival_hour = arrivals[i]
            duration_hours = durations[i]
            
            if mode == 'uncontrolled':
                # Start as soon as they arrive
                start_time_min = int(arrival_hour * 60)
                
            elif mode == 'ulo_timer':
                # Everyone waits for 11 PM (23:00)
                # Small random noise (0-15 mins)
                start_time_min = int(23 * 60) + np.random.randint(0, 15)
                
            elif mode == 'smart_managed':
                # SOLUTION: Utility spreads starts over 5 hours (11 PM - 4 AM)
                # This de-synchronizes the fleet
                window_minutes = 5 * 60 
                delay = np.random.randint(0, window_minutes)
                start_time_min = int(23 * 60) + delay
            
            # Handle bounds
            start_time_min = max(0, start_time_min)
            end_time_min = start_time_min + int(duration_hours * 60)
            
            load_curve_min[start_time_min:end_time_min] += self.charging_power
            
        # 5. Downsample to Hourly & Convert to MW
        load_curve_min = load_curve_min[:1440]
        load_curve_hourly = load_curve_min.reshape(24, 60).mean(axis=1)
        
        # Convert kW -> MW
        return load_curve_hourly / 1000.0