from src.data_loader import DataLoader
from src.features import add_grid_features
from src.forecasting import GridForecaster
from src.ev_simulator import EVSimulator
from src.physics import GridPhysics
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    loader = DataLoader(raw_path="data/raw", processed_path="data/processed")
    forecaster = GridForecaster()
    
    try:
        # 1. LOAD & TRAIN
        df_load = loader.load_ieso_demand("PUB_Demand_2024.csv")
        df_weather = loader.load_weather("en_climate_hourly_*.csv")
        df_merged = loader.merge_data(df_load, df_weather)
        
        if df_merged.empty: raise ValueError("Dataset empty!")

        df_features = add_grid_features(df_merged)
        loader.save_processed(df_features, "features_ready.csv")
        
        test_df, predictions = forecaster.train(df_features)
        
        # 2. SIMULATION SETUP
        peak_hour_idx = predictions.argmax()
        target_date = test_df.index[peak_hour_idx].date()
        
        logging.info(f"Running Physics Simulation for: {target_date}")
        
        day_mask = (test_df.index.date == target_date)
        base_load_provincial = predictions[day_mask]
        ambient_temp = test_df.loc[day_mask, 'temp_c'].values
        
        # Fallback
        if len(base_load_provincial) != 24: 
            base_load_provincial = predictions[:24]
            ambient_temp = test_df['temp_c'].values[:24]

        # SCALE DOWN (Province -> Feeder)
        # We target a 5 MW Peak Base Load
        scaling_factor = 5.0 / base_load_provincial.max()
        base_load_feeder = base_load_provincial * scaling_factor

        # 3. EV SCENARIOS
        # 1000 EVs = High Penetration
        ev_sim = EVSimulator(num_evs=1000) 
        
        # A. The Problem (ULO Timer)
        ev_ulo = ev_sim.generate_load_profile('ulo_timer')
        load_ulo = base_load_feeder + ev_ulo
        
        # B. The Solution (Smart Managed)
        ev_smart = ev_sim.generate_load_profile('smart_managed')
        load_smart = base_load_feeder + ev_smart
        
        # METRIC: Calculate Peak Reduction
        peak_ulo = load_ulo.max()
        peak_smart = load_smart.max()
        reduction_pct = ((peak_ulo - peak_smart) / peak_ulo) * 100
        
        logging.info(f"--- RESULTS ---")
        logging.info(f"ULO Peak Load:   {peak_ulo:.2f} MW (Grid Crash Risk)")
        logging.info(f"Smart Peak Load: {peak_smart:.2f} MW (Managed)")
        logging.info(f"PEAK REDUCTION:  {reduction_pct:.1f}%")
        
        # 4. PHYSICS ENGINE (Compare Scenarios)
        grid = GridPhysics()
        
        hst_ulo = []
        hst_smart = []
        
        for h in range(24):
            # Run ULO
            loading_u, _ = grid.solve_power_flow(load_ulo[h])
            temp_u, _ = grid.calculate_thermal_aging(loading_u, ambient_temp[h])
            hst_ulo.append(temp_u)
            
            # Run Smart
            loading_s, _ = grid.solve_power_flow(load_smart[h])
            temp_s, _ = grid.calculate_thermal_aging(loading_s, ambient_temp[h])
            hst_smart.append(temp_s)

        # 5. VISUALIZATION
        os.makedirs("outputs/figures", exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        hours = range(24)
        
        # Plot Loads
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Total Load (MW)', color='black')
        
        ax1.plot(hours, base_load_feeder, color='gray', linestyle=':', label='Base Load', alpha=0.5)
        ax1.plot(hours, load_ulo, color='#1f77b4', linewidth=2, label='Scenario 1: ULO (Timer Peak)')
        ax1.plot(hours, load_smart, color='#2ca02c', linewidth=2, label='Scenario 2: Smart Managed')
        
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot Temps
        ax2 = ax1.twinx()
        ax2.set_ylabel('Transformer Hot Spot Temp (°C)', color='#d62728')
        
        ax2.plot(hours, hst_ulo, color='#d62728', linestyle='--', label='Temp (ULO)')
        ax2.plot(hours, hst_smart, color='#ff9896', linestyle='--', label='Temp (Smart)')
        
        ax2.axhline(110, color='red', linestyle=':', label='Limit (110°C)')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        plt.title(f"Mitigation Analysis: Smart Charging Reduces Peak by {reduction_pct:.0f}%")
        plt.tight_layout()
        
        plt.savefig("outputs/figures/06_mitigation_analysis.png")
        logging.info("Saved Final Plot: outputs/figures/06_mitigation_analysis.png")

    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")
        raise e

if __name__ == "__main__":
    main()