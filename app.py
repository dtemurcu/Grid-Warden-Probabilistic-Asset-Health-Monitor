import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from src.data_loader import DataLoader
from src.features import add_grid_features
from src.forecasting import GridForecaster
from src.ev_simulator import EVSimulator
from src.physics import GridPhysics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Grid Warden | Asset Health", layout="wide")

st.title("⚡ Grid Warden: Transformer Life & EV Impact")
st.markdown("""
**Objective:** Quantify the **Financial & Physical Loss of Life (LoL)** on distribution assets due to EV adoption.
**Physics Engine:** IEEE C57.91 (Arrhenius Aging Model).
""")

# --- CACHED DATA LOADING ---
@st.cache_resource
def load_and_train():
    loader = DataLoader(raw_path="data/raw", processed_path="data/processed")
    forecaster = GridForecaster()
    
    df_load = loader.load_ieso_demand("PUB_Demand_2024.csv")
    df_weather = loader.load_weather("en_climate_hourly_*.csv")
    df_merged = loader.merge_data(df_load, df_weather)
    
    df_features = add_grid_features(df_merged)
    test_df, predictions = forecaster.train(df_features)
    return test_df, predictions, df_features

with st.spinner("Initializing Grid Physics Engine..."):
    test_df, predictions, df_features = load_and_train()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Simulation Parameters")

# 1. Date & Feeder
st.sidebar.subheader("1. Asset Context")
peak_idx = predictions.argmax()
default_date = test_df.index[peak_idx].date()
selected_date = st.sidebar.date_input("Simulation Day", default_date)

base_load_peak_target = st.sidebar.slider("Feeder Peak (MW)", 2.0, 9.0, 5.0)
transformer_limit = st.sidebar.number_input("Nameplate Limit (MW)", value=10.0, step=0.5)

# 2. EV Fleet
st.sidebar.subheader("2. EV Stress")
num_evs = st.sidebar.slider("Number of EVs", 0, 5000, 1500)
charging_mode = st.sidebar.selectbox("Charging Logic", ["uncontrolled", "ulo_timer", "smart_managed"])

# --- SIMULATION LOGIC ---

# 1. Get Base Load
day_mask = (test_df.index.date == selected_date)
if not any(day_mask): st.stop()

base_load_prov = predictions[day_mask]
ambient_temp = test_df.loc[day_mask, 'temp_c'].values

if len(base_load_prov) != 24:
    base_load_prov = predictions[:24]
    ambient_temp = test_df['temp_c'].values[:24]

# 2. Scale Base Load
scaling_factor = base_load_peak_target / base_load_prov.max()
base_load_feeder = base_load_prov * scaling_factor

# 3. Add EVs
ev_sim = EVSimulator(num_evs=num_evs)
ev_profile = ev_sim.generate_load_profile(mode=charging_mode)
total_load = base_load_feeder + ev_profile

# 4. Run Physics (Power Flow + Aging)
grid = GridPhysics()
hst_results = []
faa_results = [] # Acceleration Factors

for h in range(24):
    loading, _ = grid.solve_power_flow(total_load[h])
    hst, faa = grid.calculate_thermal_aging(loading, ambient_temp[h])
    hst_results.append(hst)
    faa_results.append(faa)

# --- METRICS ---
peak_load = total_load.max()
peak_temp = max(hst_results)

# Calculate Loss of Life
# Standard day = 24 hours of aging.
# If we operated at 110C all day, we age 24 hours.
# If FAA > 1, we age faster.
total_aging_hours = sum(faa_results)
aging_acceleration = total_aging_hours / 24.0

col1, col2, col3 = st.columns(3)
col1.metric("Peak Load", f"{peak_load:.2f} MW", delta=f"{peak_load - transformer_limit:.2f} MW", delta_color="inverse")
col2.metric("Max Temp", f"{peak_temp:.0f} °C", delta=f"{110 - peak_temp:.0f} Margin", delta_color="normal" if peak_temp < 110 else "inverse")

# The "Money" Metric
col3.metric("Effective Aging (24h)", f"{total_aging_hours:.1f} Hours", 
            delta="Normal" if total_aging_hours <= 24 else f"Aged {total_aging_hours/24:.1f}x Faster",
            delta_color="inverse")

# --- PLOTS ---

# Tabbed Interface for cleaner look
tab1, tab2 = st.tabs(["📉 Load Profile", "🔥 Asset Health"])

with tab1:
    fig_load = go.Figure()
    # Stacked Area for Context
    fig_load.add_trace(go.Scatter(x=list(range(24)), y=base_load_feeder, name="Base Load", line=dict(dash='dot', color='gray')))
    fig_load.add_trace(go.Scatter(x=list(range(24)), y=total_load, name="Total + EVs", line=dict(color='blue', width=3), fill='tonexty'))
    fig_load.add_hline(y=transformer_limit, line_dash="dash", line_color="red", annotation_text="Limit")
    fig_load.update_layout(title="Daily Load Profile", xaxis_title="Hour", yaxis_title="MW", hovermode="x unified")
    st.plotly_chart(fig_load, use_container_width=True)

with tab2:
    # Double Axis: Temp vs Aging Factor
    fig_temp = go.Figure()
    
    # Temp (Left)
    fig_temp.add_trace(go.Scatter(x=list(range(24)), y=hst_results, name="Hot Spot Temp", line=dict(color='red', width=3)))
    fig_temp.add_hline(y=110, line_dash="dot", line_color="black", annotation_text="Damage Threshold")
    
    # Aging Factor (Right)
    # This shows the EXPONENTIAL nature of damage
    fig_temp.add_trace(go.Scatter(x=list(range(24)), y=faa_results, name="Aging Factor (Speed)", line=dict(color='orange', dash='dot'), yaxis="y2"))
    
    fig_temp.update_layout(
        title="Thermal Stress & Aging Acceleration",
        xaxis_title="Hour",
        yaxis=dict(title="Temperature (°C)"),
        yaxis2=dict(title="Aging Factor (x Normal)", overlaying="y", side="right"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

st.caption("Simulation powered by Pandapower & XGBoost. © 2025 Grid Warden.")