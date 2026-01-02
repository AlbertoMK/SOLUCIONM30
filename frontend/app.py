import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data, load_metadata
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW, M30_EAST_SENSORS
from src.physics import TrafficPhysics
from src.optimizer import TrafficOptimizer

st.set_page_config(page_title="M-30 Digital Twin", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .road-container {
        position: relative;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border: 2px solid #555;
        transition: background-color 0.5s ease;
    }
    .lane-marking {
        position: absolute;
        width: 100%;
        height: 4px;
        background: repeating-linear-gradient(90deg, #fff 0px, #fff 40px, transparent 40px, transparent 80px);
        top: 50%;
    }
    .car-overlay {
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        z-index: 10;
        text-shadow: 2px 2px 4px #000;
        background-color: rgba(0,0,0,0.3);
        padding: 5px 15px;
        border-radius: 5px;
    }
    
    /* Metrics Layout */
    .metrics-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #444;
        margin-top: 5px;
    }
    .metric-item {
        text-align: center;
    }
    .metric-label {
        font-size: 0.9em;
        color: #aaa;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4em;
        font-weight: bold;
        color: white;
    }
    
    /* Speed Sign */
    .speed-sign {
        width: 60px;
        height: 60px;
        background-color: white;
        border: 6px solid #cc0000;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        color: black;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    /* Clock */
    .digital-clock {
        text-align: center; 
        background-color: #000; 
        color: #0f0; 
        font-family: 'Courier New', Courier, monospace; 
        padding: 10px; 
        border-radius: 10px; 
        border: 2px solid #555;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚇 M-30 Traffic Simulation (Digital Twin)")

@st.cache_data
def load_all_data(file_name="01-2019.csv"):
    # 1. Traffic Data
    file_path = DATA_PATH_RAW / "trafico" / "01-2019" / file_name
    # 2. Metadata (Map)
    meta_path = DATA_PATH_RAW / "meta" / "pmed_ubicacion_10_2018.csv"
    
    with st.spinner(f"Loading data..."):
        if not file_path.exists():
            return pd.DataFrame(), pd.DataFrame()
        
        df = load_csv_data(file_path)
        meta_df = load_metadata(meta_path)
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_features(df_clean)
    
    return df_features, meta_df

def get_road_color(speed):
    # Map speed (0-90) to Red-Green
    # 0 km/h -> Red (255, 0, 0)
    # 90 km/h -> Green (0, 255, 0)
    # 45 km/h -> Yellow (255, 255, 0)
    
    speed = max(0, min(90, speed))
    factor = speed / 90.0
    
    # Simple Linear Interpolation
    r = int(255 * (1 - factor))
    g = int(255 * factor)
    b = 0
    
    # Boost colors for "Neon" look
    # If speed is low, make it very red.
    if speed < 30: r = 255; g = 0;
    elif speed > 70: r = 0; g = 255;
    
    return f"rgb({r}, {g}, {b})"

# --- SESSION STATE ---
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'selected_sensor' not in st.session_state:
    st.session_state.selected_sensor = None

# --- LOAD DATA ---
# 1. Day Selection (Before loading data to potentially filter file loading in future)
# Moved Day Selection to Sidebar start, but we need data for map first.
# Let's keep logic: Load default -> Show Map -> User selects -> Filter.

st.sidebar.header("🕹️ Simulation Controls")
selected_date = st.sidebar.date_input("📅 Date to Analyze", value=pd.to_datetime("2019-01-01"))
demo_file = "01-2019.csv" 

df_raw, df_meta = load_all_data(demo_file)

if df_raw.empty:
    st.error("Data not found.")
    st.stop()

# --- MAP VISUALIZATION & SELECTION ---
st.markdown("### 🗺️ Select Sensor from Map")

# Filter metadata to only include sensors present in raw data
valid_sensors = df_raw['id'].unique()
map_data = df_meta[df_meta['id'].isin(valid_sensors)].copy()

# Define Color and Size based on selection
map_data['color'] = 'blue'
map_data['size'] = 10
if st.session_state.selected_sensor:
    mask = map_data['id'] == st.session_state.selected_sensor
    map_data.loc[mask, 'color'] = 'red'
    map_data.loc[mask, 'size'] = 20

# Add hover info
if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data, 
        lat="latitud", 
        lon="longitud", 
        hover_name="nombre",
        hover_data=["id", "distrito"],
        zoom=11, 
        height=350,
        color="color",
        size="size",
        color_discrete_map={'blue': '#3498db', 'red': '#e74c3c'},
        size_max=20
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False
    )
    # fig_map.update_traces(marker=dict(size=12, color='blue')) # Removed in favor of dynamic size/color
    
    # Enable Selection (Streamlit 1.37+)
    selection = st.plotly_chart(fig_map, on_select="rerun", use_container_width=True)
    
    # Handle Selection Event
    if selection and selection['selection']['points']:
        point_idx = selection['selection']['points'][0]['point_index']
        # The chart might resort data, but px usually keeps order if not aggregating? 
        # Safest to use custom data, but let's assume index matches for now 
        # or better: rely on the filtering we did.
        # Actually, clicking a point returns the index inthe dataframe passed to plotly.
        # Since we pass map_data, point_index 0 refers to map_data.iloc[0].
        sensor_id_selected = map_data.iloc[point_idx]['id']
        
        # Only update if changed to avoid loop
        if st.session_state.selected_sensor != sensor_id_selected:
            st.session_state.selected_sensor = sensor_id_selected
            st.rerun()

# Fallback if no map selection or first run
available_sensors = sorted(valid_sensors)
default_idx = 0
if st.session_state.selected_sensor in available_sensors:
    default_idx = available_sensors.index(st.session_state.selected_sensor)

# --- CONTROLS SIDEBAR ---
# 2. Sensor Selection
selected_sensor = st.sidebar.selectbox("📍 Sensor Location", options=available_sensors, index=default_idx)

# Sync Sidebar -> Session State (if changed manually)
if selected_sensor != st.session_state.selected_sensor:
    st.session_state.selected_sensor = selected_sensor

# Process Data for Sensor
sensor_data = df_raw[df_raw['id'] == selected_sensor].copy()
k_crit = TrafficPhysics.calculate_critical_density(sensor_data)
optimizer = TrafficOptimizer(critical_density_override=k_crit)
df_opt = optimizer.optimize_traffic(sensor_data)
df_opt['simulated_density'] = df_opt.apply(
    lambda row: row['intensidad'] / row['simulated_speed'] if row['simulated_speed'] > 0 else 0, axis=1
)

st.sidebar.markdown("---")

# 3. Speed Control
speed_factor = st.sidebar.slider("⏩ Simulation Speed", min_value=1, max_value=20, value=5, help="Increase to make time fly.")
# Base sleep time for real-time feel (2 min duration for 24h)
# 1440 mins / (120s * 10fps) = 1.2 real mins per frame.
# We will just step through the dataframe.
# Simulation Speed simply reduces sleep time.

# 4. Filter to Date
# Filter df_opt to selected date (ignoring year/month if we are reusing sample data for demo)
# For demo purposes, we usually just take the first 24h of the sample.
unique_dates = df_opt['fecha'].dt.date.unique()
if selected_date not in unique_dates:
    st.warning(f"Data for {selected_date} not in sample. Using first available date: {unique_dates[0]}")
    target_date = unique_dates[0]
else:
    target_date = selected_date

daily_data = df_opt[df_opt['fecha'].dt.date == target_date].sort_values('fecha').reset_index(drop=True)

# Resample for smooth animation (1 minute intervals)
daily_data_resampled = daily_data.set_index('fecha').resample('1T').interpolate(method='linear').reset_index()

# --- MAIN INTERFACE ---

# Layout: Clock | Slider
col_clock, col_controls = st.columns([1, 2])
current_frame = st.slider("⏱️ Time Scrubber", 0, len(daily_data_resampled)-1, 0, format="")

with col_clock:
    clock_ph = st.empty()

# Extract Current Row based on Slider (Initial)
row = daily_data_resampled.iloc[current_frame]
curr_time_str = row['fecha'].strftime("%H:%M")

clock_ph.markdown(f"""
<div class="digital-clock">
    <div style="font-size: 1.2em; color: #888;">{target_date}</div>
    <div style="font-size: 3em; font-weight: bold;">{curr_time_str}</div>
</div>
""", unsafe_allow_html=True)

with col_controls:
    start_btn = st.button("▶️ START / RESUME", use_container_width=True)
    stop_btn = st.button("⏸️ PAUSE", use_container_width=True)

if start_btn:
    st.session_state.simulation_running = True
if stop_btn:
    st.session_state.simulation_running = False

# --- LAYOUT VISUALS ---
col1, col2 = st.columns(2)

# Placeholders for dynamic content
with col1:
    st.markdown("### 🔴 REALITY")
    real_road_ph = st.empty()
    real_metrics_ph = st.empty()

with col2:
    st.markdown("### 🟢 DIGITAL TWIN")
    opt_road_ph = st.empty()
    opt_metrics_ph = st.empty()

def render_frame(data_row):
    # Reality
    r_speed = data_row['vmed']
    r_dens = data_row['density']
    r_color = get_road_color(r_speed)
    
    real_road_ph.markdown(f"""
    <div class="road-container" style="background-color: {r_color};">
        <div class="lane-marking"></div>
        <div class="car-overlay">{r_speed:.0f} km/h</div>
    </div>
    """, unsafe_allow_html=True)
    
    real_metrics_ph.markdown(f"""
    <div class="metrics-container">
        <div class="metric-item">
            <div class="metric-label">Density</div>
            <div class="metric-value">{r_dens:.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Limit</div>
             <div class="speed-sign">90</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Optimized
    o_speed = data_row['simulated_speed']
    o_dens = data_row['simulated_density']
    o_limit = int(data_row['optimal_speed_limit'])
    o_color = get_road_color(o_speed)
    
    opt_road_ph.markdown(f"""
    <div class="road-container" style="background-color: {o_color};">
        <div class="lane-marking"></div>
        <div class="car-overlay">{o_speed:.0f} km/h</div>
    </div>
    """, unsafe_allow_html=True)
    
    opt_metrics_ph.markdown(f"""
    <div class="metrics-container">
        <div class="metric-item">
            <div class="metric-label">Density</div>
             <div class="metric-value">{o_dens:.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Limit</div>
             <div class="speed-sign" style="border-color: {'#cc0000' if o_limit==90 else '#ffcc00'};">{o_limit}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initial Render (Static)
render_frame(row)

# --- ANIMATION LOGIC ---
if st.session_state.simulation_running:
    # We run a loop from current_frame to end
    start_idx = current_frame
    
    for i in range(start_idx, len(daily_data_resampled)):
        loop_row = daily_data_resampled.iloc[i]
        
        # Update Clock
        curr_time_str = loop_row['fecha'].strftime("%H:%M")
        clock_ph.markdown(f"""
        <div class="digital-clock">
            <div style="font-size: 1.2em; color: #888;">{target_date}</div>
            <div style="font-size: 3em; font-weight: bold;">{curr_time_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # RENDER VISUALS
        render_frame(loop_row)
        
        # Sleep based on Speed Factor
        # Base: 2 mins (120s) for 1440 frames -> 0.08s per frame.
        # Speed Factor 1 = 0.08s. Speed 10 = 0.008s.
        base_sleep = 0.08
        actual_sleep = base_sleep / speed_factor
        time.sleep(actual_sleep)
        
    st.session_state.simulation_running = False
    st.rerun()

