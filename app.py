import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import keras
import plotly.graph_objects as go
from datetime import datetime

# ─── 1. PAGE CONFIGURATION ───
st.set_page_config(page_title="SEA Heatwave Radar", page_icon="🥵", layout="wide")

# ─── 2. APP CONSTANTS & CONFIG ───
CITIES = {
    'Jakarta':   {'lat': -6.2088, 'lon': 106.8456},
    'Bangkok':   {'lat': 13.7563, 'lon': 100.5018},
    'Manila':    {'lat': 14.5995, 'lon': 120.9842},
    'Singapore': {'lat': 1.3521,  'lon': 103.8198}
}
WINDOW_SIZE = 14
FEATURES = ['T_max', 'RH_max', 'Wind', 'Heat_Index', 'doy_sin', 'doy_cos']

# ─── 3. CLIMATE MATH FUNCTIONS ───
def calculate_apparent_temperature(t, rh, ws):
    """Australian AT formula"""
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))
    ws_ms = ws / 3.6
    return t + (0.33 * e) - (0.70 * ws_ms) - 4.00

# ─── 4. CACHED ASSET LOADING ───
@st.cache_resource
def load_models(city):
    city_lower = city.lower()
    model = keras.models.load_model(f'models/{city_lower}_lstm.keras')
    scaler = joblib.load(f'models/scaler_{city_lower}.pkl')
    threshold = joblib.load(f'models/threshold_{city_lower}.pkl')
    return model, scaler, threshold

# ─── 5. LIVE API FETCHING (NOW WITH 7-DAY FORECAST) ───
@st.cache_data(ttl=3600) 
def fetch_live_weather(lat, lon):
    # past_days=14 gets the buildup, forecast_days=8 gets today + 7 days into the future
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&past_days={WINDOW_SIZE}&forecast_days=8"
        f"&daily=temperature_2m_max,relative_humidity_2m_max,windspeed_10m_max"
        f"&timezone=Asia%2FSingapore"
    )
    response = requests.get(url).json()
    df = pd.DataFrame(response['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.columns = ['T_max', 'RH_max', 'Wind']
    
    # Apply Feature Engineering
    df['Heat_Index'] = calculate_apparent_temperature(df['T_max'], df['RH_max'], df['Wind'])
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    return df

# ─── 6. SIDEBAR UI ───
st.sidebar.title("🌍 Location")
selected_city = st.sidebar.selectbox("Select City", list(CITIES.keys()))
lat, lon = CITIES[selected_city]['lat'], CITIES[selected_city]['lon']

st.sidebar.markdown("---")
st.sidebar.info("This AI predicts tropical heatwave momentum by running 7-day weather forecasts through a trained LSTM Neural Network.")

# ─── 7. MAIN DASHBOARD ───
st.title(f"🌡️ {selected_city} Weather & Heatwave Radar")

try:
    model, scaler, threshold_95 = load_models(selected_city)
    df_all = fetch_live_weather(lat, lon)
    
    # Identify "Today" in the dataframe (index 14, since past_days=14)
    today_date = df_all.index[WINDOW_SIZE]
    today_data = df_all.iloc[WINDOW_SIZE]
    
    # --- TOP ROW: WEATHER APP METRICS ---
    formatted_date = today_date.strftime("%A, %B %d, %Y")
    st.subheader(f"Current Conditions — {formatted_date}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Temperature", f"{today_data['T_max']:.1f} °C")
    col2.metric("Max Humidity", f"{today_data['RH_max']:.0f} %")
    col3.metric("Wind Speed", f"{today_data['Wind']:.1f} km/h")
    col4.metric("Feels Like (Heat Index)", f"{today_data['Heat_Index']:.1f} °C", 
                delta=f"{today_data['Heat_Index'] - threshold_95:.1f} °C to limit", delta_color="inverse")

    st.markdown("---")

    # --- MIDDLE ROW: 7-DAY AI FORECAST ---
    st.subheader("🔮 7-Day Heatwave Risk Forecast")
    
    # Prepare the 7 sliding windows
    future_dates = df_all.index[WINDOW_SIZE+1:] # Tomorrow up to Day 7
    sequences = []
    
    for i in range(1, 8):
        # Slice a 14-day window to predict day 'i'
        window_df = df_all.iloc[i : i + WINDOW_SIZE]
        scaled_window = scaler.transform(window_df[FEATURES])
        sequences.append(scaled_window)
        
    X_input = np.array(sequences) # Shape: (7, 14, 6)
    
    # Predict all 7 days at once!
    risk_scores = model.predict(X_input, verbose=0).flatten()
    
    # Create 7 columns for a beautiful weekly layout
    forecast_cols = st.columns(7)
    for i, col in enumerate(forecast_cols):
        day_name = future_dates[i].strftime("%a, %b %d")
        score = risk_scores[i]
        
        with col:
            st.markdown(f"**{day_name}**")
            # Color-coded risk numbers
            if score < 0.50:
                st.success(f"{score * 100:.1f}%")
            elif score < 0.75:
                st.warning(f"{score * 100:.1f}%")
            else:
                st.error(f"{score * 100:.1f}%")

    st.markdown("---")

    # --- BOTTOM ROW: THE DATA CHART ---
    st.subheader("📈 Heat Momentum vs. 95th Percentile Limit")
    
    fig = go.Figure()
    
    # Past 14 Days (Solid Line)
    fig.add_trace(go.Scatter(
        x=df_all.index[:WINDOW_SIZE+1], y=df_all['Heat_Index'].iloc[:WINDOW_SIZE+1], 
        mode='lines+markers', name='Past & Current Heat Index',
        line=dict(color='#888780', width=3)
    ))
    
    # Future 7 Days (Red Dashed Line)
    fig.add_trace(go.Scatter(
        x=df_all.index[WINDOW_SIZE:], y=df_all['Heat_Index'].iloc[WINDOW_SIZE:], 
        mode='lines+markers', name='Forecasted Heat Index',
        line=dict(color='#E24B4A', width=3, dash='dash')
    ))
    
    # The Danger Threshold
    fig.add_trace(go.Scatter(
        x=[df_all.index[0], df_all.index[-1]], 
        y=[threshold_95, threshold_95], 
        mode='lines', name=f'Danger Limit ({threshold_95:.1f}°C)',
        line=dict(color='gray', width=2, dash='dot')
    ))
    
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Apparent Temperature (°C)",
        hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Application Error: Ensure models are trained and saved in the `models/` directory. Error: {e}")