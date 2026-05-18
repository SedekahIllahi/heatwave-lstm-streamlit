import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import keras
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="SEA Heatwave Radar - Comparison Mode", page_icon="🥵", layout="wide")

CITIES = {
    'Jakarta':       {'lat': -6.2088, 'lon': 106.8456},
    'Bangkok':       {'lat': 13.7563, 'lon': 100.5018},
    'Manila':        {'lat': 14.5995, 'lon': 120.9842},
    'Kuala Lumpur':  {'lat': 3.1390,  'lon': 101.6869},
    'Singapore':     {'lat': 1.3521,  'lon': 103.8198},
    'Yogyakarta':    {'lat': -7.7956, 'lon': 110.3695}
}

WINDOW_SIZE = 14
FEATURES = ['T_max', 'RH_max', 'Wind', 'Heat_Index', 'doy_sin', 'doy_cos']

def calculate_apparent_temperature(t, rh, ws):
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))
    return t + (0.33 * e) - (0.70 * (ws / 3.6)) - 4.00

@st.cache_resource
def load_models(city):
    city_lbl = city.lower()
    lstm = keras.models.load_model(f'models/{city_lbl}_lstm.keras')
    rf = joblib.load(f'models/{city_lbl}_rf.pkl')
    scaler = joblib.load(f'models/scaler_{city_lbl}.pkl')
    threshold = joblib.load(f'models/threshold_{city_lbl}.pkl')
    return lstm, rf, scaler, threshold

@st.cache_data(ttl=3600) 
def fetch_live_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&past_days={WINDOW_SIZE}&forecast_days=8"
        f"&daily=temperature_2m_max,relative_humidity_2m_max,windspeed_10m_max&timezone=Asia%2FSingapore"
    )
    response = requests.get(url).json()
    df = pd.DataFrame(response['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.columns = ['T_max', 'RH_max', 'Wind']
    
    df['Heat_Index'] = calculate_apparent_temperature(df['T_max'], df['RH_max'], df['Wind'])
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    return df

# Sidebar
st.sidebar.title("🌍 Architecture Comparison")
selected_city = st.sidebar.selectbox("Select Target Region", list(CITIES.keys()))
lat, lon = CITIES[selected_city]['lat'], CITIES[selected_city]['lon']

st.title(f"🌡️ {selected_city} Model Comparison Dashboard")

try:
    lstm_model, rf_model, scaler, threshold_95 = load_models(selected_city)
    df_all = fetch_live_weather(lat, lon)
    
    today_date = df_all.index[WINDOW_SIZE]
    today_data = df_all.iloc[WINDOW_SIZE]
    
    # Current Conditions
    st.subheader(f"Current Baseline — {today_date.strftime('%A, %B %d, %Y')}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Temperature", f"{today_data['T_max']:.1f} °C")
    col2.metric("Max Humidity", f"{today_data['RH_max']:.0f} %")
    col3.metric("Wind Speed", f"{today_data['Wind']:.1f} km/h")
    col4.metric("Feels Like (AT)", f"{today_data['Heat_Index']:.1f} °C")

    st.markdown("---")

    # --- FORECAST GENERATION FOR BOTH MODELS ---
    future_dates = df_all.index[WINDOW_SIZE+1:]
    lstm_sequences = []
    
    for i in range(1, 8):
        window_df = df_all.iloc[i : i + WINDOW_SIZE]
        scaled_window = scaler.transform(window_df[FEATURES])
        lstm_sequences.append(scaled_window)
        
    X_lstm = np.array(lstm_sequences)
    X_rf = X_lstm.reshape(X_lstm.shape[0], -1) # Flatten for RF
    
    # Get probabilities from both
    lstm_risks = lstm_model.predict(X_lstm, verbose=0).flatten()
    rf_risks = rf_model.predict_proba(X_rf)[:, 1] # Extract 'True' class probability

    # --- RENDER COMPARISON SIDE-BY-SIDE ---
    st.subheader("🔮 7-Day Risk Comparison Outlook")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("### 🧠 Deep Learning LSTM Outlook")
        cols = st.columns(7)
        for i, c in enumerate(cols):
            c.markdown(f"**{future_dates[i].strftime('%a')}**")
            score = lstm_risks[i]
            if score < 0.5: c.success(f"{score*100:.0f}%")
            elif score < 0.75: c.warning(f"{score*100:.0f}%")
            else: c.error(f"{score*100:.0f}%")
            
    with comp_col2:
        st.markdown("### 🌲 Random Forest ML Outlook")
        cols = st.columns(7)
        for i, c in enumerate(cols):
            c.markdown(f"**{future_dates[i].strftime('%a')}**")
            score = rf_risks[i]
            if score < 0.5: c.success(f"{score*100:.0f}%")
            elif score < 0.75: c.warning(f"{score*100:.0f}%")
            else: c.error(f"{score*100:.0f}%")

    st.markdown("---")
    
    # Interactive Chart
    st.subheader("📈 Past and Forecasted Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all.index[:WINDOW_SIZE+1], y=df_all['Heat_Index'].iloc[:WINDOW_SIZE+1], mode='lines+markers', name='Observed Heat Index', line=dict(color='gray', width=3)))
    fig.add_trace(go.Scatter(x=df_all.index[WINDOW_SIZE:], y=df_all['Heat_Index'].iloc[WINDOW_SIZE:], mode='lines+markers', name='Forecasted Heat Index', line=dict(color='#E24B4A', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=[df_all.index[0], df_all.index[-1]], y=[threshold_95, threshold_95], mode='lines', name='95th Pct Danger Limit', line=dict(color='black', width=1, dash='dot')))
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Application Error: {e}")