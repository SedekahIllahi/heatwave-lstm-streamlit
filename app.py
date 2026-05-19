import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import keras
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="SEA Heatwave Radar - Comparison Mode", page_icon=":material/thermostat:", layout="wide")

# --- CSS INJECTION ---
st.markdown("""
<style>
    /* Hide Streamlit branding and header/footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Reduce default massive top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Premium Metric Cards */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(150, 150, 150, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #FF9800 !important; /* Subtle orange accent on hover */
    }
    
    /* Style the risk percentage boxes to look like modern badges */
    [data-testid="stAlert"] {
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

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

# Top Container for Title
title_container = st.container()

# Region Selector
st.markdown("### :material/location_on: Select Target Region")
selected_city = st.radio("Region", list(CITIES.keys()), horizontal=True, label_visibility="collapsed")
lat, lon = CITIES[selected_city]['lat'], CITIES[selected_city]['lon']

st.markdown("---")

# Render Title at the very top using the selected city
with title_container:
    st.title(f":material/dashboard: SEA Heatwave Radar - {selected_city}", help="Dashboard comparing Deep Learning and Traditional ML models for predicting heatwave risks based on 14 days of historical weather patterns.")

try:
    lstm_model, rf_model, scaler, threshold_95 = load_models(selected_city)
    df_all = fetch_live_weather(lat, lon)
    
    today_date = df_all.index[WINDOW_SIZE]
    today_data = df_all.iloc[WINDOW_SIZE]
    
    # Current Conditions
    st.subheader(f"Current Baseline — {today_date.strftime('%A, %B %d, %Y')}", help="Current weather conditions fetched from the Open-Meteo Forecast API.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Temperature", f"{today_data['T_max']:.1f} °C", help="Raw maximum temperature from Open-Meteo.")
    col2.metric("Max Humidity", f"{today_data['RH_max']:.0f} %", help="Raw maximum relative humidity from Open-Meteo.")
    col3.metric("Wind Speed", f"{today_data['Wind']:.1f} km/h", help="Raw maximum wind speed from Open-Meteo.")
    col4.metric("Feels Like (AT)", f"{today_data['Heat_Index']:.1f} °C", help="Apparent Temperature calculated combining temperature, humidity, and wind speed.")

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

    # --- RENDER COMPARISON --- 
    st.subheader(":material/calendar_month: 7-Day Risk Comparison Outlook", help="Shows the predicted probability that the Heat Index will exceed the 95th percentile danger threshold. The models analyze the past 14 days to make this prediction.")
    
    st.markdown("### :material/memory: Deep Learning LSTM Outlook", help="Long Short-Term Memory Neural Network. A deep learning model that analyzes the chronological sequence of weather patterns over the past 14 days.")
    cols_lstm = st.columns(7)
    for i, c in enumerate(cols_lstm):
        c.markdown(f"<div style='text-align: center; font-weight: bold; margin-bottom: 5px;'>{future_dates[i].strftime('%a')}</div>", unsafe_allow_html=True)
        score = lstm_risks[i]
        if score < 0.5: c.success(f"{score*100:.0f}%")
        elif score < 0.75: c.warning(f"{score*100:.0f}%")
        else: c.error(f"{score*100:.0f}%")
        
    st.markdown("### :material/forest: Random Forest ML Outlook", help="An ensemble of 100 decision trees. It treats the past 14 days as 84 independent data points to find risk patterns without strict chronological order.")
    cols_rf = st.columns(7)
    for i, c in enumerate(cols_rf):
        c.markdown(f"<div style='text-align: center; font-weight: bold; margin-bottom: 5px;'>{future_dates[i].strftime('%a')}</div>", unsafe_allow_html=True)
        score = rf_risks[i]
        if score < 0.5: c.success(f"{score*100:.0f}%")
        elif score < 0.75: c.warning(f"{score*100:.0f}%")
        else: c.error(f"{score*100:.0f}%")

    st.markdown("---")
    
    # Interactive Chart
    st.subheader(":material/monitoring: Past and Forecasted Trend", help="Plots the calculated Heat Index for the past 14 days and the upcoming 7 days based on the raw Open-Meteo forecast, compared against the 95th percentile danger limit.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all.index[:WINDOW_SIZE+1], y=df_all['Heat_Index'].iloc[:WINDOW_SIZE+1], mode='lines+markers', name='Observed Heat Index', line=dict(color='gray', width=3)))
    fig.add_trace(go.Scatter(x=df_all.index[WINDOW_SIZE:], y=df_all['Heat_Index'].iloc[WINDOW_SIZE:], mode='lines+markers', name='Forecasted Heat Index', line=dict(color='#E24B4A', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=[df_all.index[0], df_all.index[-1]], y=[threshold_95, threshold_95], mode='lines', name='95th Pct Danger Limit', line=dict(color='#FF9800', width=2, dash='dot')))
    
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(":material/info: What is the 95th Percentile Danger Limit?"):
        st.markdown("""
        The **95th Percentile Danger Limit** is a localized threshold calculated specifically for each city based on 14 years of historical weather data. 
        
        Instead of using a generic temperature to define a heatwave, the system mathematically finds the Heat Index that was only exceeded **5% of the time** in that city's history. This means the orange dotted line represents the top 5% most extreme heat events for that specific region. If the forecast crosses this line, it is considered a severe and dangerous anomaly.
        """)

except Exception as e:
    st.error(f":material/warning: Application Error: {e}")