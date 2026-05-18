import os
import sys

# ─── 1. BEYOND THE MATRIX: FORCE PYTORCH BACKEND ───
# This must be set BEFORE importing Keras. It completely bypasses the Protobuf bug!
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KERAS_BACKEND"] = "torch"

import requests
import joblib
import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ─── 2. CONFIGURATION ───
CITIES = {
    'Jakarta':   {'lat': -6.2088, 'lon': 106.8456},
    'Bangkok':   {'lat': 13.7563, 'lon': 100.5018},
    'Manila':    {'lat': 14.5995, 'lon': 120.9842},
    'Singapore': {'lat': 1.3521,  'lon': 103.8198}
}

START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
WINDOW_SIZE = 14
FEATURES = ['T_max', 'RH_max', 'Wind', 'Heat_Index', 'doy_sin', 'doy_cos']

# ─── 3. METEOROLOGICAL & DATA UTILITIES ───
def calculate_apparent_temperature(t, rh, ws):
    """Australian AT formula (incorporates wind cooling factor)"""
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))
    return t + (0.33 * e) - (0.70 * (ws / 3.6)) - 4.00

def create_sequences(data, target, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(target[i + window])
    return np.array(X), np.array(y)

def build_directories():
    """Builds storage architecture cleanly inside root directory"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("📁 Target directory infrastructure verified.")

# ─── 4. MAIN EXECUTION PIPELINE ───
def run_pipeline():
    print(f"🎉 SUCCESS! Keras is executing natively on: {keras.backend.backend().upper()}")
    build_directories()
    
    for city, coords in CITIES.items():
        city_lbl = city.lower()
        print(f"\n{'='*60}\n🛰️ PROCESSING REGION: {city.upper()}\n{'='*60}")
        
        # --- A. Historical Data Ingestion ---
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
            f"&daily=temperature_2m_max,relative_humidity_2m_max,windspeed_10m_max&timezone=Asia%2FSingapore"
        )
        print(f" -> Pulling 15-year climate archive from Open-Meteo...")
        try:
            response = requests.get(url).json()
            df = pd.DataFrame(response['daily'])
        except KeyError:
            print(f"❌ API Error: Could not parse daily data for {city}. Skipping.")
            continue
            
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.columns = ['T_max', 'RH_max', 'Wind']
        df = df.ffill().bfill()
        
        # 💾 Data Storing: Raw Drop
        df.to_csv(f'data/raw/{city_lbl}_raw.csv')
        print(f" -> Stored Raw: data/raw/{city_lbl}_raw.csv")

        # --- B. Feature Engineering & Climate Alignment ---
        df['Heat_Index'] = calculate_apparent_temperature(df['T_max'], df['RH_max'], df['Wind'])
        df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        # Dynamic localized thresholding (95th Percentile)
        threshold_95 = df['Heat_Index'].quantile(0.95)
        df['Target'] = (df['Heat_Index'] >= threshold_95).astype(int)
        
        # 💾 Data Storing: Processed Feature Drop
        df.to_csv(f'data/processed/{city_lbl}_processed.csv')
        print(f" -> Stored Processed: data/processed/{city_lbl}_processed.csv")

        # --- C. Time Series Scaling & Windows ---
        train_df = df[df.index < '2022-01-01']
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[FEATURES])
        
        # Export math transformations for Streamlit runtime mapping
        joblib.dump(scaler, f'models/scaler_{city_lbl}.pkl')
        joblib.dump(threshold_95, f'models/threshold_{city_lbl}.pkl')

        X_train, y_train = create_sequences(train_scaled, train_df['Target'].values, WINDOW_SIZE)

        # --- D. Neural Network Compilation ---
        model = Sequential([
            Input(shape=(WINDOW_SIZE, len(FEATURES))),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='binary_crossentropy', 
            metrics=[keras.metrics.AUC(name='auc')]
        )
        
        # 15x weight compensation for regional anomaly bias
        early_stop = EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True)
        
        print(f" -> Brain training initialized (Target threshold: {threshold_95:.2f}°C)...")
        model.fit(
            X_train, y_train, validation_split=0.15, epochs=50, batch_size=32,
            class_weight={0: 1.0, 1: 15.0}, callbacks=[early_stop], verbose=1
        )
        
        model.save(f'models/{city_lbl}_lstm.keras')
        print(f" -> Verification Success: Baked weights to models/{city_lbl}_lstm.keras")

    print("\n🎉 ALL ARCHITECTURES CONVERGED! Backend training complete.")

if __name__ == '__main__':
    run_pipeline()