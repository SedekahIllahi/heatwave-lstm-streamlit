import os
import sys

# Force pure-python protobuf parsing and torch backend for Keras
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ─── CONFIGURATION ───
# ─── REGIONAL CONFIGURATION (EXPANDED SCOPE) ───
CITIES = {
    'Jakarta':       {'lat': -6.2088, 'lon': 106.8456},
    'Bangkok':       {'lat': 13.7563, 'lon': 100.5018},
    'Manila':        {'lat': 14.5995, 'lon': 120.9842},
    'Kuala Lumpur':  {'lat': 3.1390,  'lon': 101.6869},
    'Singapore':     {'lat': 1.3521,  'lon': 103.8198},
    'Yogyakarta':    {'lat': -7.7956, 'lon': 110.3695}
}

START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
WINDOW_SIZE = 14
FEATURES = ['T_max', 'RH_max', 'Wind', 'Heat_Index', 'doy_sin', 'doy_cos']

def calculate_apparent_temperature(t, rh, ws):
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))
    return t + (0.33 * e) - (0.70 * (ws / 3.6)) - 4.00

def create_sequences(data, target, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(target[i + window])
    return np.array(X), np.array(y)

def build_directories():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def run_pipeline():
    build_directories()
    print(f"Keras running on: {keras.backend.backend().upper()} | Comparison Mode: Enabled")
    
    for city, coords in CITIES.items():
        city_lbl = city.lower()
        print(f"\n{'='*60}\nPROCESSING REGION: {city.upper()}\n{'='*60}")
        
        # --- A. Ingestion & Storage ---
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
            f"&daily=temperature_2m_max,relative_humidity_2m_max,windspeed_10m_max&timezone=Asia%2FSingapore"
        )
        response = requests.get(url).json()
        df = pd.DataFrame(response['daily'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.columns = ['T_max', 'RH_max', 'Wind']
        df = df.ffill().bfill()
        
        df.to_csv(f'data/raw/{city_lbl}_raw.csv')

        # --- B. Math & Feature Engineering ---
        df['Heat_Index'] = calculate_apparent_temperature(df['T_max'], df['RH_max'], df['Wind'])
        df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        threshold_95 = df['Heat_Index'].quantile(0.95)
        df['Target'] = (df['Heat_Index'] >= threshold_95).astype(int)
        df.to_csv(f'data/processed/{city_lbl}_processed.csv')

        # --- C. Scaling & Train Split ---
        train_df = df[df.index < '2022-01-01']
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[FEATURES])
        
        joblib.dump(scaler, f'models/scaler_{city_lbl}.pkl')
        joblib.dump(threshold_95, f'models/threshold_{city_lbl}.pkl')

        # Generate standard 3D sequences for LSTM
        X_train_lstm, y_train_lstm = create_sequences(train_scaled, train_df['Target'].values, WINDOW_SIZE)
        
        # Flatten the 14-day history window into a 2D row for Random Forest (14 days * 6 features = 84 features)
        X_train_rf = X_train_lstm.reshape(X_train_lstm.shape[0], -1)

        # --- D. Model 1: Train LSTM ---
        print(f" -> Training Model 1: LSTM Neural Network...")
        lstm_model = Sequential([
            Input(shape=(WINDOW_SIZE, len(FEATURES))),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
        early_stop = EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True)
        
        lstm_model.fit(
            X_train_lstm, y_train_lstm, validation_split=0.15, epochs=50, batch_size=32,
            class_weight={0: 1.0, 1: 15.0}, callbacks=[early_stop], verbose=0
        )
        lstm_model.save(f'models/{city_lbl}_lstm.keras')

        # --- E. Model 2: Train Random Forest ---
        print(f" -> Training Model 2: Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight={0: 1.0, 1: 15.0}, random_state=42)
        rf_model.fit(X_train_rf, y_train_lstm)
        joblib.dump(rf_model, f'models/{city_lbl}_rf.pkl')
        
        print(f"Stored both models for {city}!")

    print("\nALL MODELS TRAINED SUCCESSFULLY.")

if __name__ == '__main__':
    run_pipeline()