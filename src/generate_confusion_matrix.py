import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Force PyTorch backend for Keras consistency
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KERAS_BACKEND"] = "torch"
import keras

CITIES = ['Jakarta', 'Bangkok', 'Manila', 'Kuala Lumpur', 'Singapore', 'Yogyakarta']
WINDOW_SIZE = 14
FEATURES = ['T_max', 'RH_max', 'Wind', 'Heat_Index', 'doy_sin', 'doy_cos']

def create_sequences(data, target, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(target[i + window])
    return np.array(X), np.array(y)

# Create an output folder for your report assets
os.makedirs('report_outputs', exist_ok=True)

# Master list to save tabular data for your report tables
report_rows = []

print("📊 GENERATING HISTORICAL TEST EVALUATIONS (70-15-15 CHRONOLOGICAL SPLIT)...")

for city in CITIES:
    city_lbl = city.lower()
    
    # 1. Load the processed dataset
    processed_path = f'data/processed/{city_lbl}_processed.csv'
    if not os.path.exists(processed_path):
        print(f"⚠️ Processed data for {city} missing. Please run run_pipeline.py first.")
        continue
        
    df = pd.read_csv(processed_path, index_col='time', parse_dates=['time'])
    
    # 2. CHRONOLOGICAL SPLIT (70% Train, 15% Val, 15% Test)
    total_days = len(df)
    train_end_idx = int(total_days * 0.70)
    val_end_idx = int(total_days * 0.85)
    
    # We grab the Test partition (the final 15% of the timeline)
    test_df = df.iloc[val_end_idx:]
    
    # 3. Load math artifacts and scale data
    scaler = joblib.load(f'models/scaler_{city_lbl}.pkl')
    test_scaled = scaler.transform(test_df[FEATURES])
    
    # Create the matrices
    X_test_lstm, y_true = create_sequences(test_scaled, test_df['Target'].values, WINDOW_SIZE)
    X_test_rf = X_test_lstm.reshape(X_test_lstm.shape[0], -1)
    
    # 4. Load Models & Predict
    lstm_model = keras.models.load_model(f'models/{city_lbl}_lstm.keras')
    rf_model = joblib.load(f'models/{city_lbl}_rf.pkl')
    
    # Generate binary predictions (Threshold at 0.5)
    lstm_preds = (lstm_model.predict(X_test_lstm, verbose=0).flatten() >= 0.5).astype(int)
    rf_preds = rf_model.predict(X_test_rf)
    
    # 5. Generate Confusion Matrices
    cm_lstm = confusion_matrix(y_true, lstm_preds)
    cm_rf = confusion_matrix(y_true, rf_preds)
    
    # Extract structural metrics for table
    # Standard output ordering from confusion_matrix: tn, fp, fn, tp
    tn_l, fp_l, fn_l, tp_l = cm_lstm.ravel() if cm_lstm.size == 4 else (cm_lstm[0][0], 0, 0, 0)
    tn_r, fp_r, fn_r, tp_r = cm_rf.ravel() if cm_rf.size == 4 else (cm_rf[0][0], 0, 0, 0)
    
    report_lstm = classification_report(y_true, lstm_preds, output_dict=True, zero_division=0)
    report_rf = classification_report(y_true, rf_preds, output_dict=True, zero_division=0)
    
    # Save statistics for CSV export
    report_rows.append({
        'City': city, 'Model': 'LSTM', 
        'TN': tn_l, 'FP': fp_l, 'FN': fn_l, 'TP': tp_l,
        'Precision_Heatwave': report_lstm['1']['precision'], 
        'Recall_Heatwave': report_lstm['1']['recall'], 
        'F1_Score_Heatwave': report_lstm['1']['f1-score']
    })
    report_rows.append({
        'City': city, 'Model': 'Random Forest', 
        'TN': tn_r, 'FP': fp_r, 'FN': fn_r, 'TP': tp_r,
        'Precision_Heatwave': report_rf['1']['precision'], 
        'Recall_Heatwave': report_rf['1']['recall'], 
        'F1_Score_Heatwave': report_rf['1']['f1-score']
    })
    
    # 6. PLOT AND SAVE THE GRAPHICAL IMAGES FOR BAB 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Pred Normal', 'Pred Heatwave'], yticklabels=['Actual Normal', 'Actual Heatwave'])
    axes[0].set_title(f'{city} - LSTM Confusion Matrix')
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Pred Normal', 'Pred Heatwave'], yticklabels=['Actual Normal', 'Actual Heatwave'])
    axes[1].set_title(f'{city} - Random Forest Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'report_outputs/{city_lbl}_confusion_matrix.png', dpi=300)
    plt.close()
    
    print(f" ✅ Processed {city} metrics & exported report_outputs/{city_lbl}_confusion_matrix.png")

# Export complete table to data frame and save as CSV
summary_df = pd.DataFrame(report_rows)
summary_df.to_csv('report_outputs/model_comparison_metrics.csv', index=False)
print("\n🎉 SUCCESS! Full dataset table saved to: report_outputs/model_comparison_metrics.csv")