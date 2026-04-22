from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
from collections import deque

app = Flask(__name__)

# --- 1. LOAD ASSETS ---
model = joblib.load('best_forecasting_model.pkl')
model_features = joblib.load('model_features.pkl')

# Pakai file 2019 yang hanya berisi cuaca
df_test = pd.read_csv('weather_features_2019.csv') 

# --- 2. INITIAL MEMORY (DUMMY AKHIR 2018) ---
history_load = deque(maxlen=200)

def initialize_memory():
    # Data 6 jam terakhir 2018 sesuai yang Inces berikan
    data_akhir_2018 = [29690.0, 30619.0, 29932.0, 27903.0, 25450.0, 24424.0]
    
    # Isi buffer awal dengan rata-rata agar fitur lag_168h tersedia
    for _ in range(162): 
        history_load.append(28000.0) 
        
    for val in data_akhir_2018:
        history_load.append(val)
    print("✅ Memori siap. Memulai prediksi 2019 tanpa segmentasi waktu.")

initialize_memory()
current_row_idx = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_prediction')
def get_prediction():
    global current_row_idx
    
    if current_row_idx >= len(df_test):
        current_row_idx = 0
    
    row = df_test.iloc[current_row_idx]
    dt = pd.to_datetime(row['time'])
    
    # --- 3. PROCESSING FITUR ---
    features = {
        # Fitur Waktu Dasar
        'hour': dt.hour,
        'dayofweek': dt.dayofweek,
        'month': dt.month,
        'is_weekend': 1 if dt.dayofweek >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        
        # Fitur Lag (Dari Memory)
        'load_lag_6h': history_load[-7],
        'load_lag_12h': history_load[-13],
        'load_lag_24h': history_load[-25],
        'load_lag_168h': history_load[0],
        'load_rolling_mean_6h': np.mean(list(history_load)[-12:-6]),
        
        # Fitur Cuaca
        'temp_lag_6h': row['temp'],
        'temp_squared': row['temp'] ** 2,
        'humidity_lag_6h': row['humidity'],
        'wind_speed_lag_6h': row['wind_speed'],
        'pressure_lag_6h': row['pressure'],
        'wind_deg': row.get('wind_deg', 0),
        'rain_1h': row.get('rain_1h', 0),
        'clouds_all': row.get('clouds_all', 0)
    }

    # --- 4. PROTEKSI KOLOM (FORMALITAS UNTUK MODEL) ---
    # Kita buat dataframe dari fitur di atas
    X_input = pd.DataFrame([features])

    # Pastikan semua kolom yang diminta model (termasuk segmentasi) ada di X_input
    # Jika tidak ada di 'features', otomatis diisi 0 (Netral)
    for col in model_features:
        if col not in X_input.columns:
            # Jika itu kolom weather_main, cek kesesuaian
            if 'weather_main_' in col:
                current_w = str(row['weather_main']).lower()
                X_input[col] = 1 if f"weather_main_{current_w}" == col.lower() else 0
            else:
                # Kolom segmentasi waktu (Malam/Pagi/Siang) akan masuk ke sini dan diisi 0
                X_input[col] = 0

    # Samakan urutan kolom sesuai model_features.pkl
    X_input = X_input[model_features]
    
    # Prediksi
    prediction = model.predict(X_input.values)[0]    
    # Simpan hasil tebakan ke memory untuk jadi lag jam berikutnya
    history_load.append(prediction)
    current_row_idx += 1
    
    return jsonify({
        'time': str(row['time']),
        'prediction': float(prediction),
        'temp': float(row['temp']),
        'weather': str(row['weather_main'])
    })

if __name__ == '__main__':
    app.run(debug=True)