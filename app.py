from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
from collections import deque

app = Flask(__name__)

# --- 1. LOAD ASSETS & DATA ---
model = joblib.load('best_forecasting_model.pkl')
model_features = joblib.load('model_features.pkl')

# Load data cuaca mentah untuk input aplikasi
df_weather_test = pd.read_csv('weather_features_test.csv')

# --- 2. MEMORY BUFFER (BUKU CATATAN) ---
history_load = deque(maxlen=200) 
current_row_idx = 0

def load_initial_memory():
    """Mengisi memori awal dari file historis agar model punya data 'masa lalu' saat start"""
    try:
        # Kita ambil data beban asli dari dataset training/validation terakhir
        # untuk pancingan lag 168 jam (1 minggu)
        hist_df = pd.read_csv('historical_energy_log.csv') 
        last_loads = hist_df['total load actual'].tail(170).tolist()
        for val in last_loads:
            history_load.append(val)
        print("✅ Memori berhasil dipulihkan dari data historis.")
    except Exception as e:
        print(f"⚠️ Gagal pancing data: {e}. Menggunakan nilai rata-rata.")
        for _ in range(170): history_load.append(25000.0)

# Panggil fungsi pancingan saat server pertama kali jalan
load_initial_memory()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_prediction')
def get_prediction():
    global current_row_idx
    
    if current_row_idx >= len(df_weather_test):
        current_row_idx = 0
    
    # Ambil data cuaca baris saat ini
    weather_now = df_weather_test.iloc[current_row_idx]
    dt = pd.to_datetime(weather_now['time'])
    
    # 1. BUAT SEMUA FITUR (Termasuk fitur yang menyebabkan error)
    features = {
        # Fitur Waktu Dasar
        'hour': dt.hour,
        'dayofweek': dt.dayofweek,
        'month': dt.month,
        'is_weekend': 1 if dt.dayofweek >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        
        # Fitur Lag Beban (Diambil dari buffer history_load)
        'load_lag_6h': history_load[-7],
        'load_lag_12h': history_load[-13],
        'load_lag_24h': history_load[-25],
        'load_lag_168h': history_load[0],
        'load_rolling_mean_6h': np.mean(list(history_load)[-12:-6]),
        
        # Fitur Cuaca Utama
        'temp_lag_6h': weather_now['temp'],
        'temp_squared': weather_now['temp'] ** 2,
        'humidity_lag_6h': weather_now['humidity'],
        'wind_speed_lag_6h': weather_now['wind_speed'],
        'pressure_lag_6h': weather_now['pressure'],
        
        # --- FIX ERROR: Fitur Cuaca yang Hilang ---
        # Kita gunakan .get() agar jika kolom tidak ada di CSV, dia otomatis isi 0
        'wind_deg': weather_now.get('wind_deg', 0),
        'rain_1h': weather_now.get('rain_1h', 0),
        'clouds_all': weather_now.get('clouds_all', 0)
    }

    # 2. HANDLE WEATHER MAIN (One-Hot Encoding)
    for col in model_features:
        if 'weather_main_' in col:
            # Contoh: weather_main_clouds
            current_weather = str(weather_now['weather_main']).lower()
            features[col] = 1 if f"weather_main_{current_weather}" == col.lower() else 0

    # 3. KONVERSI KE DATAFRAME
    X_input = pd.DataFrame([features])

    # 4. PROTEKSI TERAKHIR: Isi kolom yang masih belum ada dengan 0
    for col in model_features:
        if col not in X_input.columns:
            X_input[col] = 0

    # 5. SAMAKAN URUTAN (WAJIB agar prediksi akurat)
    X_input = X_input[model_features]
    
    # 6. PREDIKSI
    prediction = model.predict(X_input)[0]
    
    # Simpan hasil prediksi ke memori untuk lag masa depan
    history_load.append(prediction)
    current_row_idx += 1
    
    return jsonify({
        'time': str(weather_now['time']),
        'prediction': float(prediction),
        'temp': float(weather_now['temp']),
        'weather': str(weather_now['weather_main'])
    })

if __name__ == '__main__':
    app.run(debug=True)