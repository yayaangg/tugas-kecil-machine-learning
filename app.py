from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# --- 1. LOAD ASSETS ---
# Pastikan model ini adalah hasil training ulang tanpa StandardScaler
model = joblib.load('best_forecasting_model.pkl')
model_features = joblib.load('model_features.pkl')

# Pakai dataset baru yang sudah ada kolom 'total load actual'
df_test = pd.read_csv('weather_features_2019_with_actual.csv')
current_row_idx = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_prediction')
def get_prediction():
    global current_row_idx
    
    # Reset jika data habis
    if current_row_idx >= len(df_test):
        current_row_idx = 0
    
    row = df_test.iloc[current_row_idx]
    dt = pd.to_datetime(row['time'])
    
    # --- 2. FEATURE ENGINEERING (DIRECT MODE) ---
    # Kita ambil fitur lag langsung dari dataset (karena sekarang sudah ada datanya)
    # Jika dataset 2019 kamu belum punya kolom lag, kita bisa buat dummy lag dari total_load_actual
    
    actual_now = float(row['total load actual'])
    
    features = {
        'hour': dt.hour,
        'dayofweek': dt.dayofweek,
        'month': dt.month,
        'is_weekend': 1 if dt.dayofweek >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        
        # Fitur Lag (Direct: Mengambil data aktual saat ini sebagai referensi)
        # Untuk simulasi, kita asumsikan lag ini mendekati nilai aktual jam tersebut
        'load_lag_6h': actual_now * 0.98,  # Simulasi beban 6 jam lalu
        'load_lag_12h': actual_now * 1.02, 
        'load_lag_24h': actual_now * 0.95,
        'load_lag_168h': actual_now,
        'load_rolling_mean_6h': actual_now,
        
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

    # --- 3. HANDLING ONE-HOT & MISSING COLUMNS ---
    X_input = pd.DataFrame([features])
    
    for col in model_features:
        if col not in X_input.columns:
            if 'weather_main_' in col:
                current_w = str(row['weather_main']).lower()
                X_input[col] = 1 if f"weather_main_{current_w}" == col.lower() else 0
            else:
                # Kolom time_segment akan otomatis diisi 0 karena sudah kita hapus logikanya
                X_input[col] = 0

    # Pastikan urutan kolom sesuai model_features.pkl
    X_input = X_input[model_features]
    
    # --- 4. PREDICTION ---
    # Gunakan .values untuk menghindari warning feature names jika perlu
    prediction = model.predict(X_input)[0]
    
    current_row_idx += 1
    
    # Kirim data ke Frontend
    return jsonify({
        'time': str(row['time']),
        'prediction': float(prediction),
        'actual': actual_now, # Tambahkan data asli untuk dibandingkan di grafik
        'temp': float(row['temp']),
        'weather': str(row['weather_main'])
    })

if __name__ == '__main__':
    app.run(debug=True)