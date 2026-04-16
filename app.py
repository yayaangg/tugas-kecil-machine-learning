from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# --- 1. LOAD MODEL, FEATURES, & REFERENCE DATA ---
model = joblib.load('best_forecasting_model_v2_adapted.pkl')
features_config = joblib.load('model_features.pkl')

# Membaca data acuan 2026
df_ref = pd.read_csv('energy_dataset_test_2026.csv')
df_ref['time'] = pd.to_datetime(df_ref['time'])

# Hitung median per jam dari seluruh dataset sebagai "Pola Umum" (Fallback)
df_ref['hour_ref'] = df_ref['time'].dt.hour
hourly_median = df_ref.groupby('hour_ref')['total load actual'].median().to_dict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        waktu_str = request.form['waktu']
        suhu = float(request.form['suhu'])
        kelembaban = float(request.form['kelembaban'])
        angin = float(request.form['angin'])
        tekanan = float(request.form['tekanan'])
        
        waktu_dt = datetime.strptime(waktu_str, '%Y-%m-%dT%H:%M')
        
        labels = []
        values = []
        
        for i in range(6):
            future_time = waktu_dt + timedelta(hours=i)
            target_hour = future_time.hour
            
            # --- LOGIKA PENCARIAN DATA ACUAN ---
            # 1. Coba cari data yang benar-benar pas dengan tanggal & jam di CSV
            mask = df_ref['time'] == future_time.strftime('%Y-%m-%d %H:00:00')
            row_match = df_ref[mask]
            
            if not row_match.empty:
                # Jika ditemukan (misal prediksi masih di bawah jam 6 pagi tanggal 16 April)
                ref_load = row_match['total load actual'].values[0]
            else:
                # Jika TIDAK ditemukan (misal prediksi jam malam atau hari esoknya)
                # Ambil median historis untuk jam tersebut (Pola Perilaku Umum)
                ref_load = hourly_median.get(target_hour, 33000.0)
            
            # 2. Susun Dictionary Input
            input_dict = {
                'hour': target_hour,
                'dayofweek': future_time.weekday(),
                'month': future_time.month,
                'is_weekend': 1 if future_time.weekday() >= 5 else 0,
                'hour_sin': np.sin(2 * np.pi * target_hour / 24),
                'hour_cos': np.cos(2 * np.pi * target_hour / 24),
                'temp_lag_6h': suhu, 
                'temp_squared': suhu ** 2,
                'humidity_lag_6h': kelembaban,
                'wind_speed_lag_6h': angin,
                'pressure_lag_6h': tekanan,
                # Fitur lag mengikuti ref_load yang sudah divalidasi di atas
                'load_lag_6h': ref_load * 0.98,
                'load_lag_12h': ref_load * 0.95,
                'load_lag_24h': ref_load, 
                'load_lag_168h': ref_load * 1.02,
                'load_rolling_mean_6h': ref_load * 0.97,
                'wind_deg': 150.0, 'rain_1h': 0.0, 'clouds_all': 40.0
            }

            # Tambahkan kategori cuaca (One-Hot Encoding default 0)
            weather_cats = ['clouds', 'drizzle', 'fog', 'haze', 'mist', 'rain', 'thunderstorm']
            for cat in weather_cats:
                input_dict[f'weather_main_{cat}'] = 0

            # 3. Konversi ke DataFrame & Prediksi
            df_temp = pd.DataFrame([input_dict])
            df_temp = df_temp.reindex(columns=features_config, fill_value=0)
            pred_hour = model.predict(df_temp)[0]
            
            labels.append(future_time.strftime('%H:%M'))
            values.append(float(pred_hour))

        return render_template('index.html', 
                               prediction_text=f"{values[0]:,.2f} MW",
                               labels=labels, 
                               values=values)

if __name__ == '__main__':
    app.run(debug=True)