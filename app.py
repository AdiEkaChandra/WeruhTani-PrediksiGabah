from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__, template_folder=r'D:\PROYEK\PROYEK1\htdocs\latihan-ngoding')
CORS(app) 

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('indexnew.html')

# Route untuk menangani prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Membaca dataset
    dataset_path = r'D:\PROYEK\PROYEK1\htdocs\PrediksiWaktuKeringGabah\data.csv'
    dataset = pd.read_csv(dataset_path)
    
    # Menghapus baris yang memiliki nilai yang hilang
    dataset.dropna(inplace=True)

    # Mengubah tipe data kolom 'massa' menjadi integer
    dataset['massa'] = dataset['massa'].astype(int)

    # Daftar fitur yang digunakan
    features = ['massa', 'suhu_awal', 'kadar_air_awal', 'suhu_akhir', 'kadar_air_akhir']
    target = 'waktu'

    # Membagi dataset menjadi data latih dan data uji
    X = dataset[features]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat model regresi linear dan melatihnya
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Mengambil input dari form HTML
    data = request.json
    massa = int(data['massa'])
    suhu_awal = float(data['suhu_awal'])
    kadar_air_awal = float(data['kadar_air_awal'])
    suhu_akhir = float(data['suhu_akhir'])
    kadar_air_akhir = float(data['kadar_air_akhir'])

    # Membuat DataFrame untuk data baru
    new_data = pd.DataFrame({
        'massa': [massa],
        'suhu_awal': [suhu_awal],
        'kadar_air_awal': [kadar_air_awal],
        'suhu_akhir': [suhu_akhir],
        'kadar_air_akhir': [kadar_air_akhir]
    })

    # Melakukan prediksi
    predicted_time_seconds = model.predict(new_data)[0]

    # Mengubah waktu menjadi format jam, menit, dan detik
    hours = int(predicted_time_seconds // 3600)
    minutes = int((predicted_time_seconds % 3600) // 60)
    seconds = int(predicted_time_seconds % 60)

    result = {
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
