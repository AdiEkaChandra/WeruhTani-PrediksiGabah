import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import threading

# Membaca dataset dari file CSV
dataset = pd.read_csv('data.csv')

# Menghapus baris yang memiliki nilai yang hilang
dataset.dropna(inplace=True)

# Mengubah tipe data kolom 'massa' menjadi integer
dataset['massa'] = dataset['massa'].astype(int)

# Daftar fitur yang digunakan
features = ['massa', 'suhu_awal', 'kadar_air_awal', 'suhu_akhir', 'kadar_air_akhir']

# Membaca inputan massa
massa = int(input("Masukkan nilai massa: "))

# Variabel untuk menyimpan status notifikasi
notification_sent = False

def check_notification(kadar_air_akhir):
    global notification_sent

    if not notification_sent and (kadar_air_akhir == 14 or kadar_air_akhir == 16):
        time_left = 5  # Waktu penghentian dalam menit
        print("Peringatan: Kadar air akhir akan mencapai angka 14 atau 16 dalam {} menit.".format(time_left))

        # Menandakan notifikasi telah dikirim
        notification_sent = True

# List untuk menyimpan data fitur sebelum kadar air mencapai 14
fitur_sebelum_14 = []

while True:
    # Membaca data sensor secara real-time
    suhu_awal = float(input("Masukkan nilai suhu_awal: "))
    kadar_air_awal = float(input("Masukkan nilai kadar_air_awal: "))
    suhu_akhir = float(input("Masukkan nilai suhu_akhir: "))
    kadar_air_akhir = float(input("Masukkan nilai kadar_air_akhir: "))

    # Mengecek kadar air akhir
    if kadar_air_akhir < 14:
        print("Peringatan: Kadar air akhir kurang dari 14. Siapkan untuk menghentikan program.")
        fitur_sebelum_14.append([massa, suhu_awal, kadar_air_awal, suhu_akhir, kadar_air_akhir])
    elif kadar_air_akhir <= 16:
        print("Kadar air akhir sudah mencapai angka 14-16. Menghentikan program.")
        break

    check_notification(kadar_air_akhir)  # Memeriksa pemberitahuan

    # Membuat DataFrame untuk data baru
    new_data = pd.DataFrame({
        'massa': [massa],
        'suhu_awal': [suhu_awal],
        'kadar_air_awal': [kadar_air_awal],
        'suhu_akhir': [suhu_akhir],
        'kadar_air_akhir': [kadar_air_akhir]
    })

    # Memisahkan fitur dan target dari dataset
    X = dataset[features]
    y = dataset['waktu']

    # Membagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat objek model Linear Regression
    model = LinearRegression()

    # Melatih model dengan data latih
    model.fit(X_train, y_train)

    # Memprediksi waktu berdasarkan data baru
    predicted_time_seconds = model.predict(new_data[features])

    # Mengubah waktu menjadi format jam, menit, detik
    hours = int(predicted_time_seconds // 3600)
    minutes = int((predicted_time_seconds % 3600) // 60)
    seconds = int(predicted_time_seconds % 60)

    # Menampilkan hasil prediksi
    print("Predicted Time: {} jam, {} menit, {} detik".format(hours, minutes, seconds))

    time.sleep(5)

# Menampilkan data fitur sebelum kadar air mencapai 14
print("Data fitur sebelum kadar air mencapai 14:")
for data in fitur_sebelum_14:
    print("Massa: {}, Suhu Awal: {}, Kadar Air Awal: {}, Suhu Akhir: {}, Kadar Air Akhir: {}".format(data[0], data[1], data[2], data[3], data[4]))
