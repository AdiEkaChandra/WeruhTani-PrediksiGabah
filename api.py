import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import mysql.connector
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'proyek'

mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

def load_model_and_scaler():
    model_path = 'model.pkl'
    scaler_path = 'scaler.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        return model, scaler
    else:
        raise FileNotFoundError("Model and/or scaler pickle files not found.")

def create_trigger():
    cursor = mysql.cursor()

    # Buat trigger untuk operasi INSERT
    query_insert = """
    CREATE TRIGGER gabah_insert_trigger BEFORE INSERT ON gabah
    FOR EACH ROW
    BEGIN
        SET NEW.created_at = NOW();
        SET NEW.updated_at = NOW();
    END
    """
    cursor.execute(query_insert)

    # Buat trigger untuk operasi UPDATE
    query_update = """
    CREATE TRIGGER gabah_update_trigger BEFORE UPDATE ON gabah
    FOR EACH ROW
    BEGIN
        SET NEW.updated_at = NOW();
    END
    """
    cursor.execute(query_update)

    mysql.commit()

@app.route('/training')
def train():
    cursor = mysql.cursor()
    cursor.execute('SELECT * FROM dataset')
    data = cursor.fetchall()

    dataset = pd.DataFrame(data, columns=['massa', 'suhu_awal', 'kadar_air_awal', 'suhu_akhir', 'kadar_air_akhir', 'waktu', 'column2'])
    features = ['massa', 'suhu_awal', 'kadar_air_awal', 'suhu_akhir', 'kadar_air_akhir']

    # Preprocessing data
    dataset = dataset[pd.to_numeric(dataset['massa'], errors='coerce').notnull()]
    dataset['massa'] = dataset['massa'].astype('float')

    X = dataset[features]
    y = dataset['waktu']

    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Training Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    print("Training MAE:", mae_train)
    print("Testing MAE:", mae_test)
    print("Training MSE:", mse_train)
    print("Testing MSE:", mse_test)
    print("Training R-squared:", r2_train)
    print("Testing R-squared:", r2_test)

    new_data = pd.DataFrame({
        'massa': [2000.0],
        'suhu_awal': [30.0],
        'kadar_air_awal': [12.0],
        'suhu_akhir': [30.0],
        'kadar_air_akhir': [16.0]
    })

    new_data_scaled = scaler.transform(new_data)

    predicted_time = model.predict(new_data_scaled)[0] * 3600
    hours = int(predicted_time // 3600)
    minutes = int((predicted_time % 3600) // 60)
    seconds = int((predicted_time % 3600) % 60)
    print("Predicted Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))

    # Save the model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    create_trigger()

    return "Model training complete"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'application/json' in request.content_type:
            try:
                data = request.get_json()
                massa = float(data['massa'])
                suhu_awal = float(data['suhu_awal'])
                kadar_air_awal = float(data['kadar_air_awal'])
                suhu_akhir = float(data['suhu_akhir'])
                kadar_air_akhir = float(data['kadar_air_akhir'])
                jenis_gabah = data['jenis_gabah']

                model, scaler = load_model_and_scaler()

                new_data = pd.DataFrame({
                    'massa': [massa],
                    'suhu_awal': [suhu_awal],
                    'kadar_air_awal': [kadar_air_awal],
                    'suhu_akhir': [suhu_akhir],
                    'kadar_air_akhir': [kadar_air_akhir]
                })

                new_data_scaled = scaler.transform(new_data)

                predicted_time = model.predict(new_data_scaled)[0] * 3600
                
                hours = int(predicted_time // 3600)
                minutes = int((predicted_time % 3600) // 60)
                seconds = int((predicted_time % 3600) % 60)

                # Format predicted_time as string in TIME format
                predicted_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                cursor = mysql.cursor()
                query = "INSERT INTO gabah (jenis, berat, suhu1, suhu2, kadar_air1, kadar_air2, waktu, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())"
                params = (
                    str(jenis_gabah),
                    str(massa),
                    str(suhu_awal),
                    str(suhu_akhir),
                    str(kadar_air_awal),
                    str(kadar_air_akhir),
                    str(predicted_time_str),
                )
                cursor.execute(query, params)
                mysql.commit()

                response = jsonify({
                    'data': {
                        'predicted_time': {
                            'hours': hours,
                            'minutes': minutes,
                            'seconds': seconds
                        }
                    },
                    'message': 'Prediction successful'
                })
                response.status_code = 200
                return response

            except KeyError:
                response = jsonify({
                    'message': 'Invalid JSON data. Make sure to include all the required fields: massa, suhu_awal, kadar_air_awal, suhu_akhir, kadar_air_akhir, jenis_gabah'
                })
                response.status_code = 400
                return response

        else:
            response = jsonify({
                'message': 'Invalid content type. Only application/json is supported'
            })
            response.status_code = 400
            return response

    else:
        response = jsonify({
            'message': 'Method not allowed'
        })
        response.status_code = 405
        return response

@app.route('/gabah')
def display_gabah():
    cursor = mysql.cursor()
    cursor.execute('SELECT * FROM gabah')
    data = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    result = []
    for row in data:
        result.append(dict(zip(columns, row)))

    response = jsonify({
        'data': result,
        'message': 'Data retrieval successful'
    })
    response.status_code = 200
    return response


if __name__ == '__main__':
    app.run(host='10.0.141.93')
