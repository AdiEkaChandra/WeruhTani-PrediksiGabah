import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

dataset = pd.read_csv('data.csv')
dataset

print(dataset.describe())
print(len(dataset))
print(dataset.corr())
print(abs(dataset.corr())['kadar_air_awal'].sort_values(ascending=False))

features = ['massa','suhu_awal', 'kadar_air_awal', 'suhu_akhir', 'kadar_air_akhir']
for feature in features:
    plt.scatter(x=dataset[feature], y=dataset['waktu'], label=feature)

plt.xlabel('Feature')
plt.ylabel('Waktu')
plt.title('Relationship between Features and Time')
plt.legend()
plt.show()

dataset = dataset[pd.to_numeric(dataset['massa'], errors='coerce').notnull()]
dataset['massa'] = dataset['massa'].astype('int')

X = dataset[features]
y = dataset['waktu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

new_data = pd.DataFrame({
    'massa': [500],
    'suhu_awal': [22],
    'kadar_air_awal': [20],
    'suhu_akhir': [3],
    'kadar_air_akhir': [16]
})
predicted_time_seconds = model.predict(new_data)

predicted_time_seconds

hours = int(predicted_time_seconds // 3600)
minutes = int((predicted_time_seconds % 3600) // 60)
seconds = int(predicted_time_seconds % 60)

print("Predicted Time: {} jam, {} menit, {} detik".format(hours, minutes, seconds))