import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\insurance.csv")

print(df.head())
print(df.info())

print(df.describe())
print(df.isnull().sum())

y = df['charges']
X = df.drop('charges', axis=1)
X = X.drop('index', axis=1)

X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'],drop_first=True, dtype=int)
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import joblib
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_dim = X_train_scaled.shape[1]

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error')

train=model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)


sample_data = pd.DataFrame({
    'age': [30, 25],
    'bmi': [25.0, 30.0],
    'children': [0, 2],
    'sex_male': [1, 0],
    'smoker_yes': [0, 1],
    'region_northwest': [0, 1],
    'region_southeast': [1, 0],
    'region_southwest': [0, 0]
})    

sample_data = sample_data[X.columns]
sample_data_scaled = scaler.transform(sample_data)
predictions = model.predict(sample_data_scaled)

print(predictions)

model.save('insurance_model.h5')
print("Model saved as insurance_model.h5")

from sklearn.metrics import r2_score

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
