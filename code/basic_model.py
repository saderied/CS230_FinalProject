import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/processed_itineraries.csv"
data = pd.read_csv(data_path)
data = data.head(10_000_000)
data = data.dropna()

data = data.drop("legId", axis=1)
data = data.drop("searchDate", axis=1)
data = data.drop("flightDate", axis=1)
# This is a hack for converting booleans to integers
data = data * 1
X = data.loc[:, data.columns != 'baseFare']
Y = data.loc[:, data.columns == 'baseFare']

X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)
model.save("../models/basic_model_v2.keras")

test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error: {test_mae}")
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig("../plots/basic_model_v2_plot.png")
plt.show()