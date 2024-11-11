import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import h5py


with h5py.File('/Users/connorjanowiak/Documents/Stanford/CS230/data/temporal_data.h5', 'r') as hf:
    X = hf['X'][:]
    y = hf['y'][:]


scaler = MinMaxScaler()

num_samples, timesteps, num_features = X.shape
X_reshaped = X.reshape(-1, num_features)
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(num_samples, timesteps, num_features)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

class StackedGRUModel(tf.keras.Model):
    def __init__(self, hidden_sizes, dropout, output_size):
        super(StackedGRUModel, self).__init__()

        self.gru_layers = []

        for i, hidden_size in enumerate(hidden_sizes):
            self.gru_layers.append(
                tf.keras.layers.GRU(
                    hidden_size,
                    return_sequences=(i != len(hidden_sizes) - 1),
                    dropout = dropout,
                    activation="tanh"
                )
            )
        
        self.output_layer = tf.keras.layers.Dense(output_size)

    def call(self, x, training=False):
        for gru in self.gru_layers:
            x = gru(x, training=training)
        x = self.output_layer(x)
        return x
    
hidden_sizes = [64, 32, 16]
dropout = 0.3
output_size = 1

model = StackedGRUModel(hidden_sizes, dropout, output_size)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


