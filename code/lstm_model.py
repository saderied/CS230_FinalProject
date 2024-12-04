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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=11)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=11)

class StackedLSTMModel(tf.keras.Model):
    """
    This class is adapted from LSTM model outlined in the paper found here:
        https://doi.org/10.3390/app13106032
    
    Parameters:
        hidden_sizes: a list of integers indicating the numbber of nodes per LSTM layer
        dropout: a float that represents the dropout percentage in each layer
        output_size: size of the last layer in the network (default is 1) 
    """
    def __init__(self, hidden_sizes, dropout, output_size=1):
        super(StackedLSTMModel, self).__init__()
        
        self.lstm_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    hidden_size,
                    return_sequences=(i != len(hidden_sizes) - 1),
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    activation="tanh"
                )
            )
        
        self.output_layer = tf.keras.layers.Dense(output_size)

    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        x = self.output_layer(x)
        return x
    
hidden_sizes = [64, 32, 16]
dropout = 0.3

model = StackedLSTMModel(hidden_sizes, dropout)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)
model.save("../models/lstm_dropout_0.3.keras")

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error: {test_mae}")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig("../plots/lstm_model_v1_plot.png")
plt.show()