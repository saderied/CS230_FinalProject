import shap
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



data_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/processed_itineraries.csv"
data = pd.read_csv(data_path)
data = data.head(10_000_000)
data = data.dropna()

feature_names = data.columns.to_list()
feature_names.remove('baseFare')
feature_names.remove('legId')
feature_names.remove('searchDate')
feature_names.remove('flightDate')

# The following is just a copy-paste from the basic model setup
# Could be cleaner but kept for consistency
data = data.drop("legId", axis=1)
data = data.drop("searchDate", axis=1)
data = data.drop("flightDate", axis=1)
data = data * 1
X = data.loc[:, data.columns != 'baseFare']
Y = data.loc[:, data.columns == 'baseFare']

X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# SHapley Additive Explanations
X_sample = X_train[:10000]
X_test_sample = X_test[:10000]
model = tf.keras.models.load_model("/Users/connorjanowiak/Documents/Stanford/CS230/models/basic_model_v2.keras")
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names)

# Using the magnitudes of the gradients (not yet working)
with tf.GradientTape() as tape:
    tape.watch(X_test_sample)
    predictions = model(X_test_sample)
gradients = tape.gradient(predictions, X_test_sample)
feature_importance = np.mean(np.abs(gradients), axis=0)
plt.barh(feature_names, feature_importance)
plt.xlabel("Average Gradient Magnitude")
plt.ylabel("Features")
plt.show()
