# Importación de las dependencias.
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Se leen los datos del archivo csv.
datos = pd.read_csv('altura_peso.csv')
altura = datos['Altura'].values
peso = datos['Peso'].values

# Utilización del Modelo en Keras.
modelo = Sequential([
    Input(shape=(1,)),
    Dense(1, activation='linear')
])
optimizador = SGD(learning_rate=0.0004)
modelo.compile(optimizer=optimizador, loss='mse')

# Se entrena el módelo.
historial = modelo.fit(altura, peso, epochs=10000, batch_size=len(altura), verbose=0)

# Visualización de los datos, obtenemos y mostramos los parámetros w y b.
peso_w, sesgo_b = modelo.layers[0].get_weights()
print("Parámetros del modelo entrenado:")
print(f"  - Coeficiente (w): {peso_w[0][0]:.4f}")
print(f"  - Sesgo (b): {sesgo_b[0]:.4f}")

# Grafica de la pérdida durante las épocas.
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'])
plt.title('MSE vs Épocas')
plt.xlabel('Épocas')
plt.ylabel('MSE')

# Gráfico de la recta de regresión de los datos.
plt.subplot(1, 2, 2)
plt.scatter(altura, peso, color='blue', label='Datos originales')
plt.plot(altura, modelo.predict(altura), color='red', label='Recta de regresión')
plt.title('Regresión Lineal con Keras')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# Predicción.
altura_prediccion = 176
altura_np = np.array([[altura_prediccion]])
peso_predicho = modelo.predict([altura_np])
print(f"\nPredicción:")
print(f"  - Para una altura de {altura_prediccion} cm, el peso predicho es {peso_predicho[0][0]:.2f} kg")
