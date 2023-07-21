from tabnanny import verbose
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.system('clear')

dolares = np.array([], dtype=float)
pesos = np.array([], dtype=float)

for i in range(1, 1001):
    numero_aleatorio = np.random.rand()
    numero_aleatorio_esc = 1 + numero_aleatorio * 999
    numero_aleatorio_esc = round(numero_aleatorio_esc, 3)
    dolares = np.append(dolares, numero_aleatorio_esc)

    pesos = np.append(pesos, dolares[i-1] * 3963.75)


capa1 = tf.keras.layers.Dense(units=10, input_shape=[1])
capa2 = tf.keras.layers.Dense(units=10)
capa3 = tf.keras.layers.Dense(units=5)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa1, capa2, capa3, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Entrando UwU")
historial = modelo.fit(dolares, pesos, epochs=100, verbose=False)
print("Entrenado UwU")

plt.xlabel('# Epoca')
plt.ylabel('Magnitud de perida')
plt.plot(historial.history["loss"])
plt.show()

print("Prediccion")
resultado = modelo.predict([int(input("¿Cuantos dolares quieres convertir chamo?"))])
print("El resultado es :" +str(resultado) + "Dolares")
