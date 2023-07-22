from tabnanny import verbose
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
os.system('cls')

multiplicando = np.array([], dtype=float)
multiplicador = np.array([], dtype=float)

for i in range(1, 101):
    numero_random = random.randint(1, 100)
    multiplicando = np.append(multiplicando, numero_random)
    multiplicador = np.append(multiplicador, multiplicando[i-1] * 2)


capa1 = tf.keras.layers.Dense(units=10, input_shape=[1], activation='linear')
capa2 = tf.keras.layers.Dense(units=10)
salida = tf.keras.layers.Dense(units=1, activation='linear')
modelo = tf.keras.Sequential([capa1, capa2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Entrenando UwU...")
historial = modelo.fit(multiplicando, multiplicador, epochs=130, verbose=False)
print("Ya entrené 7w7")

plt.xlabel('# Epoca')
plt.ylabel('Magnitud de perida')
plt.plot(historial.history["loss"])
#plt.show()

resultado = modelo.predict([int(input("Dame un numero pa multiplicarlo por 2: "))])
print("El resultado es: " , resultado)
