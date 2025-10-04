import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset MNIST
print("Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar las imágenes (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Expandir dimensiones para que sea compatible con CNN
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Datos de prueba: {x_test.shape}")

# Crear la red neuronal convolucional (CNN)
modelo = keras.Sequential([
    # Primera capa convolucional
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Segunda capa convolucional
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Aplanar y capas densas
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 clases (0-9)
])

# Compilar el modelo
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nArquitectura del modelo:")
modelo.summary()

# Entrenar el modelo
print("\nEntrenando modelo...")
history = modelo.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluar el modelo
print("\nEvaluando modelo...")
score = modelo.evaluate(x_test, y_test, verbose=0)
print(f"Precisión en test: {score[1]*100:.2f}%")

# Guardar el modelo
modelo.save("modelo_digitos.h5")
print("\n¡Modelo guardado como 'modelo_digitos.h5'!")

# Visualizar algunas predicciones
predicciones = modelo.predict(x_test[:5])
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = np.argmax(predicciones[i])
    real = y_test[i]
    plt.title(f"Pred: {pred}\nReal: {real}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("predicciones_ejemplo.png")
print("Ejemplos guardados en 'predicciones_ejemplo.png'")