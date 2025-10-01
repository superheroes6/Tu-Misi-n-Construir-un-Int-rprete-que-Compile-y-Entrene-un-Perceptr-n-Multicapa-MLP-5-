# =========================
# Archivo: data_utils.py
# Descripción: Carga y preprocesamiento de los datos MNIST.
# Normaliza, aplana y convierte etiquetas a one-hot.
# Uso: Importar MNISTLoader y usar load_and_preprocess().
# =========================

# ---- Imports ----
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ---- Clase MNISTLoader ----
class MNISTLoader:
    # Utilidad para cargar y preprocesar el dataset MNIST
    @staticmethod
    def load_and_preprocess():
        # Carga los datos, normaliza y convierte etiquetas a one-hot
        (trainX, trainY), (testX, testY) = mnist.load_data()
        trainX = trainX.astype(np.float32) / 255.0
        testX = testX.astype(np.float32) / 255.0
        trainX_flat = trainX.reshape(trainX.shape[0], -1)
        testX_flat  = testX.reshape(testX.shape[0], -1)
        trainY_oh = to_categorical(trainY, 10)
        testY_oh  = to_categorical(testY, 10)
        return trainX_flat, trainY_oh, testX_flat, testY_oh, trainY, testY
