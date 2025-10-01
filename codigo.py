# Importamos todas las librerías necesarias para trabajar con arrays, deep learning, visualización y métricas
import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Cargamos el dataset MNIST, que contiene imágenes de dígitos escritos a mano.
# trainX y trainY son las imágenes y etiquetas de entrenamiento, testX y testY las de prueba
(trainX, trainY), (testX, testY) = mnist.load_data()

# Mostramos la forma de los arrays para entender cómo están organizados los datos
print("TrainX:", trainX.shape)  # Número de imágenes de entrenamiento, alto y ancho
print("TrainY:", trainY.shape)  # Número de etiquetas de entrenamiento
print("TestX:", testX.shape)    # Número de imágenes de prueba
print("TestY:", testY.shape)    # Número de etiquetas de prueba

# Creamos una función para visualizar un dígito cualquiera del dataset
def display_digit(index):
    image = trainX[index].reshape([28, 28])  # Damos forma a la imagen (28x28 píxeles)
    label = trainY[index]                     # Obtenemos la etiqueta real del dígito
    plt.title(f"Training data, index: {index}, Label: {label}")
    plt.imshow(image, cmap='gray_r')         # Mostramos la imagen en escala de grises
    plt.show()

# Por ejemplo, mostramos el dígito en la posición 2
display_digit(2)

# Normalizamos los valores de píxeles a un rango 0-1 dividiendo entre 255
# Esto ayuda al entrenamiento de la red neuronal
trainX_norm = trainX / 255.0
testX_norm = testX / 255.0

# Convertimos las etiquetas a formato one-hot para clasificación multi-clase
# Por ejemplo, el dígito 3 pasa a [0,0,0,1,0,0,0,0,0,0]
trainY_cate = to_categorical(trainY)
testY_cate = to_categorical(testY)

# Calculamos la dimensión de entrada de la red neuronal (28x28 = 784 píxeles)
dimension_input = trainX.shape[1] * trainX.shape[2]

# Creamos una función que define nuestro modelo MLP (Perceptrón Multicapa)
def build_model():
    model = Sequential()  
    # La primera capa recibe vectores de 784 valores (los píxeles aplanados)
    model.add(Input(shape=(dimension_input,)))
    # Capa oculta con 784 neuronas y activación ReLU
    model.add(Dense(dimension_input, activation='relu'))
    # Segunda capa oculta con 64 neuronas y activación ReLU
    model.add(Dense(64, activation='relu'))
    # Capa de salida con 10 neuronas (una por dígito) y activación softmax
    model.add(Dense(10, activation='softmax'))

    # Compilamos el modelo indicando:
    # - Optimizer: 'adam' ajusta los pesos automáticamente
    # - Loss: 'categorical_crossentropy' porque es clasificación multi-clase
    # - Metrics: 'accuracy' para evaluar cuántos aciertos tiene
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Creamos el modelo llamando a la función
model_tofit = build_model()

# Mostramos un resumen del modelo, con el número de parámetros de cada capa
model_tofit.summary()

# Aplanamos las imágenes para que cada fila sea un vector de 784 valores
trainX_norm = trainX_norm.reshape(trainX_norm.shape[0], dimension_input)
testX_norm = testX_norm.reshape(testX_norm.shape[0], dimension_input)

# Entrenamos el modelo con los datos normalizados y las etiquetas one-hot
# Validamos con el set de prueba, usamos 5 épocas y batch size de 128
model_tofit.fit(
    trainX_norm,
    trainY_cate,
    validation_data=(testX_norm, testY_cate),
    epochs=5,
    batch_size=128,
    verbose=2
)

# Realizamos predicciones sobre el set de prueba
# argmax obtiene la clase con mayor probabilidad para cada imagen
predictions = np.array(model_tofit.predict(testX_norm)).argmax(axis=1)

# Convertimos las etiquetas reales a valores numéricos (de one-hot a dígito)
actual = testY_cate.argmax(axis=1)

# Calculamos la precisión del modelo como el porcentaje de aciertos
test_accuracy = np.mean(predictions == actual)
print("\n✅ Test accuracy:", test_accuracy)

# Generamos la matriz de confusión, que muestra cuántos dígitos fueron clasificados correctamente e incorrectamente
cm = confusion_matrix(y_true=actual, y_pred=predictions)
print("\n📊 Matriz de confusión:")
print(cm)

# Generamos un informe de clasificación detallado, con precisión, recall y f1-score para cada dígito
print("\n📄 Informe de clasificación:")
print(classification_report(actual, predictions))
