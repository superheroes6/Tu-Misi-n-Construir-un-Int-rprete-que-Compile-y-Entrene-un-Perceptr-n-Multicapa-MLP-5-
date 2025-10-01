# =========================
# Archivo: main.py
# Descripción: Script principal que ejecuta el flujo completo:
# - Carga datos MNIST
# - Entrena MLP desde cero (NumPy)
# - Compila y entrena modelo Keras usando el intérprete
# - Evalúa, grafica y exporta informe PDF
# Uso: Ejecutar para obtener resultados y PDF final.
# =========================

# ---- Imports y configuración ----
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from mlp_numpy import MLPNumPy
from mlp_keras import KerasModelCompiler
from data_utils import MNISTLoader
from report_utils import ReportGenerator

np.random.seed(42)
tf.random.set_seed(42)

# ---- Carga y preprocesamiento de datos ----
trainX_flat, trainY_oh, testX_flat, testY_oh, trainY, testY = MNISTLoader.load_and_preprocess()
print("trainX_flat.shape:", trainX_flat.shape)
print("trainY_oh.shape:", trainY_oh.shape)
print("testX_flat.shape:", testX_flat.shape)
print("testY_oh.shape:", testY_oh.shape)

# ---- Fase 1: Entrenamiento MLP NumPy ----
use_numpy_training = True
if use_numpy_training:
    # Entrenamiento rápido con submuestra para demostración
    n_sample = 5000
    X_small = trainX_flat[:n_sample]
    Y_small = trainY_oh[:n_sample]
    mlp_numpy = MLPNumPy([784, 128, 10], ['relu', 'linear'])
    hist_numpy = mlp_numpy.train(X_small, Y_small, epochs=3, batch_size=128, lr=0.1, verbose=True)
    preds_test = mlp_numpy.predict(testX_flat[:2000])
    acc_test = np.mean(preds_test == testY[:2000])
    print("NumPy MLP test accuracy (first 2000 samples):", acc_test)

# ---- Fase 2: Compilación y entrenamiento Keras ----
arch = "Dense(784,relu) -> Dense(128,relu) -> Dense(10,softmax)"
model = KerasModelCompiler.compile_model(arch, input_shape=(784,))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs_keras = 10
history_obj = model.fit(trainX_flat, trainY_oh, validation_data=(testX_flat, testY_oh),
                        epochs=epochs_keras, batch_size=128, verbose=2)
history = history_obj.history

# ---- Evaluación del modelo Keras ----
loss, acc = model.evaluate(testX_flat, testY_oh, verbose=0)
print(f"\n[Keras MLP] Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

preds_proba = model.predict(testX_flat)
preds = np.argmax(preds_proba, axis=1)
print("\nMatriz de confusión (Keras MLP):")
cm = confusion_matrix(testY, preds)
print(cm)
print("\nInforme de clasificación (Keras MLP):")
print(classification_report(testY, preds, digits=4))

# ---- Graficar resultados ----
ReportGenerator.plot_loss_accuracy(history)
ReportGenerator.plot_confusion_matrix(cm, title='Confusion matrix (Keras MLP)')

# ---- Exportar informe PDF ----
pdf_path = 'informe_mlp_mini.pdf'
ReportGenerator.export_pdf(pdf_path, arch, acc, loss, history, cm)
print(f"\nPDF generado: {pdf_path}")

# ---- Sugerencias para entregar ----
print("\nSugerencias para entregar:")
print("- Incluye celdas Markdown con las respuestas a las preguntas de análisis.")
print("- En la Parte 1 (NumPy) explica limitaciones del backprop manual.")
print("- Entrega el notebook .ipynb y el PDF 'informe_mlp_mini.pdf'.")
