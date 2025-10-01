# =========================
# Proyecto: MLP desde 0 + Intérprete -> Keras + Entrenamiento
# Autor: (tu nombre)
# Ejecutar en: Jupyter Notebook recomendado
# =========================

# -------------------------
# 0) Imports y configuración
# -------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.backends.backend_pdf import PdfPages
import re
import time
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# 1) Carga y preprocesado MNIST (común a ambas fases)
# -------------------------
(trainX, trainY), (testX, testY) = mnist.load_data()
# Normalizar 0-1
trainX = trainX.astype(np.float32) / 255.0
testX = testX.astype(np.float32) / 255.0

# Aplanar para el MLP (784)
trainX_flat = trainX.reshape(trainX.shape[0], -1)
testX_flat  = testX.reshape(testX.shape[0], -1)

# One-hot para Keras y para cálculo de pérdidas
trainY_oh = to_categorical(trainY, 10)
testY_oh  = to_categorical(testY, 10)

print("trainX_flat.shape:", trainX_flat.shape)
print("trainY_oh.shape:", trainY_oh.shape)
print("testX_flat.shape:", testX_flat.shape)
print("testY_oh.shape:", testY_oh.shape)

# ------------------------------------------------------
# 2) FASE 1: MLP "desde cero" con NumPy (forward + backprop)
# ------------------------------------------------------
# Esta implementación es educativa y funciona con redes densas.
# Soporta ReLU y softmax (salida) y entrena con SGD (mini-batch).

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    # x: (batch, classes)
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels_onehot):
    # probs: softmax outputs, labels_onehot: one-hot
    # return average loss
    m = probs.shape[0]
    # clipping for numerical stability
    clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return -np.sum(labels_onehot * np.log(clipped)) / m

class DenseNumpy:
    def __init__(self, n_in, n_out, activation='relu'):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / max(1, n_in))  # He init for ReLU
        self.b = np.zeros((1, n_out))
        self.activation = activation
        # caches for backprop
        self.Z = None
        self.A_prev = None

    def forward(self, A_prev):
        # A_prev: (batch, n_in)
        self.A_prev = A_prev
        self.Z = A_prev.dot(self.W) + self.b  # (batch, n_out)
        if self.activation == 'relu':
            return relu(self.Z)
        elif self.activation == 'linear':
            return self.Z
        else:
            # for output softmax, we return Z and let outer softmax compute
            return self.Z

    def backward(self, dZ, lr):
        # dZ: (batch, n_out)
        m = self.A_prev.shape[0]
        dW = self.A_prev.T.dot(dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        # gradient w.r.t previous layer activation
        dA_prev = dZ.dot(self.W.T)
        # update params
        self.W -= lr * dW
        self.b -= lr * db
        return dA_prev

class MLPNumPy:
    def __init__(self, layer_sizes, activations):
        # layer_sizes: list e.g. [784, 128, 10]
        # activations: list e.g. ['relu', 'linear'] (last should be 'linear' because we use softmax externally)
        assert len(layer_sizes) - 1 == len(activations)
        self.layers = []
        for i in range(len(activations)):
            self.layers.append(DenseNumpy(layer_sizes[i], layer_sizes[i+1], activation=activations[i]))

    def forward(self, X):
        out = X
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        # last layer produce logits
        logits = self.layers[-1].forward(out)
        probs = softmax(logits)
        return probs

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def train(self, X, Y_onehot, epochs=3, batch_size=128, lr=0.01, verbose=True):
        """
        X: (N, input_dim), Y_onehot: (N, n_classes)
        """
        n = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        for ep in range(epochs):
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            Y_shuf = Y_onehot[perm]
            epoch_loss = 0.0
            correct = 0
            t0 = time.time()
            for i in range(0, n, batch_size):
                xb = X_shuf[i:i+batch_size]
                yb = Y_shuf[i:i+batch_size]
                # forward to get logits
                # compute forward manually to get cached values used in backward
                # For full caching, call forward layer by layer like in forward() but keep final logits separately.
                # We'll reuse the structure: last layer stores Z and A_prev, etc.
                probs = self.forward(xb)  # this sets caches in layers
                loss = cross_entropy_loss(probs, yb)
                epoch_loss += loss * xb.shape[0]
                preds = np.argmax(probs, axis=1)
                labels = np.argmax(yb, axis=1)
                correct += np.sum(preds == labels)

                # Backprop:
                # derivative of loss wrt logits for softmax + cross-entropy: (probs - y)/m
                m_batch = xb.shape[0]
                dZ = (probs - yb) / m_batch  # shape (batch, n_classes)
                # backprop through last Dense
                dA_prev = self.layers[-1].backward(dZ, lr)
                # propagate through hidden layers in reverse order
                for l in range(len(self.layers)-2, -1, -1):
                    # activation is ReLU for hidden layers
                    # derivative of ReLU applied to pre-activation Z
                    Z = self.layers[l].Z
                    dZ_hidden = dA_prev * relu_deriv(Z)
                    dA_prev = self.layers[l].backward(dZ_hidden, lr)

            epoch_loss /= n
            epoch_acc = correct / n
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            if verbose:
                print(f"[NumPy MLP] Epoch {ep+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - time: {time.time()-t0:.1f}s")
        return history

# -------------------------
# 2.1) Ejecutar la Fase1 (ejemplo)
# -------------------------
# Para que no sea extremadamente lento por defecto, entrenamos con 1 época y una submuestra de 5000 ejemplos.
use_numpy_training = True
if use_numpy_training:
    n_sample = 5000  # reduce para que sea rápido en CPU; para resultados reales aumenta este número
    X_small = trainX_flat[:n_sample]
    Y_small = trainY_oh[:n_sample]
    mlp_numpy = MLPNumPy([784, 128, 10], ['relu', 'linear'])  # salida linear + softmax externo
    hist_numpy = mlp_numpy.train(X_small, Y_small, epochs=3, batch_size=128, lr=0.1, verbose=True)

    # Evaluar en el test (completo o submuestra)
    preds_test = mlp_numpy.predict(testX_flat[:2000])  # evaluar sobre 2000 para ahorrar tiempo
    acc_test = np.mean(preds_test == testY[:2000])
    print("NumPy MLP test accuracy (first 2000 samples):", acc_test)

# ------------------------------------------------------
# 3) FASE 2: Intérprete / compilador compile_model(architecture_string)
# ------------------------------------------------------
def parse_layer_str(layer_str):
    # Soporta: Dense(256,relu) o Dense( 128 , relu )
    m = re.match(r'\s*([A-Za-z]+)\s*\(\s*([0-9]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*', layer_str)
    if not m:
        raise ValueError(f"Layer string malformed: '{layer_str}'")
    layer_type, units, activation = m.groups()
    return layer_type, int(units), activation.lower()

def compile_model(architecture_string, input_shape):
    """
    Convierte una cadena como "Dense(256,relu) -> Dense(128,relu) -> Dense(10,softmax)"
    en un tf.keras.Sequential con capas Dense.
    input_shape: por ejemplo (784,)
    """
    parts = [p.strip() for p in architecture_string.split('->')]
    model = keras.Sequential()
    first = True
    for p in parts:
        layer_type, units, activation = parse_layer_str(p)
        if layer_type.lower() != 'dense':
            raise NotImplementedError(f"Only Dense supported for now, got: {layer_type}")
        if first:
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Dense(units, activation=activation))
            first = False
        else:
            model.add(layers.Dense(units, activation=activation))
    return model

# Ejemplo: comprobar que compila
arch_example = "Dense(784,relu) -> Dense(64,relu) -> Dense(10,softmax)"
model_test = compile_model(arch_example, input_shape=(784,))
model_test.summary()

# ------------------------------------------------------
# 4) FASE 3: Usar compile_model, compilar, entrenar, evaluar y generar resultados + PDF
# ------------------------------------------------------
# Definir arquitectura (puedes modificarla para mejorar nota)
arch = "Dense(784,relu) -> Dense(128,relu) -> Dense(10,softmax)"
model = compile_model(arch, input_shape=(784,))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar (aumenta epochs si tienes tiempo)
epochs_keras = 10
history = model.fit(trainX_flat, trainY_oh, validation_data=(testX_flat, testY_oh),
                    epochs=epochs_keras, batch_size=128, verbose=2)

# Evaluación final
loss, acc = model.evaluate(testX_flat, testY_oh, verbose=0)
print(f"\n[Keras MLP] Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

# Predicciones para matriz de confusión y reporte
preds_proba = model.predict(testX_flat)
preds = np.argmax(preds_proba, axis=1)
print("\nMatriz de confusión (Keras MLP):")
cm = confusion_matrix(testY, preds)
print(cm)
print("\nInforme de clasificación (Keras MLP):")
print(classification_report(testY, preds, digits=4))

# -------------------------
# 5) Graficas: loss/accuracy y matriz de confusión (visual)
# -------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Matriz de confusión gráfica (simple)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion matrix (Keras MLP)')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# -------------------------
# 6) Exportar PDF de 2 páginas con resultados (entregable)
# -------------------------
pdf_path = 'informe_mlp_mini.pdf'
with PdfPages(pdf_path) as pdf:
    # Página 1: resumen + gráfico accuracy
    fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 portrait approx
    fig1.suptitle('Informe MLP - Página 1', fontsize=14)
    plt.axis('off')
    txt = (
        "Proyecto: MLP desde cero + Intérprete -> Keras\n\n"
        "Arquitectura usada (Keras):\n" + arch + "\n\n"
        f"Resultados (Keras): test_accuracy = {acc:.4f}, test_loss = {loss:.4f}\n\n"
        "Notas:\n- Parte 1: Implementación MLP con NumPy (forward y backprop simple).\n"
        "- Parte 2: Intérprete compile_model(architecture_string) que construye un tf.keras.Sequential.\n"
        "- Parte 3: Entrenamiento y evaluación en MNIST con Keras.\n"
    )
    plt.text(0.01, 0.99, txt, fontsize=10, va='top')
    # añadir una pequeña gráfica de accuracy
    ax = fig1.add_axes([0.1, 0.15, 0.8, 0.35])
    ax.plot(history.history['accuracy'], label='train_acc')
    ax.plot(history.history['val_accuracy'], label='val_acc')
    ax.set_title('Accuracy por epoch (Keras)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    pdf.savefig(fig1)
    plt.close(fig1)

    # Página 2: análisis y matriz de confusión
    fig2 = plt.figure(figsize=(8.27, 11.69))
    fig2.suptitle('Informe MLP - Página 2', fontsize=14)
    plt.axis('off')
    analysis_text = (
        "Análisis y conclusiones\n\n"
        "1) Desafíos de implementar backprop a mano para una red grande:\n"
        "- Complejidad algorítmica y bugs numéricos (overflow/underflow).\n"
        "- Coste computacional: muchas operaciones de matriz y uso intensivo de memoria.\n"
        "- Debugging y optimizaciones (initialization, learning rate schedules, regularización).\n\n"
        "2) Ventajas de un 'compilador' (compile_model):\n"
        "- Abstracción: permite definir arquitecturas de forma legible y reproducible.\n"
        "- Reutilización: cambia la arquitectura sin tocar el código de entrenamiento.\n"
        "- Integración con frameworks: traduce a objetos optimizados y acelerados (e.g., Keras/TensorFlow).\n\n"
        "3) Recomendaciones:\n- Para mejor rendimiento usa CNNs (convolucional) en MNIST.\n- Añadir regularización (Dropout, BatchNorm) y aumentar epochs.\n"
    )
    plt.text(0.01, 0.98, analysis_text, fontsize=10, va='top')

    # añadir la matriz de confusión en la parte inferior
    ax2 = fig2.add_axes([0.1, 0.05, 0.8, 0.45])
    ax2.imshow(cm, interpolation='nearest')
    ax2.set_title('Matriz de confusión (Keras MLP)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    pdf.savefig(fig2)
    plt.close(fig2)

print(f"\nPDF generado: {pdf_path}")

# -------------------------
# 7) Sugerencias para el notebook / entregables
# -------------------------
print("\nSugerencias para entregar:")
print("- Incluye celdas Markdown con las respuestas a las preguntas de análisis.")
print("- En la Parte 1 (NumPy) explica limitaciones del backprop manual.")
print("- Entrega el notebook .ipynb y el PDF 'informe_mlp_mini.pdf'.")

# FIN
