# =========================
# Archivo: mlp_numpy.py
# Descripción: Implementación educativa de un MLP (perceptrón multicapa) desde cero usando NumPy.
# Contiene clases y funciones para el forward, backpropagation y entrenamiento con SGD.
# Uso: Importar la clase MLPNumPy y usar para entrenar y evaluar en MNIST.
# =========================

# ---- Imports ----
import numpy as np
import time

# ---- Funciones de activación y pérdida ----
def relu(x):
    # Función de activación ReLU (Rectified Linear Unit)
    return np.maximum(0, x)

def relu_deriv(x):
    # Derivada de ReLU, necesaria para el backpropagation
    return (x > 0).astype(np.float32)

def softmax(x):
    # Softmax para la capa de salida (clasificación multiclase)
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels_onehot):
    # Cálculo de la pérdida cross-entropy para clasificación
    m = probs.shape[0]
    clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return -np.sum(labels_onehot * np.log(clipped)) / m

# ---- Clase DenseNumpy ----
class DenseNumpy:
    # Capa densa implementada desde cero (forward y backward)
    def __init__(self, n_in, n_out, activation='relu'):
        # Inicialización de pesos y sesgos
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / max(1, n_in))
        self.b = np.zeros((1, n_out))
        self.activation = activation
        # Variables para almacenar valores intermedios (caches)
        self.Z = None
        self.A_prev = None

    def forward(self, A_prev):
        # Propagación hacia adelante (forward)
        self.A_prev = A_prev
        self.Z = A_prev.dot(self.W) + self.b
        if self.activation == 'relu':
            return relu(self.Z)
        elif self.activation == 'linear':
            return self.Z
        else:
            return self.Z

    def backward(self, dZ, lr):
        # Propagación hacia atrás (backward) y actualización de parámetros
        m = self.A_prev.shape[0]
        dW = self.A_prev.T.dot(dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ.dot(self.W.T)
        self.W -= lr * dW
        self.b -= lr * db
        return dA_prev

# ---- Clase MLPNumPy ----
class MLPNumPy:
    # Red neuronal multicapa (MLP) usando varias capas DenseNumpy
    def __init__(self, layer_sizes, activations):
        # layer_sizes: lista de tamaños de capa, activations: lista de funciones de activación
        assert len(layer_sizes) - 1 == len(activations)
        self.layers = []
        for i in range(len(activations)):
            self.layers.append(DenseNumpy(layer_sizes[i], layer_sizes[i+1], activation=activations[i]))

    def forward(self, X):
        # Propagación hacia adelante por todas las capas
        out = X
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        logits = self.layers[-1].forward(out)
        probs = softmax(logits)
        return probs

    def predict(self, X):
        # Predicción de clases (índice de la mayor probabilidad)
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def train(self, X, Y_onehot, epochs=3, batch_size=128, lr=0.01, verbose=True):
        # Entrenamiento usando mini-batch SGD y backpropagation manual
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
                probs = self.forward(xb)
                loss = cross_entropy_loss(probs, yb)
                epoch_loss += loss * xb.shape[0]
                preds = np.argmax(probs, axis=1)
                labels = np.argmax(yb, axis=1)
                correct += np.sum(preds == labels)
                m_batch = xb.shape[0]
                dZ = (probs - yb) / m_batch
                dA_prev = self.layers[-1].backward(dZ, lr)
                for l in range(len(self.layers)-2, -1, -1):
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
