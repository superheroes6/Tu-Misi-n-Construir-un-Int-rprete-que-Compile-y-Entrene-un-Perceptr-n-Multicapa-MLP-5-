# =========================
# Archivo: mlp_keras.py
# Descripción: Intérprete/compilador que convierte una cadena de arquitectura en un modelo Keras Sequential.
# Permite definir redes densas de forma legible y reproducible.
# Uso: Importar KerasModelCompiler y usar compile_model(architecture_string, input_shape).
# =========================

# ---- Imports ----
import re
from tensorflow import keras
from tensorflow.keras import layers

# ---- Clase KerasModelCompiler ----
class KerasModelCompiler:
    # Intérprete que traduce una cadena de arquitectura a un modelo Keras Sequential

    @staticmethod
    def parse_layer_str(layer_str):
        # Parsea una capa tipo "Dense(128,relu)" y extrae tipo, unidades y activación
        m = re.match(r'\s*([A-Za-z]+)\s*\(\s*([0-9]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*', layer_str)
        if not m:
            raise ValueError(f"Layer string malformed: '{layer_str}'")
        layer_type, units, activation = m.groups()
        return layer_type, int(units), activation.lower()

    @staticmethod
    def compile_model(architecture_string, input_shape):
        # Convierte la cadena de arquitectura en un modelo Keras Sequential
        parts = [p.strip() for p in architecture_string.split('->')]
        model = keras.Sequential()
        first = True
        for p in parts:
            layer_type, units, activation = KerasModelCompiler.parse_layer_str(p)
            if layer_type.lower() != 'dense':
                raise NotImplementedError(f"Only Dense supported for now, got: {layer_type}")
            if first:
                model.add(layers.Input(shape=input_shape))
                model.add(layers.Dense(units, activation=activation))
                first = False
            else:
                model.add(layers.Dense(units, activation=activation))
        return model
