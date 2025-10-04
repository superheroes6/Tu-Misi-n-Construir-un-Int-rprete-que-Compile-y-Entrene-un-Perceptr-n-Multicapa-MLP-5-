from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

# Importar nuestro módulo de Firebase
from firebase_service import FirebaseService

app = Flask(__name__)

# Inicializar Firebase Service
# CAMBIAR 'tu-proyecto.appspot.com' por tu bucket real
firebase = FirebaseService(
    credentials_path='firebase-credentials.json',
storage_bucket='mlp-digit-recognizer.appspot.com')

# Cargar el modelo entrenado
print("Cargando modelo...")
modelo = keras.models.load_model('modelo_digitos.h5')
print("Modelo cargado correctamente")

def preprocesar_imagen(imagen):
    """
    Preprocesa la imagen para que sea compatible con el modelo
    """
    # Convertir a escala de grises
    img = imagen.convert('L')
    
    # Redimensionar a 28x28
    img = img.resize((28, 28))
    
    # Convertir a array numpy
    img_array = np.array(img)
    
    # Invertir colores si es necesario (MNIST tiene fondo negro y número blanco)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalizar (0-255 -> 0-1)
    img_array = img_array.astype('float32') / 255
    
    # Expandir dimensiones: (28, 28) -> (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array, img

@app.route('/')
def index():
    return render_template('index.html', firebase_conectado=firebase.is_connected())

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Obtener la imagen del request
        if 'imagen' not in request.files:
            return jsonify({'error': 'No se encontró ninguna imagen'}), 400
        
        archivo = request.files['imagen']
        
        if archivo.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        # Leer la imagen
        imagen_original = Image.open(archivo.stream)
        
        # Preprocesar la imagen
        img_procesada, img_preview = preprocesar_imagen(imagen_original)
        
        # Hacer predicción
        prediccion = modelo.predict(img_procesada, verbose=0)
        
        # Obtener el dígito predicho y la confianza
        digito = int(np.argmax(prediccion[0]))
        confianza = float(np.max(prediccion[0])) * 100
        
        # Obtener todas las probabilidades
        probabilidades = {str(i): float(prob * 100) for i, prob in enumerate(prediccion[0])}
        
        # Preparar resultado
        resultado = {
            'digito': digito,
            'confianza': round(confianza, 2),
            'probabilidades': probabilidades
        }
        
        # Guardar en Firebase usando nuestro módulo
        firebase_result = firebase.guardar_prediccion(imagen_original, img_preview, resultado)
        
        if firebase_result:
            resultado['firebase_id'] = firebase_result['id']
            resultado['guardado_firebase'] = True
        else:
            resultado['guardado_firebase'] = False
        
        # Convertir imagen procesada a base64 para mostrar en frontend
        buffer = io.BytesIO()
        img_preview.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        resultado['imagen_procesada'] = img_base64
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

@app.route('/historial')
def historial():
    """
    Endpoint para obtener el historial de predicciones
    """
    try:
        predicciones = firebase.obtener_historial(limite=50)
        return jsonify({'predicciones': predicciones})
    except Exception as e:
        return jsonify({'error': f'Error al obtener historial: {str(e)}'}), 500

@app.route('/estadisticas')
def estadisticas():
    """
    Endpoint para obtener estadísticas de las predicciones
    """
    try:
        stats = firebase.obtener_estadisticas()
        if stats:
            return jsonify(stats)
        else:
            return jsonify({'error': 'No se pudieron obtener las estadísticas'}), 500
    except Exception as e:
        return jsonify({'error': f'Error al obtener estadísticas: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)