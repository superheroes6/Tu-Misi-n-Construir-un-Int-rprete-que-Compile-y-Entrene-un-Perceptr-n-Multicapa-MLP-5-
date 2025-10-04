"""
Módulo para gestionar todas las operaciones con Firebase
"""
import io
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

class FirebaseService:
    def __init__(self, credentials_path='firebase-credentials.json', storage_bucket=None):
        """
        Inicializa la conexión con Firebase
        
        Args:
            credentials_path: Ruta al archivo de credenciales JSON
            storage_bucket: Nombre del bucket de Storage (ej: 'tu-proyecto.appspot.com')
        """
        self.db = None
        self.bucket = None
        self.connected = False
        
        try:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.bucket = None  # No usar Storage
            self.connected = True
            print("✅ Firebase conectado correctamente (sin Storage)")
        except Exception as e:
            print(f"❌ Error al conectar Firebase: {e}")
            print("⚠️  La aplicación funcionará sin Firebase")
    
    def is_connected(self):
        """Verifica si Firebase está conectado"""
        return self.connected
    
    def guardar_prediccion(self, imagen_original, imagen_procesada, resultado):
        """
        Guarda una predicción completa en Firebase
        
        Args:
            imagen_original: Objeto PIL Image de la imagen original
            imagen_procesada: Objeto PIL Image de la imagen procesada
            resultado: Dict con 'digito', 'confianza' y 'probabilidades'
            
        Returns:
            Dict con información de la predicción guardada o None si falla
        """
        if not self.connected:
            print("⚠️  Firebase no está conectado, no se guardará la predicción")
            return None

        try:
            prediccion_id = str(uuid.uuid4())
            timestamp = datetime.now()

            # No guardar imágenes en Storage
            url_original = None
            url_procesada = None

            doc_data = {
                'id': prediccion_id,
                'fecha': timestamp,
                'digito_predicho': int(resultado['digito']),
                'confianza': float(resultado['confianza']),
                'probabilidades': resultado['probabilidades'],
                'imagen_original_url': url_original,
                'imagen_procesada_url': url_procesada,
            }

            self.db.collection('predicciones').document(prediccion_id).set(doc_data)

            print(f"✅ Predicción guardada en Firebase: {prediccion_id}")

            return {
                'id': prediccion_id,
                'url_original': url_original,
                'url_procesada': url_procesada,
                'fecha': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            print(f"❌ Error al guardar en Firebase: {e}")
            return None

    def _guardar_imagen_storage(self, imagen, ruta):
        """
        Guarda una imagen en Firebase Storage
        
        Args:
            imagen: Objeto PIL Image
            ruta: Ruta donde se guardará en Storage
            
        Returns:
            URL pública de la imagen
        """
        # No guardar imágenes en Storage
        return None
    
    def obtener_historial(self, limite=50):
        """
        Obtiene el historial de predicciones
        
        Args:
            limite: Número máximo de predicciones a obtener
            
        Returns:
            Lista de predicciones ordenadas por fecha (más recientes primero)
        """
        if not self.connected:
            return []
        
        try:
            predicciones_ref = self.db.collection('predicciones')\
                .order_by('fecha', direction=firestore.Query.DESCENDING)\
                .limit(limite)
            
            predicciones = []
            for doc in predicciones_ref.stream():
                data = doc.to_dict()
                # Convertir timestamp a string
                data['fecha'] = data['fecha'].strftime('%Y-%m-%d %H:%M:%S')
                predicciones.append(data)
            
            return predicciones
            
        except Exception as e:
            print(f"❌ Error al obtener historial: {e}")
            return []
    
    def obtener_prediccion_por_id(self, prediccion_id):
        """
        Obtiene una predicción específica por su ID
        
        Args:
            prediccion_id: ID de la predicción
            
        Returns:
            Dict con los datos de la predicción o None si no existe
        """
        if not self.connected:
            return None
        
        try:
            doc = self.db.collection('predicciones').document(prediccion_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['fecha'] = data['fecha'].strftime('%Y-%m-%d %H:%M:%S')
                return data
            return None
        except Exception as e:
            print(f"❌ Error al obtener predicción: {e}")
            return None
    
    def eliminar_prediccion(self, prediccion_id):
        """
        Elimina una predicción de Firestore y sus imágenes de Storage
        
        Args:
            prediccion_id: ID de la predicción a eliminar
            
        Returns:
            True si se eliminó correctamente, False si hubo error
        """
        if not self.connected:
            return False
        
        try:
            # No eliminar imágenes de Storage
            # Eliminar documento de Firestore
            self.db.collection('predicciones').document(prediccion_id).delete()
            print(f"✅ Predicción eliminada: {prediccion_id}")
            return True
        except Exception as e:
            print(f"❌ Error al eliminar predicción: {e}")
            return False
    
    def obtener_estadisticas(self):
        """
        Obtiene estadísticas generales de las predicciones
        
        Returns:
            Dict con estadísticas (total, por dígito, confianza promedio, etc.)
        """
        if not self.connected:
            return None
        
        try:
            predicciones_ref = self.db.collection('predicciones').stream()
            
            total = 0
            digitos_count = {str(i): 0 for i in range(10)}
            suma_confianza = 0
            
            for doc in predicciones_ref:
                data = doc.to_dict()
                total += 1
                digito = str(data['digito_predicho'])
                digitos_count[digito] += 1
                suma_confianza += data['confianza']
            
            return {
                'total_predicciones': total,
                'distribucion_digitos': digitos_count,
                'confianza_promedio': round(suma_confianza / total, 2) if total > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error al obtener estadísticas: {e}")
            return None