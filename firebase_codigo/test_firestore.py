# test_firestore.py
from firebase_admin import credentials, firestore

# Importa la configuración si está en firebase_config.py
from firebase_codigo.firebase_config import db  # Ajusta la ruta según tu estructura

# Crear una colección si no existe
try:
    collection_ref = db.collection('nombre_coleccion')
    collection_ref.document('doc1').set({'campo': 'valor'})
    print("Colección creada con éxito.")
except Exception as e:
    print(f"Error al crear la colección: {e}")