import firebase_admin
from firebase_admin import credentials, firestore

# Inicializar Firebase con la clave privada
cred = credentials.Certificate("firebase_codigo/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Inicializar Firestore
db = firestore.client()
