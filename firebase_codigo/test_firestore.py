from firebase_codigo.firebase_config import db

# Añadir un documento a la colección 'predictions'
doc_ref = db.collection("predictions").add({
    "filename": "test.png",
    "digit": 7
})

print("Documento guardado:", doc_ref)
