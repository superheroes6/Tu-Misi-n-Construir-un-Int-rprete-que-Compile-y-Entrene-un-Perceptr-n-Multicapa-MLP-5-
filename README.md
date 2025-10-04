# Proyecto MNIST Flask App

Este proyecto implementa una aplicación web con Flask para predecir dígitos escritos a mano usando un modelo MLP entrenado con MNIST.

## Estructura del proyecto

```
mnist_flask_app/
│
├─ model/
│   └─ mlp_model.h5                # modelo entrenado
│
├─ app/
│   ├─ __init__.py                 # inicia la app Flask
│   ├─ routes.py                   # rutas principales de la web
│   ├─ predictor.py                # clase que carga el modelo y predice
│   ├─ utils.py                    # funciones auxiliares
│   └─ forms.py                    # (opcional) formularios con Flask-WTF
│
├─ templates/
│   └─ index.html                  # página web
│
├─ static/
│   └─ style.css                   # estilos CSS (opcional)
│
├─ firebase.py                     # aquí se guardan las imágenes y resultados 
│
├─ app.py                          # punto de entrada principal
├─ requirements.txt
└─ README.md
```

Ejecutar app.py para ejecutar el codigo

https://github.com/superheroes6/Tu-Misi-n-Construir-un-Int-rprete-que-Compile-y-Entrene-un-Perceptr-n-Multicapa-MLP-5-.git
