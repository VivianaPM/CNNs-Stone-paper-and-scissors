# import os
# def verificar_descarga_imagenes(path):
#     # Descripción de clases
#     clases = ["paper", "rock", "scissors"]

#     # Verificar que las carpetas para cada clase existan
#     for clase in clases:
#         clase_path = os.path.join(path, clase)
#         if os.path.isdir(clase_path):
#             print(f"Carpeta '{clase}' encontrada.")
#         else:
#             print(f"Carpeta '{clase}' NO encontrada.")

#     # Verificar la cantidad de imágenes en cada carpeta
#     for clase in clases:
#         clase_path = os.path.join(path, clase)
#         if os.path.isdir(clase_path):
#             imagenes = [f for f in os.listdir(clase_path) if f.endswith('.png')]  # Filtrar solo archivos .png
#             print(f"Carpeta '{clase}' contiene {len(imagenes)} imágenes.")
#         else:
#             print(f"Carpeta '{clase}' NO contiene imágenes.")
import os
# ----------------------------------------------------------------
# Construcción de modelo
import tensorflow as tf
# ----------------------------------------------------------------
# Operaciones atematicas
import random
import numpy as np
# ----------------------------------------------------------------
# Manejo de imagenes
from PIL import Image
# ----------------------------------------------------------------
# Graficos
import matplotlib.pyplot as plt


class ImageLoader:
    def __init__(self, path, clases, num_img_clase):
        self.path = path
        self.clases = clases
        self.num_img_clase = num_img_clase
        self.num_entrena = round(num_img_clase * 0.70)
        self.num_prueba = round(num_img_clase * 0.30)
        self.imagenes_entrena = []
        self.clases_entrena = []
        self.imagenes_prueba = []
        self.clases_prueba = []

    def cargar_imagenes(self):
        for idx, clase in enumerate(self.clases):
            clase_path = os.path.join(self.path, clase)
            archivos_imagen = [f for f in os.listdir(clase_path) if f.endswith('.png')]
            random.shuffle(archivos_imagen)

            # Dividir en entrenamiento y prueba
            archivos_entrena = archivos_imagen[:self.num_entrena]
            archivos_prueba = archivos_imagen[self.num_entrena:self.num_entrena + self.num_prueba]

            # Cargar imágenes de entrenamiento
            for archivo in archivos_entrena:
                imagen = Image.open(os.path.join(clase_path, archivo)).resize((200, 300))
                imagen_array = np.array(imagen) / 255.0  # Escala al rango [0, 1]
                self.imagenes_entrena.append(imagen_array)
                self.clases_entrena.append(idx)  # Asigna identificador de clase

            # Cargar imágenes de prueba
            for archivo in archivos_prueba:
                imagen = Image.open(os.path.join(clase_path, archivo)).resize((200, 300))
                imagen_array = np.array(imagen) / 255.0  # Escala al rango [0, 1]
                self.imagenes_prueba.append(imagen_array)
                self.clases_prueba.append(idx)  # Asigna identificador de clase

        # Convertir listas a arrays de numpy
        self.imagenes_entrena = np.array(self.imagenes_entrena)
        self.clases_entrena = np.array(self.clases_entrena)
        self.imagenes_prueba = np.array(self.imagenes_prueba)
        self.clases_prueba = np.array(self.clases_prueba)

        return self.imagenes_entrena, self.clases_entrena, self.imagenes_prueba, self.clases_prueba

class NeuronalNetwoker:
    def __init__(self):
        self.modelo = self.Construir_Modelo()

    def Construir_Modelo(self):
        modelo = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 300, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        modelo.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        return modelo

    def entrenamiento(self, img_E, clases_E, epocas = 50):
        historial = self.modelo.fit(img_E, clases_E, epochs=epocas)
        return historial
    
    def evaluacion(self, img_P, clases_p):
        loss, accuracy = self.modelo.evaluate(img_P, clases_p)
        return loss, accuracy
    
    def prediccion(self, imagen, clase_correcta=None):
        # Normalizar la imagen al rango [0, 1]
        imagen = np.array(imagen) / 255.0

        # Agregar dimensión para representar el batch si es necesario
        if len(imagen.shape) == 3:
            imagen = np.expand_dims(imagen, axis=0)

        # Realizar predicción
        predicciones = self.modelo.predict(imagen)
        clase_predicha = np.argmax(predicciones, axis=1)[0]

        # Almacenar probabilidades para cada clase
        probabilidad = predicciones[0]

        # Verificar si la predicción fue correcta, solo si `clase_correcta` es válido
        if clase_correcta is not None and 0 <= clase_correcta <= 2:
            acierto = (clase_predicha == clase_correcta)
        else:
            acierto = None  # No se verifica exactitud si `clase_correcta` no es válido

        # Construir el diccionario de resultados
        resultado = {
            "probabilidades": probabilidad,
            "clase_predicha": clase_predicha,
            "clase_correcta": clase_correcta,
            "Exactitud": acierto
        }

        return resultado