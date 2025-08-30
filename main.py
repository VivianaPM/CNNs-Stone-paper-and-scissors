import psutil
import tensorflow as tf
import subprocess
import os
# # Importe para entablar conexion
# import kagglehub
# from logic import ImageLoader
# Importe de tkinter
import tkinter as tk
#Importe de archivos 
from view import WindowApp
# -------------------------------------------------
# Reducir el nivel de registro de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Desactivar operaciones personalizadas oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # # Descargar el conjunto de datos
    # path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    # print("Path to dataset files:", path)

    # descripcion = ("paper", "rock", "scissors")
    # num_img_clase = 700

    # loader = ImageLoader(path, descripcion, num_img_clase)
    # imagenes_entrena, clases_entrena, imagenes_prueba, clases_prueba = loader.cargar_imagenes()

    # # Aquí puedes continuar con el uso de las imágenes en tu red CNN
    # print("Datos de entrenamiento y prueba cargados exitosamente.")
    
    # # Por ejemplo:
    # print(f"Número de imágenes de entrenamiento: {len(imagenes_entrena)}")
    # print(f"Número de imágenes de prueba: {len(imagenes_prueba)}")

    # Crear una instancia de la ventana de la aplicación
    root = tk.Tk()
    app = WindowApp(root)
    root.mainloop()

def get_system_info():
    # Información de la CPU
    cpu_info = psutil.cpu_times_percent(interval=1)
    max_cpu_usage = max(cpu_info.user, cpu_info.system)

    # Información de la RAM
    virtual_memory = psutil.virtual_memory()
    max_ram_usage = virtual_memory.total / (1024 ** 3)  # en GB

    # Información de la GPU (usando TensorFlow)
    gpu_info = None
    if tf.config.list_physical_devices('GPU'):
        gpu_info = tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])
        max_gpu_memory = gpu_info.get('memory_limit', 'No info') / (1024 ** 2)  # en MB
    else:
        max_gpu_memory = 'No GPU detected'

    # También podemos utilizar nvidia-smi si quieres más información específica de la GPU
    try:
        gpu_nvidia_info = subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", shell=True)
        max_gpu_memory = gpu_nvidia_info.decode('utf-8').strip() + " MB"
    except Exception as e:
        gpu_nvidia_info = "No GPU detected or nvidia-smi not available"
        
    # Mostramos la información recopilada
    print(f"Uso máximo de CPU: {max_cpu_usage}%")
    print(f"Uso máximo de RAM: {max_ram_usage:.2f} GB")


if __name__ == "__main__":
    get_system_info()
    main()
