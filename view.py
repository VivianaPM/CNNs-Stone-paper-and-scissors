import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import kagglehub
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# ----------------------------------------------------------------
# Otros archuivos.py
from logic import ImageLoader, NeuronalNetwoker

class WindowApp:
    def __init__(self, root):
        self.tam = 12
        self.root = root
        self.root.title("CNNs Application")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.root.config(background='#f0f0f0')

        # Configuración de estilos
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#ffffff', relief='solid')
        self.style.configure('Header.TLabel', background='#ffffff', font=('Arial', 11, 'bold'))
        self.style.configure('Normal.TLabel', background='#ffffff', font=('Arial', 10))
        self.style.configure('Custom.TButton', font=('Arial', 10, 'bold'))

        # Contenedor principal
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # 1. Path Frame (Superior)
        self.path_frame = ttk.Frame(self.main_container, style='Custom.TFrame', padding="15")
        self.path_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.button = ttk.Button(self.path_frame, text="Abrir Imagen", style='Custom.TButton', 
                               command=self.open_file, width=20)
        self.button.grid(row=0, column=0, padx=(0, 20))
        
        self.button_A = ttk.Button(self.path_frame, text="Analizar", style='Custom.TButton', 
                                command=self.get_data, width=20)
        self.button_A.grid(row=0, column=1, padx=(0, 20))

        self.button_E = ttk.Button(self.path_frame, text="Entrenar", style='Custom.TButton', 
                                command=self.cargar_datos, width=20)
        self.button_E.grid(row=0, column=2, padx=(0, 20))

        self.path_label = ttk.Label(self.path_frame, text="Ninguna imagen seleccionada", 
                                  style='Normal.TLabel')
        self.path_label.grid(row=0, column=3, sticky="w")

        # 2. Visualization Frame (Medio)
        self.visu_frame = ttk.Frame(self.main_container, style='Custom.TFrame', padding="15")
        self.visu_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configurar grid para dos columnas de igual tamaño
        self.visu_frame.grid_columnconfigure(0, weight=1)
        self.visu_frame.grid_columnconfigure(1, weight=1)
        
        # Frame para la imagen (izquierda)
        self.image_frame = ttk.Frame(self.visu_frame, padding="5")
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.image_label = ttk.Label(self.image_frame, text="Vista previa de imagen", 
                                   style='Header.TLabel')
        self.image_label.grid(row=0, column=0, pady=(0, 10))
        
        # Canvas para la imagen con scrollbars
        self.image_canvas = tk.Canvas(self.image_frame, width=500, height=400, 
                                    background='#e0e0e0')
        self.image_canvas.grid(row=1, column=0, sticky="nsew")
        self.image_display = self.image_canvas.create_image(0, 0, anchor="nw")
        
        self.plot_frame = ttk.Frame(self.visu_frame, padding="5")
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self.plot_label = ttk.Label(self.plot_frame, text="Gráfica de Error", 
                                  style='Header.TLabel')
        self.plot_label.grid(row=0, column=0, pady=(0, 10))
        
        # Crear figura de matplotlib
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.plot_ax = self.fig.add_subplot(111)
        
        # Crear el canvas de matplotlib
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")


        # 3. Data Frame (Inferior)
        self.data_frame = ttk.Frame(self.main_container, style='Custom.TFrame', padding="15")
        self.data_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Configurar tres columnas de igual tamaño
        self.data_frame.grid_columnconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(1, weight=1)
        self.data_frame.grid_columnconfigure(2, weight=1)
        
        # Sección de Probabilidades
        self.prob_section = ttk.LabelFrame(self.data_frame, text="Probabilidades", padding="10")
        self.prob_section.grid(row=0, column=0, sticky="ew", padx=5)
        
        labels = [("Papel", 0), ("Piedra", 1), ("Tijera", 2)]
        for label, col in labels:
            ttk.Label(self.prob_section, text=label).grid(row=0, column=col, padx=10)
            entry = ttk.Entry(self.prob_section, width=self.tam)
            entry.grid(row=1, column=col, padx=10, pady=5)
            setattr(self, f"{label.lower()}_entry", entry)

        # Sección de Clasificación
        self.class_section = ttk.LabelFrame(self.data_frame, text="Clasificación", padding="10")
        self.class_section.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(self.class_section, text="Predicción").grid(row=0, column=0, padx=10)
        self.predict = ttk.Entry(self.class_section, width=self.tam)
        self.predict.grid(row=1, column=0, padx=10, pady=5)
        
        ttk.Label(self.class_section, text="Correcta").grid(row=0, column=1, padx=10)
        self.correct = ttk.Entry(self.class_section, width=self.tam)
        self.correct.grid(row=1, column=1, padx=10, pady=5)
        
        # Sección de Métricas
        self.metrics_section = ttk.LabelFrame(self.data_frame, text="Métricas", padding="10")
        self.metrics_section.grid(row=0, column=2, sticky="ew", padx=5)
        
        ttk.Label(self.metrics_section, text="Exactitud").grid(row=0, column=0, padx=10)
        self.accuracy = ttk.Entry(self.metrics_section, width=self.tam)
        self.accuracy.grid(row=1, column=0, padx=10, pady=5)
        
        ttk.Label(self.metrics_section, text="Pérdida").grid(row=0, column=1, padx=10)
        self.loss = ttk.Entry(self.metrics_section, width=self.tam)
        self.loss.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(self.metrics_section, text="Conclusión").grid(row=2, column=0, columnspan=2, pady=(10,0))
        self.conclusion = ttk.Entry(self.metrics_section, width=30)
        self.conclusion.grid(row=3, column=0, columnspan=2, pady=5)

        # Configurar el peso de las filas
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)

    def cargar_datos(self):
        # Descargar el conjunto de datos
        path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
        print("Path to dataset files:", path)

        descripcion = ("paper", "rock", "scissors")
        num_img_clase = 700

        loader = ImageLoader(path, descripcion, num_img_clase)
        imagenes_entrena, clases_entrena, imagenes_prueba, clases_prueba = loader.cargar_imagenes()
        
        # Crear y entrenar la red neuronal
        self.modelo = NeuronalNetwoker()  # Crear la instancia del modelo
        historial = self.modelo.entrenamiento(imagenes_entrena, clases_entrena, epocas=20)  # Entrenar el modelo
        
        # Graficar el historial de error
        self.graphic(historial)

        # los valores de perdida y exactitud de la red
        perdida, exactitud = self.modelo.evaluacion(imagenes_prueba, clases_prueba)
        # Borrar datos anteriores
        self.accuracy.delete(0, tk.END)
        self.loss.delete(0, tk.END)

        self.accuracy.insert(0, f"{exactitud:.4f}")
        self.loss.insert(0, f"{perdida:.4f}")

        print("Datos de entrenamiento y prueba cargados exitosamente.")
        print(f"Número de imágenes de entrenamiento: {len(imagenes_entrena)}")
        print(f"Número de imágenes de prueba: {len(imagenes_prueba)}")

    def graphic(self, historial):
        # Limpiar la gráfica anterior
        self.plot_ax.clear()

        # Crear la nueva gráfica
        if 'loss' in historial.history:
            self.plot_ax.plot(historial.history['loss'], label='Pérdida de entrenamiento')
            self.plot_ax.set_xlabel('Épocas')
            self.plot_ax.set_ylabel('Pérdida')
            self.plot_ax.set_title('Pérdida del modelo durante el entrenamiento')
            self.plot_ax.legend()

            # Ajustar el layout y redibujar
            self.fig.tight_layout()
            self.plot_canvas.draw()

            # Mostrar la gráfica
            plt.show()
        else:
            print("No se encontró la clave 'loss' en el historial.")

    def open_file(self):
        image_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if image_path:
            self.path_label.config(text=f"Imagen seleccionada: {image_path}")
            # Cargar la imagen
            self.loaded_image = Image.open(image_path) #alamcena en imagen para el entrenamiento
            image = self.loaded_image  # Fix: Added this line to reference the image
            
            # Obtener dimensiones del canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Calcular ratio para mantener proporción
            img_ratio = min(canvas_width/image.width, canvas_height/image.height)
            new_width = int(image.width * img_ratio)
            new_height = int(image.height * img_ratio)
            
            # Redimensionar imagen
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Actualizar la imagen en el canvas
            self.image_canvas.delete(self.image_display)
            self.image_display = self.image_canvas.create_image(
                canvas_width//2, canvas_height//2,  # Centrar la imagen
                image=photo, anchor="center"
            )
            self.image_canvas.image = photo  # Mantener referencia


    def get_image(self):
        return self.loaded_image
    
    def get_data(self):
        self.modelo = NeuronalNetwoker()

        # Obtener el valor de clase correcta ingresado por el usuario
        clase_correcta_str = self.correct.get()

        # Mapear el texto ingresado a un índice correspondiente
        clases_map = {"Papel": 0, "Piedra": 1, "Tijera": 2}
        clase_correcta = clases_map.get(clase_correcta_str, None)

        if clase_correcta is None:
            print("Clase correcta no válida. Debe ser 'Papel', 'Piedra' o 'Tijera'.")
            return

        imagen = self.get_image()
        # Realizar la predicción y pasar la clase correcta
        resultado = self.modelo.prediccion(imagen, clase_correcta=clase_correcta)
        # Mostrar los resultados en la interfaz
        self.mostrar_resultados(resultado)

    def mostrar_resultados(self, resultado):
        # Mostrar los resultados de la predicción en los campos correspondientes
        self.predict.delete(0, tk.END)
        self.correct.delete(0, tk.END)
        self.conclusion.delete(0, tk.END)

        # Rellenar los campos con los resultados de la predicción
        self.predict.insert(0, resultado['clase_predicha'])
        self.correct.insert(0, resultado['clase_correcta'] if resultado['clase_correcta'] is not None else "N/A")
        
        # Mostrar si la predicción fue correcta
        conclusion_str = "Correcto" if resultado['Exactitud'] else "Incorrecto" if resultado['Exactitud'] is not None else "N/A"
        self.conclusion.insert(0, conclusion_str)

        # Mostrar las probabilidades en cada entry correspondiente
        for label, col in [("Papel", 0), ("Piedra", 1), ("Tijera", 2)]:
            entry = getattr(self, f"{label.lower()}_entry")
            probabilidad = resultado['probabilidades'][col]
            entry.delete(0, "end")
            entry.insert(0, f"{probabilidad:.4f}")