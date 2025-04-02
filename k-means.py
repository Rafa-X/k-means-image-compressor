from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.font import Font
from PIL import Image, ImageTk
from skimage import io
import numpy as np
import os

from sklearn.cluster import KMeans


class ImageCompressor(Frame):
    def __init__(self, main):
        super().__init__(main)
        main.title("Compresión de Imagen con K-Means")
        main.geometry("1060x650")
        self.grid(sticky="n") #posiciona en centro-arriba
        font = Font(family="Consolas", size=11) 

        self.filename = ''  # var for image route selected
        self.compressed_filename = ''

        # controls for functions
        c_functions = Canvas(self, width=100, height=300, bg="gray", bd=5, relief="ridge")
        c_functions.grid(column=0, row=0, columnspan=2)

        button_select = Button(c_functions, text="Seleccionar Imagen", command=self.open_file, font=font)
        #button_select.pack(pady=10, padx=10)
        button_select.grid(column=0, row=0, padx=10, pady=10)

        button_compress = Button(c_functions, text="Comprimir Imagen", command=self.compress_selected_image, font=font)
        #button_compress.pack(pady=10)
        button_compress.grid(column=0, row=1, pady=10)


        # canvas for images
        self.canvas_original = Canvas(self, width=500, height=500, bg="gray", bd=5, relief="ridge")
        self.canvas_original.grid(column=0, row=1, padx=10)
        self.canvas_compressed = Canvas(self, width=500, height=500, bg="gray", bd=5, relief="ridge")
        self.canvas_compressed.grid(column=1, row=1)


    def open_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png")])
        if not self.filename:
            messagebox.showinfo("Aviso", "No se seleccionó ninguna imagen.")
            return
        # carga imagen en canvas
        self.load_image_canvas(self.canvas_original, self.filename)


    def display_images_sizes(self, canvas, filename):
        route = filename
        size = os.path.getsize(route)
        print(f"Tamaño de la imagen original: {size} bytes")


    def save_as_palette_png(self, image_array):
        # Extract the original filename (without extension)
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        self.compressed_filename = f"{base_name}_compressed.png"

        img = Image.fromarray(image_array)
        img = img.convert('P', palette=Image.ADAPTIVE, colors=16)  # Convert to palette mode with 16 colors
        img.save(self.compressed_filename, optimize=True)  # Save with max compression


    def load_image_canvas(self, canvas, filename):
        image = Image.open(filename) # Cargar la imagen
        width, height = image.size # Obtener dimensiones originales
        scale = min(500 / width, 500 / height)  # Calcular escala para mantener proporciones

        # Redimensionar la imagen manteniendo proporciones
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convertir imagen a formato compatible con Tkinter
        tk_image = ImageTk.PhotoImage(image)

        # Limpiar Canvas y mostrar imagen
        canvas.delete("all")
        canvas.create_image(500//2, 500//2, image=tk_image, anchor="center")
        canvas.image = tk_image  # Evitar que Python elimine la referencia

        self.display_images_sizes(canvas, filename)  #display in screen the image size


    def compress_selected_image(self):
        image = io.imread(self.filename)
        rows, cols = image.shape[0], image.shape[1]
        image = image.reshape(rows * cols, 3)

        #print(image[0:100])

        print('\n Compressing... \n')
        kMeans = KMeans(n_clusters=16, n_init=10, random_state=42)
        kMeans.fit(image)

        centers = np.asarray(kMeans.cluster_centers_, dtype=np.uint8)
        labels = np.asarray(kMeans.labels_, dtype=np.uint8)
        labels = np.reshape(labels, (rows, cols))

        newImage = np.zeros((rows, cols, 3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                newImage[i, j, :] = centers[labels[i, j], :]
        
        # Save as palette-based PNG with strict compression
        self.save_as_palette_png(newImage)

        # Loads the image in the canvas
        self.load_image_canvas(self.canvas_compressed, self.compressed_filename)

        print('Image has been compressed successfully.')


if __name__ == '__main__':
    main = Tk()
    app = ImageCompressor(main)
    app.mainloop()