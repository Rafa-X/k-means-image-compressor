from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

class ImageCompressor:
    def __init__(self, root):
        root.title("Compresión de Imagen con K-Means")
        root.geometry("400x300")

        self.filename = ''  # var for image route selected
        self.compressed_filename = ''

        label = tk.Label(root, text="Seleccione una imagen para comprimir", font=("Arial", 14))
        label.pack(pady=20)

        button_select = tk.Button(root, text="Seleccionar Imagen", command=self.open_file, font=("Arial", 12))
        button_select.pack(pady=10)

        button_compress = tk.Button(root, text="Comprimir Imagen", command=self.compress_selected_image, font=("Arial", 12))
        button_compress.pack(pady=10)

    def open_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png")])
        if not self.filename:
            messagebox.showinfo("Aviso", "No se seleccionó ninguna imagen.")
            return
        print(self.filename)

    def print_images_sizes(self):
        route_1 = self.filename
        route_2 = self.compressed_filename

        size_1 = os.path.getsize(route_1)
        size_2 = os.path.getsize(route_2)

        print(f"Tamaño de la imagen original: {size_1} bytes")
        print(f"Tamaño de la imagen comprimida: {size_2} bytes")


    def save_as_palette_png(self, image_array):
        # Extract the original filename (without extension)
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        self.compressed_filename = f"{base_name}_compressed.png"

        img = Image.fromarray(image_array)
        img = img.convert('P', palette=Image.ADAPTIVE, colors=16)  # Convert to palette mode with 16 colors
        img.save(self.compressed_filename, optimize=True)  # Save with max compression

    def compress_selected_image(self):
        image = io.imread(self.filename)
        rows, cols = image.shape[0], image.shape[1]
        image = image.reshape(rows * cols, 3)

        print('Compressing...')
        print('Note: This can take a while for a large image file.')
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

        print('Image has been compressed successfully.')
        self.print_images_sizes()

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageCompressor(root)
    root.mainloop()
