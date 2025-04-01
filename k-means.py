from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class ImageCompressor:
    def __init__(self, root):
        root.title("Compresión de Imagen con K-Means")
        root.geometry("400x300")

        self.filename = '' # var for image route selected

        label = tk.Label(root, text="Seleccione una imagen para comprimir", font=("Arial", 14))
        label.pack(pady=20)

        button_select = tk.Button(root, text="Seleccionar Imagen", command=self.open_file, font=("Arial", 12))
        button_select.pack(pady=10)

        button_compress = tk.Button(root, text="Comprimir Imagen", command=self.compress_selected_image, font=("Arial", 12))
        button_compress.pack(pady=10)

        print(self.filename)
        # reading the image
        #self.image = io.imread(self.filename)

        
    def compress_selected_image(self):
        image = io.imread(self.filename)
        # preprocessing
        rows, cols = image.shape[0], image.shape[1]
        image = image.reshape(rows * cols, 3)

        # modelling
        print('Compressing...')
        print('Note: This can take a while for a large image file.')
        kMeans = KMeans(n_clusters = 16)
        kMeans.fit(image)

        # getting centers and labels
        centers = np.asarray(kMeans.cluster_centers_, dtype=np.uint8)
        labels = np.asarray(kMeans.labels_, dtype = np.uint8)
        labels = np.reshape(labels, (rows, cols))
        print('Almost done.')

        # reconstructing the image
        newImage = np.zeros((rows, cols, 3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                    # assinging every pixel the rgb color of their label's center
                    newImage[i, j, :] = centers[labels[i, j], :]
        io.imsave(self.filename.split('.')[0] + 'compressed.png', newImage)

        print('Image has been compressed sucessfully.')
        self.print_images_sizes()

    def print_images_sizes(self):
        # Ruta del archivo
        route_1 = './dog.png'
        route_2 = './compressed.png'

        # Obtener el tamaño en bytes
        size_1 = os.path.getsize(route_1)
        size_2 = os.path.getsize(route_2)

        print(f"El tamaño del archivo es: {size_1} bytes")
        print(f"El tamaño del archivo es: {size_2} bytes")

    def open_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not self.filename:
            messagebox.showinfo("Aviso", "No se selecciono ninguna imagen.")
            return
        print(self.filename)

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageCompressor(root)
    root.mainloop()