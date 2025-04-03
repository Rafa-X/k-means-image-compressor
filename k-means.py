import numpy as np
import os
import cv2
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.font import Font
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageCompressor(Frame):
    def __init__(self, main):
        super().__init__(main)
        main.title("Compresión de Imagen con K-Means")
        main.geometry("1060x660")
        self.grid(sticky="n")
        font = Font(family="Consolas", size=11) 

        self.filename = ''  # image path 
        self.compressed_filename = ''  # image path
        self.centroids = None
        self.palette_original = None

        # canvas for controls and information
        c_graphics = Canvas(self, width=400, height=300, bg="#273357", bd=5, relief="ridge")
        c_graphics.grid(column=0, row=0, columnspan=2)
        c_functions = Canvas(c_graphics, width=400, height=50, bg="#273357")
        c_functions.grid(column=0, row=0, columnspan=5)

        # K value input
        k_label = Label(c_functions, text="K: ", font=font, bd=2, relief="solid", bg="#991b00", fg="white")
        k_label.grid(column=2, row=0, pady=10, sticky="e")
        default = StringVar()
        default.set("16")  # default value for K -> 16 colours to be compressed
        k_value = Entry(c_functions, font=font, bd=2, relief="solid", width=5, textvariable=default)
        k_value.grid(column=3, row=0, padx=(0, 10), pady=10, sticky="w")

        # controls for select and k-means compress functions
        button_select = Button(c_functions, text="Select Image", command=self.open_file, font=font, bd=2, relief="solid", bg="#fffed0")
        button_select.grid(column=0, row=0, padx=(10, 0), pady=10)

        button_compress = Button(c_functions, text="Compress Image", command=lambda:self.compress_image(int(k_value.get())), font=font, bd=2, relief="solid", bg="#DAF7A6")
        button_compress.grid(column=1, row=0, padx=30, pady=10) 

        button_show_palette_original = Button(c_graphics, text="Show Palette", command=lambda:self.show_palette(self.palette_original, self.centroids), font=font, bd=2, relief="solid")
        button_show_palette_original.grid(column=0, row=1, padx=10, pady=10, sticky="e")

        button_show_grafhic_color = Button(c_graphics, text="Show Graphic Color", command=lambda:self.show_grafhics_colors(self.palette_original), font=font, bd=2, relief="solid")
        button_show_grafhic_color.grid(column=1, row=1, padx=10, pady=10)

        button_show_clusters = Button(c_graphics, text="Show Clusters", command=lambda:self.show_clusters(int(k_value.get())), font=font, bd=2, relief="solid")
        button_show_clusters.grid(column=2, row=1, padx=10, pady=10, sticky="w", columnspan=2)

        # labels for images sizes
        self.size_original =   Label(c_graphics, text="Size: ", font=font, bd=2, relief="solid", width=20, anchor="w")
        self.size_original.grid(column=0, row=2, padx=(15, 100), pady=(0, 10))
        self.size_compressed = Label(c_graphics, text="Size: ", font=font, bd=2, relief="solid", width=20, anchor="w")
        self.size_compressed.grid(column=2, row=2, padx=(100, 15), pady=(0, 10),columnspan=2)

        # canvas for original and compressed image
        self.canvas_original = Canvas(self, width=500, height=500, bg="gray", bd=5, relief="ridge")
        self.canvas_original.grid(column=0, row=1, padx=10)
        self.canvas_compressed = Canvas(self, width=500, height=500, bg="gray", bd=5, relief="ridge")
        self.canvas_compressed.grid(column=1, row=1)

    #------------------------ File functions ---------------------
    def open_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not self.filename:
            messagebox.showinfo("Warning", "No image was selected.")
            return
        self.load_image_canvas(self.canvas_original, self.filename) #load imagen in canvas
        self.size_original.config(text="Size: " + str(os.path.getsize(self.filename)) + " bytes")  

    def save_as_compressed_jpg(self, image_array):
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        self.compressed_filename = f"{base_name}_compressed.jpg"  #create the file name

        img = Image.fromarray(image_array).convert('RGB')
        img.save(self.compressed_filename, 'JPEG', quality=85)  #save the image as a .jpg

    def load_image_canvas(self, canvas, filename):
        image = Image.open(filename)  # loads the image
        width, height = image.size    # get its dimensions
        scale = min(500 / width, 500 / height)  # calculate the scale to fit in the canvas

        # redimention the image keeping its proportions
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Get palette of the image
        self.palette_original = self.get_palette(image)

        # convert the image into a Photo copmatible with Tkinter
        tk_image = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(500//2, 500//2, image=tk_image, anchor="center")  # load into canvas
        canvas.image = tk_image  # prevents to erase the image reference 
    
    # ------------------------ K-Means functions ---------------------
    def initialize_centroids(self, image, K):
        """Selects K random pixels from the image to be the initial centroids"""
        num_pixels = image.shape[0]
        random_indices = np.random.choice(num_pixels, K, replace=False)
        centroids = image[random_indices]
        return centroids
    
    def assign_clusters(self, image, centroids):
        """Assigns each pixel in the image to the closest centroid"""
        distances = np.linalg.norm(image[:, np.newaxis] - centroids, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        return closest_centroids
    
    def update_centroids(self, image, closest_centroids, K):
        """Computes new centroids as the mean of assigned pixels"""
        new_centroids = np.array([image[closest_centroids == k].mean(axis=0) for k in range(K)])
        return new_centroids

    def k_means(self, image, K, max_iters=10, tolerance=1e-4):
        """Runs the K-Means clustering algorithm on an image"""
        centroids = self.initialize_centroids(image, K)
        
        for _ in range(max_iters):
            closest_centroids = self.assign_clusters(image, centroids)
            new_centroids = self.update_centroids(image, closest_centroids, K)
            if np.linalg.norm(new_centroids - centroids) < tolerance:
                break
            centroids = new_centroids

        return centroids, closest_centroids
    
    #------------------------ Image reconstruction ---------------------
    def reconstruct_image(self, closest_centroids, centroids, original_shape):
        """Replaces each pixel in the image with its assigned centroid color"""
        compressed_image = centroids[closest_centroids]
        compressed_image = np.reshape(compressed_image, original_shape)
        return np.clip(compressed_image.astype(np.uint8), 0, 255)
    
    #------------------------ Image compression ---------------------
    def compress_image(self, K):
        """ Loads an image, applies K-Means compression, and displays results"""
        image = cv2.imread(self.filename)  #loads the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert BGR to RGB
        original_shape = image.shape
        image_flattened = image.reshape(-1, 3)  # convert to a list of pixels [R,G,B]
        
        # build the image with the located centroids
        self.centroids, closest_centroids = self.k_means(image_flattened, K)

        compressed_image = self.reconstruct_image(closest_centroids, self.centroids, original_shape)
        
        # save the image and load it in the IU
        self.save_as_compressed_jpg(compressed_image)
        self.load_image_canvas(self.canvas_compressed, self.compressed_filename)
        self.size_compressed.config(text="Tamaño: " + str(os.path.getsize(self.compressed_filename)) + " bytes")  

    
    #------------------------ Palette generation ---------------------
    def get_palette(self, image = None):
        image = cv2.imread(self.filename)  #loads the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert BGR to RGB
        image_flattened = image.reshape(-1, 3)  # convert to a list of pixels [R,G,B]

        unique_colors = {tuple(color) for color in image_flattened}
        color_palette = np.array(list(unique_colors), dtype=np.uint8)

        return color_palette

    #------------------------ Show Grafhics functions ---------------------
    """def show_palette(self, palette_original, palete_compressed):
        fig = plt.figure(figsize=(10, 2))

        plt.imshow([palette_original / 255])
        plt.title(f"PALETA DE COLORES \nNumero de colores: {palette_original.shape[0]} \n")
        plt.axis("off")  # Ocultar ejes para enfocarse en los colores

        plt.imshow([palete_compressed / 255])
        plt.title(f"PALETA DE COLORES \nNumero de colores: {palete_compressed.shape[0]} \n")
        plt.axis("off")  # Ocultar ejes para enfocarse en los colores
        plt.show()"""

    def show_palette(self, palette_original, palette_compressed):
        fig, axes = plt.subplots(1, 2, figsize=(10, 2))  # Crear dos subgráficos en una fila

        # Mostrar la paleta original
        axes[0].imshow([palette_original / 255])
        axes[0].set_title(f"PALETA DE COLORES ORIGINAL \nNúmero de colores: {palette_original.shape[0]} \n")
        axes[0].axis("off")  # Ocultar ejes

        # Mostrar la paleta comprimida
        axes[1].imshow([palette_compressed / 255])
        axes[1].set_title(f"PALETA DE COLORES K-MEANS \nNúmero de colores: {palette_compressed.shape[0]} \n")
        axes[1].axis("off")  # Ocultar ejes

        plt.show()
    
    def show_clusters(self, k):
        centroids, closest_centroids = self.k_means(self.palette_original, k)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(k):
            ax.scatter(
                self.palette_original[closest_centroids == i, 0],
                self.palette_original[closest_centroids == i, 1],
                self.palette_original[closest_centroids == i, 2], 
                s=30, label=f'Cluster {i}'
            )

            ax.scatter(
                centroids[:, 0], 
                centroids[:, 1], 
                centroids[:, 2],
                c='red', marker='x', s=100, label=f'Centroide  {i}'
            )

        ax.set_xlabel('Rojo (Red)')
        ax.set_ylabel('Verde (Green)')
        ax.set_zlabel('Azul (Blue)')
        ax.legend()
        plt.title("Clusters generados por K-Means")
        plt.show()
    
    def show_grafhics_colors(self, palette):
        reds = []
        greens = []
        blues = []

        for color in palette:
            reds.append(color[0])
            greens.append(color[1])
            blues.append(color[2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(reds, greens, blues, c=palette/255, s=100)
        ax.set_xlabel('Rojo (R)')
        ax.set_ylabel('Verde (G)')
        ax.set_zlabel('Azul (B)')
        plt.title("Distribución de colores en espacio RGB")
        plt.show()
    

if __name__ == '__main__':
    main = Tk()
    app = ImageCompressor(main)
    app.mainloop()