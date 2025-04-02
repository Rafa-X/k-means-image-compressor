import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from skimage import io
import os

os.environ['OMP_NUM_THREADS'] = '4'

def get_data_from_image(image_path):
    image = io.imread(image_path)
    print(image.shape)
    rows, cols, _ = image.shape
    image = image.reshape(-1, 3)

    unique_colors = {tuple(color) for color in image}
    palette = list(unique_colors)

    return np.array(palette)

# Show the data in 3D space
# and the clusters in 2D space
def show_data(palette):
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

def show_clusters(palette, labels, centroids, k):
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(k):
        ax.scatter(
            palette[labels == i, 0],
            palette[labels == i, 1],
            palette[labels == i, 2], 
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
    plt.show()

def show_palette(palette, centroids):
    fig, ax = plt.subplots(figsize=(15, k * 2))

    for i in range(k):
        cluster_points = palette[labels == i]
        
        centroid_color = centroids[i] / 255
        ax.add_patch(plt.Rectangle(
            (len(cluster_points), i), 10, 1, color=centroid_color
        ))
        ax.text(
            len(cluster_points) + 0.5, i + 0.5, f'Centroide {i}',
            color='white', ha='center', va='center', fontsize=10
        )

        for idx, color in enumerate(cluster_points):
            normalized_color = color / 255
            ax.add_patch(plt.Rectangle(
                (idx, i), 1, 1, color=normalized_color
            ))
        
    ax.set_xlim(0, palette.shape[0])
    ax.set_ylim(-0.5, k - 0.5)
    ax.set_xticks([])
    ax.set_yticks(range(k))
    ax.set_yticklabels([f'Cluster {i}' for i in range(k)])
    ax.set_aspect('auto')
    plt.title('Paleta de colores agrupados por cluster y centroides')
    plt.show()

# K-Means Implementation
def initialize_centroids(X, k):
    np.random.seed(42)
    random_indices = np.random.permutation(X.shape[0])[:k]
    return X[random_indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


# Main code

palette = get_data_from_image("Cherry_Leaves.png")
k = 2
centroids, labels = kmeans(palette, k)

show_data(palette)
show_clusters(palette, labels, centroids, k)
show_palette(palette, centroids)
