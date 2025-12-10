import skimage
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter

def toonify(contentPath):

    image = skimage.io.imread(contentPath)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    image = uniform_filter(image, size=(7, 7, 1)) 
    image = skimage.restoration.denoise_bilateral(image, channel_axis=-1)
    # aplicando filtro bilateral

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # agora é necessário aplicar um k-means
    # o objetivo é reduzir a paleta de cores
    pixels = image.reshape(-1, 3)

    # O melhor K vai ser o q maximiza o score
    real_k = 8

    # Aplicar K-means com o melhor k encontrado
    kmeans = KMeans(n_clusters=real_k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Obter os labels e os centroides dos clusters
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Reconstruir a imagem com a paleta reduzida
    image_reduced = centers[labels].reshape(image.shape)
    plt.imshow(image_reduced)
    plt.axis('off')
    plt.show()

    edges = skimage.feature.canny(skimage.color.rgb2gray(image), sigma=2)
    edges = skimage.morphology.dilation(edges)
    edgeMask = np.invert(edges) # Aplicando o aguçamento na imagem
    image = image_reduced * edgeMask[:,:,None]
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image_reduced

toonify("golfinho.jpg")