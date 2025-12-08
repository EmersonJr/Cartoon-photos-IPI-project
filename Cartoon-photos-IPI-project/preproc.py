import skimage
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

image = skimage.io.imread("") # n sei qual o melhor jeito de fazer isso aqui ainda
image = skimage.color.rgb2gray(image)
# usando um Kernel 20x20
footprint = skimage.morphology.disk(20)
image = skimage.filters.rank.mean(image, footprint)
# filtro de media
image = skimage.filters.median(image, footprint)
# aplicando um filtro de mediana tambem com o mesmo kernel
image = skimage.filters.gaussian(image)
# aplicando filtro gaussiano
image = skimage.restoration.denoise_bilateral(image)
# aplicando filtro bilateral

# agora é necessário aplicar um k-means
# o objetivo é reduzir a paleta de cores

pixels = image.reshape(-1, 1)

k_range = range(2, 11)  # testa de 2 a 10 clusters
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    inertias.append(kmeans.inertia_)

    if len(pixels) > 10000: 
        sample_idx = np.random.choice(len(pixels), 10000, replace=False)
        score = silhouette_score(pixels[sample_idx], kmeans.predict(pixels[sample_idx]))
    else:
        score = silhouette_score(pixels, kmeans.labels_)
    silhouette_scores.append(score)
    # testando diferentes valores de K para encontrar o mais adequado

# O melhor K vai ser o q maximiza o score
real_k = k_range[np.argmax(silhouette_scores)]

# Aplicar K-means com o melhor k encontrado
kmeans = KMeans(n_clusters=real_k, random_state=42, n_init=10)
kmeans.fit(pixels)

# Obter os labels e os centroides dos clusters
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Reconstruir a imagem com a paleta reduzida
image_reduced = centers[labels].reshape(image.shape)
