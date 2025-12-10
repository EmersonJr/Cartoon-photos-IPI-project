import skimage
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature, color, util
from skimage.filters.rank import median
from skimage.measure import label, regionprops
from skimage.morphology import thin, dilation
from skimage.color import rgb2gray
import cv2

KERNEL = np.ones((7,7), dtype=bool)
MATRIZ_ESTRU = np.array([[1,1],
                         [1,1]], dtype=bool)


def treatColors(image):

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    aux = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    d = 2 * 3 + 1

    for _ in range(14):
        L, A, B = cv2.split(aux)
        L = cv2.bilateralFilter(L, d, 8, 3)
        A = cv2.bilateralFilter(A, d, 3, 3)
        B = cv2.bilateralFilter(B, d, 3, 3)
        aux = cv2.merge([L, A, B])

    image = cv2.cvtColor(aux, cv2.COLOR_LAB2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    imageFilt = np.zeros_like(image)
    for i in range(3):
        imageFilt[:,:,i] = median(image[:,:,i], footprint=KERNEL)
    image = imageFilt
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    a = 12
    image = (image // a) * a
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    imageFilt = np.zeros_like(image)
    for i in range(3):
        imageFilt[:,:,i] = median(image[:,:,i], footprint=KERNEL)
    image = imageFilt
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    return image
def treatEdges(image):

    edges = feature.canny(rgb2gray(image), sigma=2)

    labeld = label(edges)
    result = np.zeros_like(edges)
    for r in regionprops(labeld):
        if r.area >= 10:
            result[labeld == r.label] = True

    return dilation(thin(result), footprint=MATRIZ_ESTRU)

def toonifyReal(imagePath):

    image = skimage.io.imread(imagePath)

    edges = treatEdges(image)
    image = treatColors(image)

    image[edges] = [0, 0, 0]

    plt.imshow(image)
    plt.axis('off')
    plt.show()


toonifyReal("Lenna.png")