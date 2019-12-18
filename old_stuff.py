import project_functions as pfuncs
import os
import cv2
import numpy as np
import skimage
from skimage import img_as_float, img_as_ubyte
from skimage.filters import gabor
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import MiniBatchKMeans

if not os.path.isdir('temp'):
    os.mkdir('temp')

image = cv2.imread('image.jpg')
height, width, depth = image.shape

# 1ο ερώτημα
# Αναπαράσταση Εικόνας στον Χρωματικό Χώρο Lab

print('Converting RGB to LAB image...')
lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# 2ο ερώτημα
# Διακριτοποίηση του Χρωματικού Χώρου Lab με βάση ένα σύνολο συναφών εικόνων εκπαίδευσης

# https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means
# https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/

clusters = (8, 16, 32)
iterations = 100
batch_size = 100

print('Color Quantization')

for cluster_count in clusters:
    print('Applying K-Means Clustering...')
    print(f'Number of clusters: {cluster_count}')
    print(f'Iterations: {iterations}')
    print(f'Batch Size: {batch_size}')

    reshaped_image = lab_image.reshape(height * width, depth)

    k_means = MiniBatchKMeans(
        n_clusters=cluster_count,
        max_iter=iterations,
        batch_size=batch_size)

    print('Computing centroids...')

    pixels_labels = k_means.fit_predict(reshaped_image)

    centroids = k_means.cluster_centers_.astype('uint8')

    print('Centroids:')
    print(centroids)

    print('Constructing quantized image...')
    quantized_image = centroids[pixels_labels]
    quantized_image = quantized_image.reshape((height, width, depth))
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

    path = f'temp/quantized_image_k={cluster_count}.jpg'
    print(f'Saving image: {path}')
    cv2.imwrite(path, quantized_image)

# 3ο ερώτημα
# Κατάτμηση Εικόνας σε Superpixels σύμφωνα με τον αλγόριθμο SLIC

# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
# https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/

segments = (100, 200, 300)
sigma = 5

print('Extracting superpixels')

float_image = img_as_float(image)
superpixel_groups = []

for segment_count in segments:
    print(f'Applying SLIC algorithm')
    print(f'Segments: {segment_count}')
    print(f'Sigma: {sigma}')

    superpixels = slic(
        float_image,
        n_segments=segment_count,
        sigma=sigma
    )

    superpixel_groups.append(superpixels)

    print('Marking boundaries...')
    slic_image = mark_boundaries(float_image, superpixels)
    slic_image = img_as_ubyte(slic_image)

    path = f'temp/slic_image_segments={segment_count}_sigma={sigma}.jpg'
    print(f'Saving image: {path}')

    cv2.imwrite(path, slic_image)

# 4ο ερώτημα
# Εξαγωγή Χαρακτηριστικών Υφής (SURF Features & Gabor Features) ανά Super Pixel

# https://www.pyimagesearch.com/2014/12/29/accessing-individual-superpixel-segmentations-python/

superpixels = superpixel_groups[0]

# kernel = cv2.getGaborKernel(ksize=,
#                             sigma=,
#                             theta=,
#                             lambd=,
#                             gamma=,
#                             psi=,
#                             ktype=cv2.CV_32F)

for superpixel in np.unique(superpixels):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask[superpixels == superpixel] = 255

    image_part = cv2.bitwise_and(image, image, mask=mask)
    real_response, imaginary_response = gabor(image_part[:, :, 0], frequency=0.6)

    cv2.imshow('Real response', real_response)
    cv2.imshow('Imaginary response', imaginary_response)
    cv2.waitKey()
