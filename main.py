import os
import cv2
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import MiniBatchKMeans
from skimage import io, img_as_float, img_as_uint, img_as_ubyte
import numpy as np
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

clusters = 8
iterations = 100
batch_size = 100

print('K-Means Clustering')
print(f'Number of clusters: {clusters}')
print(f'Iterations: {iterations}')
print(f'Batch Size: {batch_size}')

reshaped_image = lab_image.reshape(height * width, depth)

k_means = MiniBatchKMeans(
    n_clusters=clusters,
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
quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)

cv2.imwrite(f'temp/quantized_image_k={clusters}.jpg', quantized_image)
cv2.imshow('Image', image)
cv2.imshow(f'Quantized Image (k={clusters})', quantized_image)

# 3ο ερώτημα
# Κατάτμηση Εικόνας σε Superpixels σύμφωνα με τον αλγόριθμο SLIC

# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
# https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/

segments = 100
sigma = 5

float_image = img_as_float(image)

superpixels = slic(
    float_image,
    n_segments=segments,
    sigma=sigma
)

slic_image = mark_boundaries(float_image, superpixels)
slic_image = img_as_ubyte(slic_image)

io.imsave(f'temp/slic_image_segments={segments}_sigma={sigma}.jpg', slic_image)
cv2.imshow(f'SLIC Image (segments={segments}, sigma={sigma})', slic_image)
cv2.waitKey(0)
