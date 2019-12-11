import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

image = cv2.imread('image.jpg')
height, width, depth = image.shape

# 1ο ερώτημα εργασίας
# Αναπαράσταση Εικόνας στον Χρωματικό Χώρο Lab

print('Converting RGB to LAB image...')
lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# 2ο ερώτημα εργασίας
# Διακριτοποίηση του Χρωματικού Χώρου Lab με βάση ένα σύνολο συναφών εικόνων εκπαίδευσης

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

centroids = k_means.cluster_centers_.astype("uint8")

print('Centroids:')
print(centroids)

print('Constructing quantized image...')
quantized_image = centroids[pixels_labels]
quantized_image = quantized_image.reshape((height, width, depth))
quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)

cv2.imshow("Image", image)
cv2.imshow("Quantized Image", quantized_image)
cv2.waitKey(0)
