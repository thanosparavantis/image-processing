import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import MiniBatchKMeans

def get_lab_dataset(path):
    # Store all paths of images from the directory
    paths = []
    for file in os.listdir(path):
        paths.append(os.path.join(path, file))

    # Read images using cv2
    dataset = [cv2.imread(path) for path in paths]

    # Convert the images to lab color space
    dataset = [cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for image in dataset]

    return dataset


def quantize_images(lab_dataset, colors=32):
    reshaped_dataset = []
    for image in lab_dataset:
        height, width, depth = image.shape
        reshaped = image.reshape((height * width, depth))
        reshaped_dataset.append(reshaped)

    kmeans = MiniBatchKMeans(n_clusters=colors)

    for image in reshaped_dataset:
        kmeans.fit(image)

    clusters = kmeans.cluster_centers_.astype('uint8')

    quantized_dataset = []
    for idx, image in enumerate(reshaped_dataset):
        pixel_labels = kmeans.predict(image)
        quantized = clusters[pixel_labels]
        height, width, depth = lab_dataset[idx].shape
        quantized = quantized.reshape((height, width, depth))
        quantized_dataset.append(quantized)

    return quantized_dataset


def slic_images(quantized_dataset, segments=100):
    slic_dataset = []

    for image in quantized_dataset:
        float_image = img_as_float(image)
        groups = slic(image=float_image, n_segments=segments)
        marked_image = mark_boundaries(image, groups)
        marked_image = img_as_ubyte(marked_image)
        slic_dataset.append(marked_image)

    return slic_dataset


def plot_images(path, dataset, rows=5, cols=2):
    rgb_dataset = [cv2.cvtColor(image, cv2.COLOR_LAB2RGB) for image in dataset]

    fig = plt.figure(figsize=(100, 100))

    for idx, image in enumerate(rgb_dataset, start=1):
        axis = fig.add_subplot(rows, cols, idx)
        axis.set_axis_off()
        plt.imshow(image)

    plt.savefig(path)
    plt.show()
