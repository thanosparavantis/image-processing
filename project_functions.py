import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import MiniBatchKMeans

# https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means
# https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
# https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
# https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/
# https://www.pyimagesearch.com/2014/12/29/accessing-individual-superpixel-segmentations-python/
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
# https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
# https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

shutil.rmtree('./temp')
os.mkdir('./temp')


def get_lab_dataset(path):
    # Find all image paths in the dataset folder
    paths = []
    filenames = sorted(os.listdir(path))
    for filename in filenames:
        image_path = os.path.join(path, filename)
        print(f'Found image: {image_path}')
        paths.append(image_path)

    # Read images and convert them to LAB color space
    dataset = [cv2.imread(path) for path in paths]
    dataset = [cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for image in dataset]

    # For each image store L, a, b channels
    if not os.path.isdir('./temp/lab'):
        os.mkdir('./temp/lab')

    for idx, image in enumerate(dataset):
        L, a, b = cv2.split(image)
        cv2.imwrite(f'./temp/lab/{idx + 1}_L.jpg', L)
        cv2.imwrite(f'./temp/lab/{idx + 1}_a.jpg', a)
        cv2.imwrite(f'./temp/lab/{idx + 1}_b.jpg', b)

    return dataset


def quantize_images(lab_dataset, colors=32):
    # Reshape color images from (height, width, depth) -> (height*width, depth)
    # so we can provide them as input for k-means
    reshaped_dataset = []

    for image in lab_dataset:
        height, width, depth = image.shape
        reshaped = image.reshape((height * width, depth))
        reshaped_dataset.append(reshaped)

    # Initialize mini batch k-means with a specific number of clusters
    kmeans = MiniBatchKMeans(n_clusters=colors)

    # Run k-means for each LAB image in our dataset
    for idx, image in enumerate(reshaped_dataset):
        print(f'K-Means fitting on image #{idx + 1}')
        kmeans.fit(image)

    # Find computed cluster coordinates and convert them to integers
    clusters = kmeans.cluster_centers_.astype('uint8')

    quantized_dataset = []

    for idx, image in enumerate(reshaped_dataset):
        print(f'Quantizing image #{idx + 1}')

        # Assign a quantized color label for every pixel of the image
        pixel_labels = kmeans.predict(image)

        # Compute the quantized image and reshape to it's original dimensions
        quantized = clusters[pixel_labels]
        height, width, depth = lab_dataset[idx].shape
        quantized = quantized.reshape((height, width, depth))

        quantized_dataset.append(quantized)

    # Save quantized images
    if not os.path.isdir('./temp/quantized'):
        os.mkdir('./temp/quantized')

    for idx, image in enumerate(quantized_dataset):
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(f'./temp/quantized/{idx + 1}.jpg', image)

    return quantized_dataset


def slic_images(quantized_dataset):
    slic_dataset = []
    slic_groups = []

    for idx, image in enumerate(quantized_dataset):
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        print(f'Running SLIC on image #{idx + 1}')

        # Find the boundaries for each superpixel in the image
        groups = slic(image=img_as_float(image), n_segments=50)

        # Loop through all superpixels and extract them using a mask
        superpixels = []

        for group in np.unique(groups):
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask[groups == group] = 255

            # Create an image that only contains the superpixel
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            superpixels.append(masked_image)

        slic_groups.append(superpixels)

        # Create an image that contains all superpixel boundaries
        marked_image = mark_boundaries(image, groups)
        marked_image = img_as_ubyte(marked_image)

        slic_dataset.append(marked_image)

    # Save images with superpixel boundaries
    if not os.path.isdir('./temp/slic'):
        os.mkdir('./temp/slic')

    for idx, image in enumerate(slic_dataset):
        cv2.imwrite(f'./temp/slic/{idx + 1}.jpg', image)

    # Save every superpixel individually grouped by image
    if not os.path.isdir('./temp/superpixels'):
        os.mkdir('./temp/superpixels')

    for idx1, group in enumerate(slic_groups):
        if not os.path.isdir(f'./temp/superpixels/{idx1 + 1}'):
            os.mkdir(f'./temp/superpixels/{idx1 + 1}')

        for idx2, superpixel in enumerate(group):
            cv2.imwrite(f'./temp/superpixels/{idx1 + 1}/{idx2 + 1}.jpg', superpixel)

    return slic_dataset, slic_groups


def compute_surf(slic_groups):
    surf_groups = []
    surf = cv2.xfeatures2d.SURF_create()

    for idx, group in enumerate(slic_groups):
        print(f'Computing SURF features for image #{idx + 1}')
        surf_images = []

        for superpixel in group:
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)
            image = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)
            surf_images.append(image)

        surf_groups.append(surf_images)

    if not os.path.isdir('./temp/surf'):
        os.mkdir('./temp/surf')

    for idx1, group in enumerate(surf_groups):
        if not os.path.isdir(f'./temp/surf/{idx1 + 1}'):
            os.mkdir(f'./temp/surf/{idx1 + 1}')

        for idx2, image in enumerate(group):
            cv2.imwrite(f'./temp/surf/{idx1 + 1}/{idx2 + 1}.jpg', image)


def build_gabor_kernels():
    filters = []

    ksize = (31, 31)
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    psi = 0.0

    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        print(f'New gabor filter ksize={ksize}, sigma={sigma}, theta={theta}, lambd={lambd}, gamma={gamma}, psi={psi}')
        filters.append(kernel)

    return filters


def apply_gabor_kernels(kernels, image):
    response = np.zeros_like(image)

    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        np.maximum(response, filtered, response)

    return response


def compute_gabor(slic_groups):
    gabor_groups = []
    kernels = build_gabor_kernels()

    for idx1, superpixels in enumerate(slic_groups):
        print(f'Computing Gabor features for image #{idx1 + 1}')
        gabor_images = []

        for idx2, superpixel in enumerate(superpixels):
            image = apply_gabor_kernels(kernels, superpixel)
            gabor_images.append(image)

        gabor_groups.append(gabor_images)

    if not os.path.isdir('./temp/gabor'):
        os.mkdir('./temp/gabor')

    for idx1, group in enumerate(gabor_groups):
        if not os.path.isdir(f'./temp/gabor/{idx1 + 1}'):
            os.mkdir(f'./temp/gabor/{idx1 + 1}')

        for idx2, image in enumerate(group):
            cv2.imwrite(f'./temp/gabor/{idx1 + 1}/{idx2 + 1}.jpg', image)


def plot_images(path, dataset, rows, cols):
    rgb_dataset = [cv2.cvtColor(image, cv2.COLOR_LAB2RGB) for image in dataset]

    fig = plt.figure(figsize=(100, 100))

    for idx, image in enumerate(rgb_dataset, start=1):
        axis = fig.add_subplot(rows, cols, idx)
        axis.set_axis_off()
        plt.imshow(image)

    plt.savefig(path)
    plt.show()
