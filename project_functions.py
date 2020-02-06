import math
import os
import shutil
import sys

import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn import svm, metrics, preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

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


def quantize_images(lab_dataset):
    # Reshape color images from (height, width, depth) -> (height*width, depth)
    # so we can provide them as input for k-means
    reshaped_dataset = []

    for image in lab_dataset:
        height, width, depth = image.shape
        reshaped = image.reshape((height * width, depth))
        reshaped_dataset.append(reshaped)

    # Initialize mini batch k-means with a specific number of clusters
    kmeans = MiniBatchKMeans(n_clusters=32)

    # Run k-means for each LAB image in our dataset
    for idx, image in enumerate(reshaped_dataset):
        print(f'K-Means fitting on image #{idx + 1}')
        kmeans.fit(image)

    # Find computed cluster coordinates and convert them to integers
    centroids = kmeans.cluster_centers_.astype('uint8')

    quantized_dataset = []

    for idx, image in enumerate(reshaped_dataset):
        print(f'Quantizing image #{idx + 1}')

        # Assign a quantized color label for every pixel of the image
        pixel_labels = kmeans.predict(image)

        # Compute the quantized image and reshape to it's original dimensions
        quantized = centroids[pixel_labels]
        height, width, depth = lab_dataset[idx].shape
        quantized = quantized.reshape((height, width, depth))

        quantized_dataset.append(quantized)

    # Save quantized images
    if not os.path.isdir('./temp/quantized'):
        os.mkdir('./temp/quantized')

    for idx, image in enumerate(quantized_dataset):
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(f'./temp/quantized/{idx + 1}.jpg', image)

    return quantized_dataset, centroids


def slic_images(quantized_dataset):
    slic_dataset = []
    slic_groups = []
    slic_centroids = []

    for idx, image in enumerate(quantized_dataset):
        # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        print(f'Running SLIC on image #{idx + 1}')

        # Find the boundaries for each superpixel in the image
        groups = slic(image=img_as_float(image), n_segments=50, compactness=10, sigma=1)
        group_ids = np.unique(groups)

        slic_centroids.append(np.array([np.mean(np.nonzero(groups == i), axis=1) for i in group_ids]))

        # Loop through all superpixels and extract them using a mask
        superpixels = []

        for group in group_ids:
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
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(f'./temp/slic/{idx + 1}.jpg', image)

    # Save every superpixel individually grouped by image
    if not os.path.isdir('./temp/superpixels'):
        os.mkdir('./temp/superpixels')

    for idx1, group in enumerate(slic_groups):
        if not os.path.isdir(f'./temp/superpixels/{idx1 + 1}'):
            os.mkdir(f'./temp/superpixels/{idx1 + 1}')

        for idx2, superpixel in enumerate(group):
            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_LAB2BGR)
            cv2.imwrite(f'./temp/superpixels/{idx1 + 1}/{idx2 + 1}.jpg', superpixel)

    return slic_groups, slic_centroids


def compute_surf(slic_groups):
    surf_descriptors = []
    surf_groups = []
    surf = cv2.xfeatures2d.SURF_create()

    for idx, group in enumerate(slic_groups):
        print(f'Computing SURF features for image #{idx + 1}')
        image_descriptors = []
        surf_images = []

        for idx, superpixel in enumerate(group):
            print(f'Processing superpixel #{idx}')

            keypoints, descriptors = surf.detectAndCompute(superpixel, None)
            image = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)

            image_descriptors.append(descriptors)
            surf_images.append(image)

        surf_descriptors.append(image_descriptors)
        surf_groups.append(surf_images)

    if not os.path.isdir('./temp/surf'):
        os.mkdir('./temp/surf')

    for idx1, group in enumerate(surf_groups):
        if not os.path.isdir(f'./temp/surf/{idx1 + 1}'):
            os.mkdir(f'./temp/surf/{idx1 + 1}')

        for idx2, image in enumerate(group):
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
            cv2.imwrite(f'./temp/surf/{idx1 + 1}/{idx2 + 1}.jpg', image)

    return surf_descriptors


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

    for idx1, group in enumerate(slic_groups):
        print(f'Computing Gabor features for image #{idx1 + 1}')
        gabor_images = []

        for idx2, superpixel in enumerate(group):
            print(f'Processing superpixel #{idx2}')
            image = apply_gabor_kernels(kernels, superpixel)
            gabor_images.append(image)

        gabor_groups.append(gabor_images)

    if not os.path.isdir('./temp/gabor'):
        os.mkdir('./temp/gabor')

    for idx1, group in enumerate(gabor_groups):
        if not os.path.isdir(f'./temp/gabor/{idx1 + 1}'):
            os.mkdir(f'./temp/gabor/{idx1 + 1}')

        for idx2, image in enumerate(group):
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
            cv2.imwrite(f'./temp/gabor/{idx1 + 1}/{idx2 + 1}.jpg', image)

    return gabor_groups


def make_dataset(slic_groups, slic_centroids, surf_descriptors, gabor_groups, centroids):
    # Store all discrete AB colors
    ab_centroids = centroids[:, 1:]

    color_lookup = {}

    for idx, color in enumerate(ab_centroids):
        color_lookup[color[0], color[1]] = idx
        print(f'Color (a:{color[0]}, b:{color[1]}) with index {idx}')

    # Store the AB colors for all pixels in superpixels
    images_with_ab = []

    for group in slic_groups:
        image = []
        for superpixel in group:
            image.append(superpixel[:, :, 1:])
        images_with_ab.append(image)

    # Store the centroid AB color for each superpixel
    images_with_centroid_ab = []

    for group in slic_centroids:
        image = []
        for centroid in group:
            c_a = centroid[0]
            c_b = centroid[1]

            min_dist = sys.maxsize
            color = ab_centroids[0]

            for quantized_color in ab_centroids:
                a = quantized_color[0]
                b = quantized_color[1]
                distance = math.sqrt(((c_a - a) ** 2) + ((c_b - b) ** 2))

                if distance < min_dist:
                    min_dist = distance
                    color = quantized_color

            image.append(color)
        images_with_centroid_ab.append(image)

    # Store the SURF descriptors average for each superpixel
    images_with_surf_avg = []

    for group in surf_descriptors:
        image = []
        for descriptor in group:
            nonzero = descriptor[descriptor != 0]
            average = np.mean(nonzero)
            image.append(average)
        images_with_surf_avg.append(image)

    # Store the Gabor filter responses for each superpixel
    images_with_gabor_avg = []

    for group in gabor_groups:
        image = []
        for response in group:
            nonzero = response[response != 0]
            average = np.mean(nonzero)
            image.append(average)
        images_with_gabor_avg.append(image)

    # Create dataset for SVM input and labels
    X = []
    y = []

    for i in range(len(slic_groups)):
        for j in range(len(images_with_surf_avg[i])):
            surf_feature = images_with_surf_avg[i][j]
            gabor_feature = images_with_gabor_avg[i][j]
            color = images_with_centroid_ab[i][j]

            X.append([surf_feature, gabor_feature])
            y.append(color_lookup[color[0], color[1]])

    X = preprocessing.scale(X)

    return X, y, ab_centroids


def train_svm(X, y):
    print('Training SVM!')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')

    return classifier


def test_get_image(path):
    test_image = cv2.imread(path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    return test_image


def test_slic_image(test_image):
    print('Running SLIC on test image')

    groups = slic(image=img_as_float(test_image), n_segments=50, compactness=0.1, sigma=1)
    group_ids = np.unique(groups)

    test_slic_centroids = np.array([np.mean(np.nonzero(groups == i), axis=1) for i in group_ids])

    test_superpixels = []

    for group in group_ids:
        mask = np.zeros(test_image.shape[:2], dtype="uint8")
        mask[groups == group] = 255

        masked_image = cv2.bitwise_and(test_image, test_image, mask=mask)
        test_superpixels.append(masked_image)

    marked_image = mark_boundaries(test_image, groups)
    marked_image = img_as_ubyte(marked_image)

    if not os.path.isdir('./temp/slic_test'):
        os.mkdir('./temp/slic_test')

    cv2.imwrite(f'./temp/slic_test/test.jpg', marked_image)

    if not os.path.isdir('./temp/superpixels_test'):
        os.mkdir('./temp/superpixels_test')

    for idx, superpixel in enumerate(test_superpixels):
        cv2.imwrite(f'./temp/superpixels_test/{idx + 1}.jpg', superpixel)

    return test_superpixels, test_slic_centroids


def test_compute_surf(test_superpixels):
    print('Computing SURF features for test image')

    test_surf_descriptors = []
    surf_groups = []
    surf = cv2.xfeatures2d.SURF_create()

    for superpixel in test_superpixels:
        keypoints, descriptors = surf.detectAndCompute(superpixel, None)
        image = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)

        test_surf_descriptors.append(descriptors)
        surf_groups.append(image)

    if not os.path.isdir('./temp/surf_test'):
        os.mkdir('./temp/surf_test')

    for idx, image in enumerate(surf_groups):
        cv2.imwrite(f'./temp/surf_test/{idx + 1}.jpg', image)

    return test_surf_descriptors


def test_compute_gabor(test_superpixels):
    print('Computing Gabor features for test image')
    test_gabor_groups = []
    kernels = build_gabor_kernels()

    for superpixel in test_superpixels:
        image = apply_gabor_kernels(kernels, superpixel)
        test_gabor_groups.append(image)

    if not os.path.isdir('./temp/gabor_test'):
        os.mkdir('./temp/gabor_test')

    for idx, image in enumerate(test_gabor_groups):
        cv2.imwrite(f'./temp/gabor_test/{idx + 1}.jpg', image)

    return test_gabor_groups


def test_make_dataset(test_surf_descriptors, test_gabor_groups):
    surf_avg = []

    for idx, descriptor in enumerate(test_surf_descriptors):
        nonzero = descriptor[descriptor != 0]
        average = np.mean(nonzero)
        surf_avg.append(average)

    gabor_avg = []

    for idx, response in enumerate(test_gabor_groups):
        nonzero = response[response != 0]
        average = np.mean(nonzero)
        gabor_avg.append(average)

    test_X = []

    for i in range(len(surf_avg)):
        surf_feature = surf_avg[i]
        gabor_feature = gabor_avg[i]
        test_X.append([surf_feature, gabor_feature])

    test_X = preprocessing.scale(test_X)

    return test_X


def test_color_image(test_image, ab_centroids, classifier, test_X, test_superpixels):
    labels = classifier.predict(test_X)
    colors = ab_centroids[labels]

    colored_image = np.zeros((test_image.shape[0], test_image.shape[1], 3), dtype='uint8')

    for idx, superpixel in enumerate(test_superpixels):
        for i in range(superpixel.shape[0]):
            for j in range(superpixel.shape[1]):
                if superpixel[i, j] > 0:
                    L = test_image[i, j]
                    a = colors[idx, 0]
                    b = colors[idx, 1]

                    print(f'Coloring pixel ({i}, {j}) with LAB color (L:{L}, a:{a}, b:{b})')

                    colored_image[i, j, 0] = L
                    colored_image[i, j, 1] = a
                    colored_image[i, j, 2] = b

    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(f'./temp/test_colored.jpg', colored_image)

    print('Done')
