import os
import shutil

import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing, svm, metrics
from sklearn.cluster import MiniBatchKMeans


class ImageProcessor:
    TempFolderPath = './temp'
    SourceFilePath = './image_2.jpg'
    TargetFilePath = './image_3.jpg'
    SourceSuperpixelFilePath = f'{TempFolderPath}/source_superpixels'
    TargetSuperpixelFilePath = f'{TempFolderPath}/target_superpixels'
    SourceSurfFilePath = f'{TempFolderPath}/source_surf'
    TargetSurfFilePath = f'{TempFolderPath}/target_surf'
    GaborKernelsFilePath = f'{TempFolderPath}/gabor_kernels'
    SourceGaborFilePath = f'{TempFolderPath}/source_gabor'
    TargetGaborFilePath = f'{TempFolderPath}/target_gabor'

    def __init__(self):
        shutil.rmtree(self.TempFolderPath)
        os.mkdir(self.TempFolderPath)
        os.mkdir(self.SourceSuperpixelFilePath)
        os.mkdir(self.TargetSuperpixelFilePath)
        os.mkdir(self.SourceSurfFilePath)
        os.mkdir(self.TargetSurfFilePath)
        os.mkdir(self.GaborKernelsFilePath)
        os.mkdir(self.SourceGaborFilePath)
        os.mkdir(self.TargetGaborFilePath)

        self.colors_lab = []
        self.k_means = None
        self.source_q = None
        self.source_superpixels = []
        self.source_centroids = []
        self.target_superpixels = []
        self.source_surf = []
        self.target_surf = []
        self.source_gabor = []
        self.target_gabor = []
        self.gabor_kernels = []
        self.colors_ab = []
        self.colors_ab_idx = {}
        self.source_x = []
        self.source_y = []
        self.target_x = []
        self.target_y = []
        self.classifier = None

    def change_colorspace(self):
        print('Changing colorspace for images...')

        source_img = cv2.imread(self.SourceFilePath)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(source_img)

        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_L.jpg'), L)
        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_a.jpg'), a)
        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_b.jpg'), b)

        target_img = cv2.imread(self.TargetFilePath)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(self.TempFolderPath, 'target_greyscale.jpg'), target_img)

        print('Done.')

    def quantize_source(self):
        print('Quantizing colors of source image...')

        source_img = cv2.imread(self.SourceFilePath)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)

        height, width, depth = source_img.shape
        reshaped_img = source_img.reshape((height * width, depth))

        self.k_means = MiniBatchKMeans(n_clusters=8)
        self.k_means.fit(reshaped_img)

        self.colors_lab = self.k_means.cluster_centers_.astype('uint8')

        pixel_labels = self.k_means.predict(reshaped_img)
        self.source_q = self.colors_lab[pixel_labels]
        self.source_q = self.source_q.reshape((height, width, depth))

        rgb_img = cv2.cvtColor(self.source_q, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_quantized.jpg'), rgb_img)

        print('Done.')

    def slic_source(self):
        print('Applying SLIC for source image...')

        source_img = cv2.imread(self.SourceFilePath)

        groups = slic(image=img_as_float(source_img), n_segments=100, compactness=10, sigma=1)
        group_ids = np.unique(groups)

        self.source_centroids = np.array([np.mean(np.nonzero(groups == i), axis=1) for i in group_ids])

        for group in group_ids:
            mask = np.zeros(self.source_q.shape[:2], dtype="uint8")
            mask[groups == group] = 255
            superpixel = cv2.bitwise_and(self.source_q, self.source_q, mask=mask)
            self.source_superpixels.append(superpixel)
            cv2.imwrite(os.path.join(self.SourceSuperpixelFilePath, f'source_superpixel_{group}.jpg'), superpixel)

        rgb_img = cv2.cvtColor(self.source_q, cv2.COLOR_LAB2BGR)
        slic_img = img_as_ubyte(mark_boundaries(rgb_img, groups))
        cv2.imwrite(os.path.join(self.TempFolderPath, f'source_slic.jpg'), slic_img)

        print('Done.')

    def slic_target(self):
        print('Applying SLIC for target image...')

        target_img = cv2.imread(self.TargetFilePath)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        groups = slic(image=img_as_float(target_img), n_segments=100, compactness=0.1, sigma=1)
        group_ids = np.unique(groups)

        for group in group_ids:
            mask = np.zeros(target_img.shape[:2], dtype="uint8")
            mask[groups == group] = 255
            superpixel = cv2.bitwise_and(target_img, target_img, mask=mask)
            self.target_superpixels.append(superpixel)
            cv2.imwrite(os.path.join(self.TargetSuperpixelFilePath, f'target_superpixel_{group}.jpg'), superpixel)

        slic_img = img_as_ubyte(mark_boundaries(target_img, groups))
        cv2.imwrite(os.path.join(self.TempFolderPath, f'target_slic.jpg'), slic_img)

        print('Done.')

    def computer_source_surf(self):
        print('Computing surf features for source image...')

        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)

        for idx, superpixel in enumerate(self.source_superpixels):
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)
            self.source_surf.append(descriptors)
            surf_img = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)
            cv2.imwrite(os.path.join(self.SourceSurfFilePath, f'source_surf_{idx}.jpg'), surf_img)

        print('Done.')

    def compute_target_surf(self):
        print('Computing surf features for target image...')

        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)

        for idx, superpixel in enumerate(self.target_superpixels):
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)
            self.target_surf.append(descriptors)
            surf_img = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)
            cv2.imwrite(os.path.join(self.TargetSurfFilePath, f'target_surf_{idx}.jpg'), surf_img)

        print('Done.')

    def build_kernels(self):
        print('Building gabor kernels...')

        for idx, theta in enumerate(np.arange(0, np.pi, np.pi / 16)):
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, 0.5, 0.0, cv2.CV_32F)

            h, w = kernel.shape[:2]
            kernel_img = cv2.resize(kernel, (10 * w, 10 * h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.GaborKernelsFilePath, f'gabor_kernel_{idx}.jpg'), kernel_img)

            kernel /= 1.5 * kernel.sum()
            self.gabor_kernels.append(kernel)

        print('Done.')

    def apply_kernels(self, image):
        response = np.zeros_like(image)
        responses = []

        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            responses.append(filtered)
            np.maximum(response, filtered, response)

        return response, responses

    def computer_source_gabor(self):
        print('Computing gabor features for source image...')

        for idx, superpixel in enumerate(self.source_superpixels):
            gabor_img, responses = self.apply_kernels(superpixel)
            self.source_gabor.append(responses)
            cv2.imwrite(os.path.join(self.SourceGaborFilePath, f'source_gabor_{idx}.jpg'), gabor_img)

        print('Done.')

    def computer_target_gabor(self):
        print('Computing gabor features for target image...')

        for idx, superpixel in enumerate(self.target_superpixels):
            gabor_img, responses = self.apply_kernels(superpixel)
            self.target_gabor.append(responses)
            cv2.imwrite(os.path.join(self.TargetGaborFilePath, f'target_gabor_{idx}.jpg'), gabor_img)

        print('Done.')

    def make_source_dataset(self):
        print('Making dataset for source image...')

        self.colors_ab = self.colors_lab[:, 1:]

        for idx, color in enumerate(self.colors_ab):
            self.colors_ab_idx[color[0], color[1]] = idx

        centroid_colors = []

        for superpixel in self.source_superpixels:
            x_s, y_s, _ = np.nonzero(superpixel)
            items = [superpixel[i, j, :] for i, j in zip(x_s, y_s)]
            items = np.array(items)
            avg_L = np.mean(items[:, 0])
            avg_a = np.mean(items[:, 1])
            avg_b = np.mean(items[:, 2])
            label = self.k_means.predict([[avg_L, avg_a, avg_b]])
            color = self.colors_lab[label, 1:]
            centroid_colors.append(color)

        surf_avg = []

        for surf in self.source_surf:
            average = np.mean(surf, axis=0).tolist()
            surf_avg.append(average)

        gabor_avg = []

        for gabor_superpixel in self.source_gabor:
            local_avg = []
            for gabor in gabor_superpixel:
                average = np.mean(gabor[gabor != 0])
                local_avg.append(average)
            gabor_avg.append(local_avg)

        sup_count = len(self.source_superpixels)

        for i in range(sup_count):
            surf_feature = surf_avg[i]
            gabor_feature = gabor_avg[i]
            color = centroid_colors[i]

            sample = surf_feature + gabor_feature
            self.source_x.append(sample)
            self.source_y.append(self.colors_ab_idx[color[0, 0], color[0, 1]])

        self.source_x = preprocessing.scale(self.source_x)
        print(self.source_y)

        print('Done.')

    def make_target_dataset(self):
        print('Making dataset for target image...')

        surf_avg = []

        for surf in self.target_surf:
            average = np.mean(surf, axis=0).tolist()
            surf_avg.append(average)

        gabor_avg = []

        for gabor_superpixel in self.target_gabor:
            local_avg = []
            for gabor in gabor_superpixel:
                average = np.mean(gabor[gabor != 0])
                local_avg.append(average)
            gabor_avg.append(local_avg)

        sup_count = len(self.target_superpixels)

        for i in range(sup_count):
            surf_feature = surf_avg[i]
            gabor_feature = gabor_avg[i]

            sample = surf_feature + gabor_feature
            self.target_x.append(sample)

        self.target_x = preprocessing.scale(self.target_x)
        print('Done.')

    def train_svm(self):
        print('Training SVM...')

        self.classifier = svm.SVC()
        self.classifier.fit(self.source_x, self.source_y)
        predictions = self.classifier.predict(self.source_x)
        print(metrics.accuracy_score(self.source_y, predictions))

        print('Done.')

    def colorize_target(self):
        print('Colorizing target image...')

        labels = self.classifier.predict(self.target_x)
        color_labels = self.colors_ab[labels]

        print(labels)

        target_img = cv2.imread(self.TargetFilePath)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        colored_img = np.zeros((target_img.shape[0], target_img.shape[1], 3), dtype='uint8')

        for idx, superpixel in enumerate(self.target_superpixels):
            x_s, y_s = np.nonzero(superpixel)

            for i, j in zip(x_s, y_s):
                L = target_img[i, j]
                a = color_labels[idx, 0]
                b = color_labels[idx, 1]

                colored_img[i, j, 0] = L
                colored_img[i, j, 1] = a
                colored_img[i, j, 2] = b

        colored_img = cv2.cvtColor(colored_img, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(self.TempFolderPath, f'target_colored.jpg'), colored_img)

        print('Done.')
