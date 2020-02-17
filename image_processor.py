import os
import shutil

import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing, svm, metrics
from sklearn.cluster import MiniBatchKMeans


class ImageProcessor:
    # The folder where all temporary files for each run will be saved
    TempFolderPath = './temp'

    # The source image to be used for training
    SourceFilePath = './source.jpg'

    # The target image to be used for colorization
    TargetFilePath = './target.jpg'

    # The folder where all source superpixels will be stored
    SourceSuperpixelFilePath = f'{TempFolderPath}/source_superpixels'

    # The folder where all target superpixels will be stored
    TargetSuperpixelFilePath = f'{TempFolderPath}/target_superpixels'

    # The folder where all source surf features will be stored
    SourceSurfFilePath = f'{TempFolderPath}/source_surf'

    # The folder where all target surf features will be stored
    TargetSurfFilePath = f'{TempFolderPath}/target_surf'

    # The folder where all generated gabor kernels will be stored
    GaborKernelsFilePath = f'{TempFolderPath}/gabor_kernels'
    
    # The folder where all source gabor features will be stored
    SourceGaborFilePath = f'{TempFolderPath}/source_gabor'

    # The folder where all target gabor features will be stored
    TargetGaborFilePath = f'{TempFolderPath}/target_gabor'

    def __init__(self):
        # Delete the temp folder so we can store the new files
        shutil.rmtree(self.TempFolderPath)

        # Create all temp folders
        os.mkdir(self.TempFolderPath)
        os.mkdir(self.SourceSuperpixelFilePath)
        os.mkdir(self.TargetSuperpixelFilePath)
        os.mkdir(self.SourceSurfFilePath)
        os.mkdir(self.TargetSurfFilePath)
        os.mkdir(self.GaborKernelsFilePath)
        os.mkdir(self.SourceGaborFilePath)
        os.mkdir(self.TargetGaborFilePath)

        # Initialize all shared variables between methods
        self.colors_lab = []
        self.k_means = None
        self.source_q = None
        self.source_superpixels = []
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

        # Convert the source image from RGB to LAB and store all three channels
        source_img = cv2.imread(self.SourceFilePath)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(source_img)

        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_L.jpg'), L)
        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_a.jpg'), a)
        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_b.jpg'), b)

        # Convert the target image from RGB to greyscale and store it
        target_img = cv2.imread(self.TargetFilePath)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(self.TempFolderPath, 'target_greyscale.jpg'), target_img)

        print('Done.')

    def quantize_source(self):
        print('Quantizing colors of source image...')

        # Get the source image with LAB colors
        source_img = cv2.imread(self.SourceFilePath)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)

        # Reshape the image so we can feed it as input for k-means
        height, width, depth = source_img.shape
        reshaped_img = source_img.reshape((height * width, depth))

        # Initialize k-means algorithm, fit on source image
        self.k_means = MiniBatchKMeans(n_clusters=8)
        self.k_means.fit(reshaped_img)

        # Store centroid LAB colors for later
        self.colors_lab = self.k_means.cluster_centers_.astype('uint8')

        # Reconstruct the quantized version of the source image
        pixel_labels = self.k_means.predict(reshaped_img)
        self.source_q = self.colors_lab[pixel_labels]
        self.source_q = self.source_q.reshape((height, width, depth))

        # Convert to RGB and store it
        rgb_img = cv2.cvtColor(self.source_q, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(self.TempFolderPath, 'source_quantized.jpg'), rgb_img)

        print('Done.')

    def slic_source(self):
        print('Applying SLIC for source image...')

        # Get the RGB version of the source image
        source_img = cv2.imread(self.SourceFilePath)

        # Run the slic algorithm
        groups = slic(image=img_as_float(source_img), n_segments=100, compactness=10, sigma=1)
        group_ids = np.unique(groups)

        # Iterate through each SLIC group
        for group in group_ids:
            # Create a mask to separate the superpixel
            mask = np.zeros(self.source_q.shape[:2], dtype="uint8")
            mask[groups == group] = 255
            superpixel = cv2.bitwise_and(self.source_q, self.source_q, mask=mask)

            # Store the superpixel for later
            self.source_superpixels.append(superpixel)

            # Save the superpixel
            cv2.imwrite(os.path.join(self.SourceSuperpixelFilePath, f'source_superpixel_{group}.jpg'), superpixel)

        # Get the quantized source image and mark all SLIC boundaries
        rgb_img = cv2.cvtColor(self.source_q, cv2.COLOR_LAB2BGR)
        slic_img = img_as_ubyte(mark_boundaries(rgb_img, groups))

        # Save the SLIC image
        cv2.imwrite(os.path.join(self.TempFolderPath, f'source_slic.jpg'), slic_img)

        print('Done.')

    def slic_target(self):
        print('Applying SLIC for target image...')

        # Get the greyscale target image
        target_img = cv2.imread(self.TargetFilePath)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # Run the SLIC algorithm
        groups = slic(image=img_as_float(target_img), n_segments=100, compactness=0.1, sigma=1)
        group_ids = np.unique(groups)

        # Iterate through each SLIC group
        for group in group_ids:
            # Create a mask to separate the superpixel
            mask = np.zeros(target_img.shape[:2], dtype="uint8")
            mask[groups == group] = 255
            superpixel = cv2.bitwise_and(target_img, target_img, mask=mask)

            # Store the superpixel for later
            self.target_superpixels.append(superpixel)

            # Save the superpixel
            cv2.imwrite(os.path.join(self.TargetSuperpixelFilePath, f'target_superpixel_{group}.jpg'), superpixel)

        # Mark all SLIC boundaries
        slic_img = img_as_ubyte(mark_boundaries(target_img, groups))

        # Save the SLIC image
        cv2.imwrite(os.path.join(self.TempFolderPath, f'target_slic.jpg'), slic_img)

        print('Done.')

    def computer_source_surf(self):
        print('Computing surf features for source image...')

        # Create SURF object and set it to extended mode
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)

        # Iterate through all superpixels
        for idx, superpixel in enumerate(self.source_superpixels):
            # Store keypoints and descriptors
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)

            # Store the descriptors for later
            self.source_surf.append(descriptors)

            # Mark the keypoints on the superpixel
            surf_img = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)

            # Save the superpixel with keypoints
            cv2.imwrite(os.path.join(self.SourceSurfFilePath, f'source_surf_{idx}.jpg'), surf_img)

        print('Done.')

    def compute_target_surf(self):
        print('Computing surf features for target image...')

        # Create SURF object and set it to extended mode
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)

        # Iterate through all superpixels
        for idx, superpixel in enumerate(self.target_superpixels):
            # Store keypoints and descriptors
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)

            # Store the descriptors for later
            self.target_surf.append(descriptors)

            # Mark the keypoints on the superpixel
            surf_img = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)

            # Save the superpixel with keypoints
            cv2.imwrite(os.path.join(self.TargetSurfFilePath, f'target_surf_{idx}.jpg'), surf_img)

        print('Done.')

    def build_kernels(self):
        print('Building gabor kernels...')

        for idx, theta in enumerate(np.arange(0, np.pi, np.pi / 16)):
            # Create 16 gabor kernels with different orientations
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, 0.5, 0.0, cv2.CV_32F)

            # Store the kernel image
            h, w = kernel.shape[:2]
            kernel_img = cv2.resize(kernel, (10 * w, 10 * h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.GaborKernelsFilePath, f'gabor_kernel_{idx}.jpg'), kernel_img)

            # Normalize and save it for later
            kernel /= 1.5 * kernel.sum()
            self.gabor_kernels.append(kernel)

        print('Done.')

    def apply_kernels(self, image):
        # Create a response image that will include all reactions to the kernel
        response = np.zeros_like(image)

        # Separately store every reaction image for each kernel
        responses = []

        for kernel in self.gabor_kernels:
            # Create a filter and apply the kernel to the image
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)

            # Store the individual response
            responses.append(filtered)

            # Keep adding into the response image as we go through the kernels
            np.maximum(response, filtered, response)

        return response, responses

    def computer_source_gabor(self):
        print('Computing gabor features for source image...')

        for idx, superpixel in enumerate(self.source_superpixels):
            # Apply all gabor kernels to the superpixel
            gabor_img, responses = self.apply_kernels(superpixel)

            # Store all 16 responses for later
            self.source_gabor.append(responses)

            # Save the response image of the superpixel
            cv2.imwrite(os.path.join(self.SourceGaborFilePath, f'source_gabor_{idx}.jpg'), gabor_img)

        print('Done.')

    def computer_target_gabor(self):
        print('Computing gabor features for target image...')

        for idx, superpixel in enumerate(self.target_superpixels):
            # Apply all gabor kernels to the superpixel
            gabor_img, responses = self.apply_kernels(superpixel)

            # Store all 16 responses for later
            self.target_gabor.append(responses)

            # Save the response image of the superpixel
            cv2.imwrite(os.path.join(self.TargetGaborFilePath, f'target_gabor_{idx}.jpg'), gabor_img)

        print('Done.')

    def make_source_dataset(self):
        print('Making dataset for source image...')

        # Store a, b values of all quantized colors from k-means
        self.colors_ab = self.colors_lab[:, 1:]

        # Keep a LAB to index dictionary for all quantized colors
        for idx, color in enumerate(self.colors_ab):
            self.colors_ab_idx[color[0], color[1]] = idx

        # Calculate the LAB color for each superpixel
        centroid_colors = []

        for superpixel in self.source_superpixels:
            # Find all nonzero pixels within the superpixel
            x_s, y_s, _ = np.nonzero(superpixel)
            items = [superpixel[i, j, :] for i, j in zip(x_s, y_s)]
            items = np.array(items)

            # Calculate the mean of L, a, b values
            avg_L = np.mean(items[:, 0])
            avg_a = np.mean(items[:, 1])
            avg_b = np.mean(items[:, 2])

            # Quantized the mean color of the superpixel using k-means
            label = self.k_means.predict([[avg_L, avg_a, avg_b]])

            # Store a, b values of the superpixel
            color = self.colors_lab[label, 1:]
            centroid_colors.append(color)

        # Calculate 128 surf values
        surf_avg = []

        for surf in self.source_surf:
            # Compute the mean surf values for each superpixel
            average = np.mean(surf, axis=0).tolist()
            surf_avg.append(average)

        # Calculate 16 gabor values
        gabor_avg = []

        for gabor_superpixel in self.source_gabor:
            local_avg = []
            for gabor in gabor_superpixel:
                # Compute the mean gabor values for each superpixel
                average = np.mean(gabor[gabor != 0])
                local_avg.append(average)
            gabor_avg.append(local_avg)

        sup_count = len(self.source_superpixels)

        for i in range(sup_count):
            # For each superpixel get the surf, gabor values and the color
            surf_feature = surf_avg[i]
            gabor_feature = gabor_avg[i]
            color = centroid_colors[i]

            sample = surf_feature + gabor_feature
            self.source_x.append(sample)
            self.source_y.append(self.colors_ab_idx[color[0, 0], color[0, 1]])

        # Apply regularization to the dataset
        self.source_x = preprocessing.scale(self.source_x)
        print(self.source_y)

        print('Done.')

    def make_target_dataset(self):
        print('Making dataset for target image...')

        # Calculate 128 surf values
        surf_avg = []

        for surf in self.target_surf:
            # Compute the mean surf values for each superpixel
            average = np.mean(surf, axis=0).tolist()
            surf_avg.append(average)

        # Calculate 16 gabor values
        gabor_avg = []

        for gabor_superpixel in self.target_gabor:
            local_avg = []
            for gabor in gabor_superpixel:
                # Compute the mean gabor values for each superpixel
                average = np.mean(gabor[gabor != 0])
                local_avg.append(average)
            gabor_avg.append(local_avg)

        sup_count = len(self.target_superpixels)

        for i in range(sup_count):
            # For each superpixel get the surf and gabor values
            # Note that we are missing the color value here
            surf_feature = surf_avg[i]
            gabor_feature = gabor_avg[i]

            sample = surf_feature + gabor_feature
            self.target_x.append(sample)

        # Apply regularization to the dataset
        self.target_x = preprocessing.scale(self.target_x)
        print('Done.')

    def train_svm(self):
        print('Training SVM...')

        # Create a new SVM and train it on source data and labels
        self.classifier = svm.SVC()
        self.classifier.fit(self.source_x, self.source_y)

        # Make predictions and compute accuracy
        predictions = self.classifier.predict(self.source_x)
        print(metrics.accuracy_score(self.source_y, predictions))

        print('Done.')

    def colorize_target(self):
        print('Colorizing target image...')

        # Get predicted labels using the SVM for the target dataset
        labels = self.classifier.predict(self.target_x)

        # Get a, b values for color
        color_labels = self.colors_ab[labels]

        print(labels)

        # Get the greyscale target image
        target_img = cv2.imread(self.TargetFilePath)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # Create a blank copy of the target image to colorize
        colored_img = np.zeros((target_img.shape[0], target_img.shape[1], 3), dtype='uint8')

        for idx, superpixel in enumerate(self.target_superpixels):
            # For each superpixel find nonzero pixels
            x_s, y_s = np.nonzero(superpixel)

            for i, j in zip(x_s, y_s):
                # Colorize every pixel according to the predicted values
                L = target_img[i, j]
                a = color_labels[idx, 0]
                b = color_labels[idx, 1]

                colored_img[i, j, 0] = L
                colored_img[i, j, 1] = a
                colored_img[i, j, 2] = b

        # Convert the colorized image from LAB to RGB and store it
        colored_img = cv2.cvtColor(colored_img, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(self.TempFolderPath, f'target_colored.jpg'), colored_img)

        print('Done.')
