import project_functions as funcs

# lab_dataset = funcs.get_lab_dataset(path='./dataset_2')
# quantized_dataset, centroids = funcs.quantize_images(lab_dataset)
# slic_groups, slic_centroids = funcs.slic_images(quantized_dataset)
# surf_descriptors = funcs.compute_surf(slic_groups)
# gabor_groups = funcs.compute_gabor(slic_groups)
# X, y, ab_centroids = funcs.make_dataset(slic_groups, slic_centroids, surf_descriptors, gabor_groups, centroids)
# classifier = funcs.train_svm(X, y)
#
# test_image = funcs.test_get_image(path='./test.jpg')
# test_superpixels, test_slic_centroids = funcs.test_slic_image(test_image)
# test_surf_descriptors = funcs.test_compute_surf(test_superpixels)
# test_gabor_groups = funcs.test_compute_gabor(test_superpixels)
# test_X = funcs.test_make_dataset(test_surf_descriptors, test_gabor_groups)
#
# funcs.test_color_image(test_image, ab_centroids, classifier, test_X, test_superpixels)
from image_processor import ImageProcessor

processor = ImageProcessor()
processor.change_colorspace()
processor.quantize_source()
processor.slic_source()
processor.slic_target()
processor.build_kernels()
processor.computer_source_surf()
processor.compute_target_surf()
processor.computer_source_gabor()
processor.computer_target_gabor()
processor.make_source_dataset()
processor.make_target_dataset()
processor.train_svm()
processor.colorize_target()
