# ------------------------------------------ #
# Image Processing                           #
# ------------------------------------------ #
# P16036 | Ioannidis Panagiotis              #
# P16097 | Nikas Dionisios                   #
# P16112 | Paravantis Athanasios             #
# ------------------------------------------ #

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
