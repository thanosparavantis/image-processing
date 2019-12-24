import project_functions as funcs

lab_dataset = funcs.get_lab_dataset(path='./dataset')

quantized_dataset = funcs.quantize_images(lab_dataset)

slic_dataset, slic_groups = funcs.slic_images(quantized_dataset)

funcs.compute_surf(slic_groups)

funcs.compute_gabor(slic_groups)
