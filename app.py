import project_functions as funcs

lab_dataset = funcs.get_lab_dataset(
    path='./dataset'
)

# funcs.plot_images(
#     path='./temp/lab_result.jpg',
#     dataset=lab_dataset
# )

quantized_dataset = funcs.quantize_images(
    lab_dataset=lab_dataset
)

# funcs.plot_images(
#     path='./temp/quantized_result.jpg',
#     dataset=quantized_dataset
# )

slic_dataset, slic_groups = funcs.slic_images(
    quantized_dataset=quantized_dataset
)

# funcs.plot_images(
#     path='./temp/slic_result.jpg',
#     dataset=slic_dataset)

funcs.compute_surf(slic_groups)
