# Image Processing
A Python project that was developed as a university assignment for the subject of Image Processing. The program takes an input image and a reference dataset of photos. The goal is to colorize the greyscale image using a trained support vector machine. To achieve that, we have implemented a variety of image processing techniques. First, we change color spaces from RGB to LAB. Then, we apply the SLIC algorithm to find the group of superpixels for each image. These segments along with SURF and GABOR features are given as input for the SVM. Using scikit-learn, we use machine learning techniques to predict the color of a superpixel using the dataset superpixels as reference. The output of the program returns the colorized version of the input image.

**Group members involved in this project:**  
Ioannidis Panagiotis, Paravantis Athanasios, Nikas Dionisios

**Browse through related projects on thanosparavantis.com:**  
https://www.thanosparavantis.com/projects/image-processing
