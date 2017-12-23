from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import circle
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util as u
import sys
import parameters as hp
import numpy as np
from glob import glob
import os
import datetime as dt
from sklearn.utils import shuffle
import visualize as vs


params = hp.ImageProcessingParameters()



if len(sys.argv) > 1:
    u.printf ("Arguments found, loading...")
    params.min_sigma = float(sys.argv[1])
    params.max_sigma = float(sys.argv[2])
    params.num_sigma = float(sys.argv[3])
    params.overlap = float(sys.argv[4])
    params.log_Scale = sys.argv[5]
    params.threshold = float(sys.argv[6])
    params.dataset_location = sys.argv[7]
    params.output_dir = sys.argv[8]
    params.create_paths()
else:
    u.printf("Loading params for LoG processing...")
    params.min_sigma = 28
    params.max_sigma = 82
    params.num_sigma = 6.0
    params.threshold = 0.13
    params.overlap = 0.96
    params.log_Scale = False
    params.dataset_location = '/Users/aolpin/Documents/School/thesis/dataset-test/'
    params.output_dir = '/Users/aolpin/Documents/School/thesis/results/testresults/'


def run_log(in_image):
    """
    Run Laplacian of Gaussian Image processing
    :return: A data matrix of found blobs
    """
    data_matrix = blob_log(in_image, min_sigma=params.min_sigma, max_sigma=params.max_sigma,
                           num_sigma=params.num_sigma, \
                           overlap=params.overlap, log_scale=params.log_Scale,
                           threshold=params.threshold)
    data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
    return data_matrix

def display_data(image, prediction, filename):
    """
    This function is used to display predictions
    :param in_image: The image to display predictions on
    :param circles: A collection of blob predictions
    :param output_path: The path to save the created image to
    :return: None
    """
    plt.clf()
    #plt.imshow(image, interpolation='nearest', cmap="gray")
    plt.imsave('{}/images/{}'.format(params.output_dir,filename.replace(".bmp",".jpg")), image, cmap='gray')
    plt.clf()
    #plt.imshow(prediction, interpolation='nearest', cmap="gray")
    plt.imsave('{}/masks/{}'.format(params.output_dir,(filename.replace(".bmp", "_mask.jpg"))), prediction, cmap='gray')

def compare_accuracy(prediction, mask):
    per_pixel_accuracy.append(vs.calculate_test_accuracy(prediction,mask))
    if (prediction == mask).all():
        per_image_accuracy.append(1)
    else:
        per_image_accuracy.append(0)

def produce_matrix(image_shape, circles):
    matrix = np.zeros((image_shape[0],image_shape[1]))
    for cir in circles:
        y, x, r = cir
        rr, cc = circle(y, x, r ,(image_shape[0],image_shape[1]))
        matrix[rr, cc ] = 1

    return matrix


def statistics():
    """
    This function writes all parameter values to the console
    :return: none
    """
    u.printf("Running processing with parameters: Min Sigma:{} Max Sigma:{} Num Sigma:{} Overlap:{}"
             " Threshold:{} Log Scale:{}".format(params.min_sigma, params.max_sigma,
                                                                  params.num_sigma,
                                                                  params.overlap,
                                                                  params.threshold,
                                                                  params.log_Scale))

if __name__ == '__main__':
    u.printf("Starting model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))
    statistics()
    per_image_accuracy=[]
    per_pixel_accuracy = []
    image_files = glob('{}images/*.bmp'.format(params.dataset_location))

    image_files = shuffle(image_files, random_state=346)
    u.printf("Found: {} ".format(len(image_files)))

    image_count=1
    for input_file in image_files:
        file_name = os.path.basename(input_file)
        mask_file_name = "{}masks/{}".format(params.dataset_location,file_name.replace(".bmp","_mask.jpg"))

        image = misc.imread(input_file, flatten=True).astype(np.uint8)
        mask = misc.imread(mask_file_name, flatten=True).astype(np.uint8)
        mask[mask > 0] = 1

        circles = run_log(image)
        prediction = produce_matrix(image.shape, circles)

        compare_accuracy(prediction.astype(np.uint8), mask)

        if image_count % params.training_display == 0:
            u.printf("Processed: {} images".format(image_count))

        display_data(image, prediction, file_name)

        if image_count == params.to_run:
            break

        image_count+=1

    accuracy_array = np.array(per_pixel_accuracy).sum(0)
    denominator = float(len(per_image_accuracy))
    u.printf("Test Accuracy: {} Image Accuracy: {} F1: {} Recall: {} Precision: {}".format(
        (accuracy_array[0] / denominator),
        (float(sum(per_image_accuracy)) / denominator),
        (accuracy_array[1] / denominator),
        (accuracy_array[2] / denominator),
        (accuracy_array[3] / denominator)))

    u.printf("Finished model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))
