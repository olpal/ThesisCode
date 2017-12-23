import numpy as np
from glob import glob
import random as rd
from sklearn.utils import shuffle
import scipy.misc as sc
import parameters as params
import util as u
import os
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DataSet:
    """
    This class provides various functions for interacting with a dataset
    """
    def __init__(self, location, output):
        """
        Constructor for the DataSet class
        :param location: The location of the dataset
        :param output: The output location to save data to
        """
        self.dp = params.DataSetParameters(location, output)

    def load_datasets(self,random_seed):
        """
        This function loads a list of file names for both images and binary maps while performing kfold splitting
        :return: a collection of label file paths, image file paths, and fold splits
        """
        u.printf ("Loading file names from {}...".format(self.dp.dataset_location))
        dataset_images, dataset_labels = self.read_image_label_names()

        u.printf ("Randomizing loaded data...")
        dataset_labels, dataset_images = shuffle(dataset_labels, dataset_images, random_state=random_seed)

        u.printf ("Producing {} folds...".format(self.dp.k_fold))
        k_fold = KFold(self.dp.k_fold, random_state=random_seed)
        k_fold_splits = k_fold.split(dataset_images)

        return dataset_labels, dataset_images, k_fold_splits

    def read_image_label_names(self):
        """
        This function is used to load a list of label files
        :return: An array of image and mask file paths
        """
        dataset_images = []
        dataset_labels = []
        number_loaded = 0
        files = glob('{}images/*.{}'.format(self.dp.dataset_location,self.dp.input_file_ext))
        for input_file in files:
            number_loaded += 1

            file_name = os.path.basename(input_file)
            mask_file_name = "{}masks/{}".format(self.dp.dataset_location, file_name.replace(".bmp", "_mask.jpg"))

            if not os.path.isfile(mask_file_name):
                print("{} does not exist as a file".format(mask_file_name))
                continue

            dataset_images.append(input_file)
            dataset_labels.append(mask_file_name)

            if number_loaded >= self.dp.samples_to_load != 0:
                break
            if number_loaded % self.dp.display == 0 and self.dp.display != 0:
                u.printf ("Loaded {} file names".format(number_loaded))

        u.printf ("Total file names: {}".format(len(files)))
        return np.array(dataset_images), np.array(dataset_labels)

    def load_image_data(self, file_names, start_position, to_load, params, binary=False):
        """
        This function is used to load image data
        :param label_names: An array of filenames to load training data from
        :param start_position: The start position in the array
        :param to_load: The number of records to load
        :param params: Model Parameter object
        :return: An array where each entry is an array of values representing an image
        """
        data_set_images = []
        number_loaded = 0
        number_of_files = len(file_names)
        image_position = start_position
        while image_position < number_of_files and number_loaded < to_load:

            if binary:
                unflattened_image=sc.imread(file_names[image_position],flatten=True, mode='L').astype(float)
                unflattened_image[unflattened_image > 0] = 1
            else:
                unflattened_image = sc.imread(file_names[image_position], flatten=True).astype(float)

            if params.cnn:
                data_set_images.append(unflattened_image.flatten())
            else:
                data_set_images.append(unflattened_image)
            number_loaded += 1
            image_position += 1

        if params.cnn:
            return np.array(data_set_images)
        else:
            return np.array(data_set_images).reshape((-1,params.image_size,params.image_size,1))

    @staticmethod
    def next_batch(labels, images, batch_size):
        """
        This function is used to generate the next batch of images and labels
        :param labels: An array of labeled data
        :param images: An array of image data
        :param batch_size: The size of the batch to generate
        :return: A mini batch derived from the passed in data arrays
        """
        data_size=len(labels)
        if data_size > batch_size:
            low_range = 0
            high_range=data_size-batch_size
            start_pos = rd.randint(low_range, high_range)
            return images[start_pos:(start_pos + batch_size)], labels[start_pos:(start_pos + batch_size)]
        else:
            return images, labels
