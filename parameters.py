import util as u
import os

class ImageProcessingParameters():
    """
    This class contains a series of parameters used for image processing algorithms
    """
    # DOG,LOG,DOH - The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs
    min_sigma = 0
    # DOG,LOG,DOH - The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs
    max_sigma = 0
    # LOG,DOH - The number of intermediate values of standard deviations to consider between min_sigma and max_sigma
    num_sigma = 0
    # DOG,LOG,DOH - The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored.
    # Reduce this to detect blobs with less intensities.
    threshold = 0.0
    # DOG,LOG,DOH - A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold,
    # the smaller blob is eliminated"""
    overlap = 0.0
    # LOG,DOH - If set intermediate values of standard deviations are interpolated using a logarithmic scale to
    # the base 10. If not, linear interpolation is used
    log_Scale = False
    # DOG - The ratio between the standard deviation of Gaussian Kernels used for computing the Difference of Gaussians
    sigma_ratio = 0.0
    image_postfix = 1
    image_prefix = ""
    display = False
    export_overlay = True
    export_data = False
    mode = 1
    input_file = ""
    overlay_file = ""
    output_dir = ""
    dataset_location = ""
    training_display = 1
    to_run = 0

    def create_paths(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists("{}/images".format(self.output_dir)):
            os.mkdir("{}/images".format(self.output_dir))
        if not os.path.exists("{}/masks".format(self.output_dir)):
            os.mkdir("{}/masks".format(self.output_dir))

class DataSetParameters():
    """
    This class contains a series of parameters for use in the data set processing
    """
    """File extensions"""
    image_prefix = "img"
    input_file_ext = "png"
    output_file_ext = "pdf"
    output_graph_ext = "pdf"
    """Variables"""
    k_fold = 10
    samples_to_load = 0
    display = 5000
    max_classes = 0

    def __init__(self, location, output):
        """
        Constructor for the data set class
        :param location: Location of the data set
        :param output: Folder to save output data to
        """
        self.dataset_location = location
        self.output_dir=output
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

class ExecutionParameters():
    """
    This class contains a series of parameters to be used when using a trained model
    """
    model_directory = ""
    variable_full_path = ""
    model_full_path = ""
    input_image = ""
    output_directory = ""
    prediction_file_name = "predicted_image.png"
    x_size = 256
    y_size = 256
    x_overlap = 0
    y_overlap = 0
    export_data=False
    export_image=True
    cnn=False
    overlay=False

class ModelParameters():
    """
    This class contains a series of parameters and functions used when training a model
    """
    """File Paths"""
    dataset_location = "/Users/aolpin/Documents/School/thesis/dataset/"
    output_dir = "/Users/aolpin/Documents/School/thesis/results/testresults/"
    saved_model_file_name = "model.ckpt"
    saved_variable_file_name = "variables.txt"
    saved_images_file_name = "images.txt"
    saved_model_file_path = ""
    saved_variable_file_path = ""
    """Display variables"""
    training_display=100
    testing_display = 1000
    """Fixed Variables"""
    image_size=256
    number_of_classes=image_size*image_size
    visualize_images = 10
    testing_batch_size = 100
    """Default model variables"""
    """These will be overridden with passed in variables"""
    training_batch_size = 100
    learning_rate = 0.001
    dropout = 0.0
    epochs = 1
    convolutional_layer_count = 4
    neuron_multiplier = 0
    convolutional_filter = 8
    save_model = False
    cnn = False
    random_seed=859
    current_fold=0


    def __init__(self):
        """
        Constructor for the Model Parameters class
        """
        self.check_convolutional_layers()

    def generate_model_paths(self):
        """
        This function is used to generate full paths to both the model and variables files
        :return:
        """
        self.saved_model_file_path = "{}/{}".format(self.output_dir, self.saved_model_file_name)
        self.saved_variable_file_path = "{}/{}".format(self.output_dir, self.saved_variable_file_name)
        self.saved_images_file_name = "{}/{}".format(self.output_dir, self.saved_images_file_name)

    def check_convolutional_layers(self):
        """
        This function is used to validate the number of convolutional layers will work with the provided image size
        :return: None
        """
        u.printf ("Validating convolutional layers...")
        previous_size = self.image_size
        for i in range(self.convolutional_layer_count):
            current_size = previous_size / 2
            if current_size != 1:
                previous_size = current_size
                continue
            u.printf ("Maximum Convolutional Layer value reached\nSetting Convolutional Layer Count value to maximum: {}".format(i))
            self.convolutional_layer_count = i
            break

    def check_batch_size(self, data_set_size, batch_size):
        """
        This function checks that the mini batch size will work with the data set size
        :param data_set_size: Size of the data set
        :param batch_size: Size of the mini batch to use
        :return: None
        """
        u.printf ("Validating batch size of {}...".format(batch_size))
        original_batch_size=batch_size
        if batch_size > data_set_size:
            while batch_size > data_set_size:
                batch_size /= 2
            u.printf ("Batch size {} is greater than data set size {}\nBatch size has been reduced to {}".format(original_batch_size,data_set_size,batch_size))

    def check_visualize_image(self, test_label_count):
        if self.visualize_images > test_label_count:
            self.visualize_images = test_label_count