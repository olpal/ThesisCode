from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import misc
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util as u
import sys
import parameters as hp
import numpy as np
import csv

class ImageProcessing():
    """
    This class is used to provide various image processing functions
    """
    def __init__(self):
        """
        Constructor for the image processing class
        """
        self.params = hp.ImageProcessingParameters()

    def load_parameters(self,input_file,overlay_image,output_dir,mode,image_prefix,image_postfix,export_data,export_overlay):
        """
        This function is used to set image processing parameters
        :param input_file: Full path to the image file to process
        :param overlay_image: Full path to the image file to overlay data onto
        :param output_dir: Full path to a directory to save data in
        :param mode: Integer representing which image processing mode to use 1=log, 2=dog, 3=doh
        :param image_prefix: String to add before saved image file names
        :param image_postfix: String to add after saved image file names
        :param export_data: Boolean indicating whether to save a data text file
        :param export_overlay: Boolean indicating whether to produce an overlay image
        :return: None
        """
        self.params.input_file = input_file
        self.params.output_dir = output_dir
        self.params.overlay_file = overlay_image
        self.params.mode = mode
        self.params.image_prefix = image_prefix
        self.params.image_postfix = image_postfix
        self.params.export_data = export_data
        self.params.export_overlay = export_overlay

    def set_params_log(self):
        """
        Set image processing parameters for optimal LoG processing
        :return: None
        """
        u.printf("Loading params for LoG processing...")
        self.params.min_sigma = 20
        self.params.max_sigma = 40
        self.params.num_sigma = 5
        self.params.threshold = 0.2
        self.params.overlap = 0.7
        self.params.log_Scale = False


    def set_params_dog(self):
        """
        Set image processing parameters for optimal DoG processing
        :return: None
        """
        u.printf("Loading params for DoG processing...")
        self.params.min_sigma = 20
        self.params.max_sigma = 50
        self.params.threshold = 2.0
        self.params.overlap = 0.5
        self.params.sigma_ratio = 1.6


    def set_params_doh(self):
        """
        Set image processing parameters for optimal DoH processing
        :return: None
        """
        u.printf("Loading params for DoH processing...")
        self.params.min_sigma = 15
        self.params.max_sigma = 60
        self.params.num_sigma = 15
        self.params.threshold = 0.01
        self.params.overlap = 0.5
        self.params.log_Scale = True


    def load_data(self):
        """
        Load the provided image
        :return: None
        """
        u.printf("Loading image...")
        global image
        image = misc.imread(self.params.input_file, flatten=True).astype(np.uint8)


    def run_log(self):
        """
        Run Laplacian of Gaussian Image processing
        :return: A data matrix of found blobs
        """
        u.printf("Running Laplacian of Gaussian")
        data_matrix = blob_log(image, min_sigma=self.params.min_sigma, max_sigma=self.params.max_sigma, num_sigma=self.params.num_sigma, \
                               overlap=self.params.overlap, log_scale=self.params.log_Scale, threshold=self.params.threshold)
        data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
        return data_matrix


    def run_dog(self):
        """
        Run Difference of Gaussian processing
        :return: Matrix of found blobs
        """
        u.printf("Running Difference of Gaussian")
        data_matrix = blob_dog(image, min_sigma=self.params.min_sigma, max_sigma=self.params.max_sigma, sigma_ratio=self.params.sigma_ratio,
                               overlap=self.params.overlap, threshold=self.params.threshold)
        data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
        return data_matrix


    def run_doh(self):
        """
        Run Determinant of Hessian processing algorithm
        :return: Matrix of blob predictions
        """
        u.printf("Running Determinant of Hessian")
        return blob_doh(image, min_sigma=self.params.min_sigma, max_sigma=self.params.max_sigma, num_sigma=self.params.num_sigma,
                        overlap=self.params.overlap, log_scale=self.params.log_Scale, threshold=self.params.threshold)


    def display_data(self, in_image, circles, output_path):
        """
        This function is used to display predictions
        :param in_image: The image to display predictions on
        :param circles: A collection of blob predictions
        :param output_path: The path to save the created image to
        :return: None
        """
        plt.clf()
        plt.imshow(in_image, interpolation='nearest', cmap="gray")
        fig = plt.gcf()
        axix = fig.gca()
        for circle in circles:
            y, x, r = circle
            circle = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
            axix.add_artist(circle)
        if self.params.display:
            plt.show()
        else:
            fig.savefig(output_path, dpi=250)

    def export_data(self, circles, output_file):
        """
        This function creates a file with x,y and radius coordinates
        :param circles: List of blobs to export
        :param output_file: Path to save data file to
        :return: None
        """
        try:
            with open(output_file, 'w') as csv_file:
                writer = csv.writer(csv_file,delimiter=",")
                for circle in circles:
                    y, x, r = circle
                    writer.writerow([x,y,r])
        except:
            u.printf("Unable to write data to file: {}".format(output_file))



    def statistics(self):
        """
        This function writes all parameter values to the console
        :return: none
        """
        u.printf("Running processing with parameters: Min Sigma:{} Max Sigma:{} Num Sigma:{} Overlap:{} Sigma Ratio:{}"
                 " Threshold:{} Log Scale:{} Image Postfix:{}".format(self.params.min_sigma, self.params.max_sigma, self.params.num_sigma,
                                                                      self.params.overlap, self.params.sigma_ratio, self.params.threshold,
                                                                      self.params.log_Scale, self.params.image_postfix))

    def execute(self):
        """
        This function is used to execute an image processing run
        :return: None
        """
        self.load_data()
        execution_type=""
        if self.params.mode==1:
            if len(sys.argv) <= 1:
                self.set_params_log()
            self.statistics()
            data = self.run_log()
            execution_type="log"
        elif self.params.mode==2:
            if len(sys.argv) <= 1:
                self.set_params_dog()
            self.statistics()
            data = self.run_dog()
            execution_type="dog"
        elif self.params.mode==3:
            if len(sys.argv) <= 1:
                self.set_params_doh()
            self.statistics()
            data = self.run_doh()
            execution_type="doh"

        u.printf("Generating blob image")
        self.display_data(image, data, ("{}/{}_{}_{}.pdf".format(self.params.output_dir, execution_type,
                                                                 self.params.image_prefix ,self.params.image_postfix)))
        if self.params.export_overlay:
            u.printf("Creating overlay image")
            overlay_image = misc.imread(self.params.overlay_file)
            self.display_data(overlay_image, data, ("{}/{}_{}_overlay_{}.pdf".format(self.params.output_dir, execution_type,
                                                                                     self.params.image_prefix, self.params.image_postfix)))
        if self.params.export_data:
            u.printf("Exporting CSV data")
            self.export_data(data, ("{}/{}_{}_data_{}.csv".format(self.params.output_dir, execution_type,
                                                                  self.params.image_prefix, self.params.image_postfix)))

