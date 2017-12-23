import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cnn as cn
import parameters as hp
import util as u
import csv
import numpy as np
import sys
import datetime as dt

version=2.0

class Model():
    """
    Class used to execute a trained model
    """
    def __init__(self):
        """
        Constructor for the Model class
        :param model_directory: Directory to saved model files
        :param input_image: Image to use for making predictions on
        :param output_directory: Path to save data to
        """
        u.printf ("Creating model parameters")
        self.train_params = hp.ModelParameters()
        self.exec_params = hp.ExecutionParameters()

        u.printf ("Initializing Neural Network parameters...")
        self.x = tf.placeholder('float32', [None, (self.train_params.image_size * self.train_params.image_size)])
        self.keep_prob = tf.placeholder(tf.float32)

    def load_model(self):
        """
        This function loads a model from the saved model files
        :return: Boolean indicating if the laod was successfull
        """
        parameters=[]
        self.exec_params.variable_full_path='{}/{}'.format(self.exec_params.model_directory,self.train_params.saved_variable_file_name)
        self.exec_params.model_full_path = '{}/{}'.format(self.exec_params.model_directory, self.train_params.saved_model_file_name)
        try:
            with open(self.exec_params.variable_full_path, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    parameters.append(row)
                self.train_params.number_of_classes = int(parameters[0][0])
                self.train_params.neuron_multiplier = int(parameters[0][1])
                self.train_params.convolutional_filter = int(parameters[0][2])
                self.train_params.convolutional_layer_count = int(parameters[0][3])
                return True
        except:
            print "No file found at location {}".format(self.exec_params.variable_full_path)
            return False

    def create_network(self):
        """
        This function creates a Convolutional Neural Network
        :return: The output layer to the Fully Connected Network
        """
        """Neural Network Variables"""
        cnn = cn.CNN(self.x, self.keep_prob, self.train_params.convolutional_layer_count, self.train_params.image_size, self.train_params.number_of_classes,
                     self.train_params.neuron_multiplier, self.train_params.convolutional_filter, cnn=self.exec_params.cnn)
        return cnn.return_network()

    def segment_image(self,image):
        """
        Function to segment an image into a series of smaller images
        :param image: The image to segment
        :return: An array of tuples of slice coordinates where each tuple contains: ystart, yend, xstart, xend
        """
        slices = []
        shape = image.shape
        yStartPos = 0
        finalY = False
        while True:
            finalX = False
            xStartPos = 0
            while True:
                slices.append((yStartPos, (yStartPos + self.exec_params.y_size), xStartPos, (xStartPos + self.exec_params.x_size)))
                xStartPos = (xStartPos + (self.exec_params.x_size - self.exec_params.x_overlap))
                if (xStartPos + self.exec_params.x_size) > shape[1] and finalX:
                    break
                elif (xStartPos + self.exec_params.x_size) > shape[1]:
                    xStartPos = (shape[1] - self.exec_params.x_size)
                    finalX = True
            yStartPos = (yStartPos + (self.exec_params.y_size - self.exec_params.y_overlap))
            if (yStartPos + self.exec_params.y_size) > shape[0] and finalY:
                break
            elif (yStartPos + self.exec_params.y_size) > shape[0]:
                yStartPos = (shape[0] - self.exec_params.y_size)
                finalY = True

        return slices

    def stitch_image(self, slices, predictions, probabilities, shape):
        """
        This function stitches a segmented image back together
        :param slices: An array of slice tuples to use for stitching process
        :param predictions: An array of predictions where each entry is an array of binary data
        :param shape: Shape of the image
        :return:
        """
        predicted_image = np.zeros((shape[0],shape[1]),dtype=int)
        predicted_image_prob = np.zeros((shape[0], shape[1]), dtype=float)
        image_position=0
        for slice in slices:
            predicted_image[slice[0]:slice[1], slice[2]:slice[3]] = np.reshape(predictions[image_position],(
                self.train_params.image_size,self.train_params.image_size))
            predicted_image_prob[slice[0]:slice[1], slice[2]:slice[3]] = np.reshape(probabilities[image_position], (
                self.train_params.image_size, self.train_params.image_size))
            image_position+=1
        return predicted_image, predicted_image_prob

    def predict_images(self, sess, network, image, slices):
        """
        This function is used to create predictions
        :param sess: A tensorflow session
        :param network: The network to use for generating predictions
        :param image: The image to create predictions for
        :param slices: Slice coordinates to use for segmenting the master image
        :return:
        """
        predicted_images=[]
        predicted_images_probabilities= []
        prediction_rounded = tf.round(tf.nn.sigmoid(network))
        prediction_true = tf.nn.sigmoid(network)
        for slice in slices:
            to_process=[]
            to_process_image = image[slice[0]:slice[1], slice[2]:slice[3]]
            to_process.append(np.reshape(to_process_image, (self.train_params.image_size*self.train_params.image_size,)))
            rounded_values, true_values = sess.run([prediction_rounded,prediction_true], feed_dict={self.x: to_process, self.keep_prob: 1.})
            predicted_images.append(rounded_values)
            predicted_images_probabilities.append(true_values)
        return predicted_images, predicted_images_probabilities

    def export_image(self, image_data, prob_data, original_image):
        """
        This function saves an image
        :param image_data: An array of image data to save
        :return: None
        """
        #plt.imshow(image_data, cmap="gray")
        plt.imsave('{}/{}'.format(self.exec_params.output_directory,self.exec_params.prediction_file_name), image_data, cmap="gray", dpi=250)
        plt.clf()
        if self.exec_params.overlay:
            map_visualized = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), prob_data))
            plt.axis('off')
            plt.imshow(original_image, cmap='gray_r')
            plt.imshow(map_visualized, cmap='jet', alpha=0.35, interpolation='none', vmin=0, vmax=1)
            plt.savefig('{}/{}'.format(self.exec_params.output_directory, self.exec_params.prediction_file_name.replace(".png","_jet.png")), dpi=200)

    def execute(self):
        """
        This function is the master execution function of the class
        :return: None
        """
        u.printf("Executing version {} of the code".format(version))
        if not self.load_model():
            exit()
        network = self.create_network()
        tf.nn.sigmoid(network)
        image = plt.imread(self.exec_params.input_image)
        slices = self.segment_image(image)

        u.printf("Predicting image...")
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.exec_params.model_full_path)
            predicted_images, predicted_images_probabilities=self.predict_images(sess, network, image, slices)

        u.printf("Stitching image")
        full_predicted_image, full_probability_image = self.stitch_image(slices,predicted_images,predicted_images_probabilities,image.shape)
        self.export_image(full_predicted_image,full_probability_image,image)

model = Model()

if len(sys.argv) > 1:
    u.printf ("Arguments found, loading...")
    model.exec_params.input_image = sys.argv[1]
    model.exec_params.output_directory = sys.argv[2]
    model.exec_params.model_directory = sys.argv[3]
    model.exec_params.cnn = (sys.argv[4] == "True")
else:
    model.exec_params.input_image = "/Users/aolpin/Documents/School/thesis/dataset-test/images/1_2016-11-27-03_38.bmp"
    #model.exec_params.output_directory = "/Users/aolpin/Documents/School/thesis/results/testresults/cnn"
    model.exec_params.output_directory = "/Users/aolpin/Documents/School/thesis/results/testresults/fcn"
    #model.exec_params.model_directory = "/Users/aolpin/Documents/School/thesis/results/cnnresults1/2017-10-24--00-20-08-057174142"
    model.exec_params.model_directory = "/Users/aolpin/Documents/School/thesis/results/fcnresults3/2017-10-25--01-16-07-143592620"
    model.exec_params.cnn = False
    model.exec_params.overlay = True

if __name__ == '__main__':
    u.printf("Starting model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))
    model.execute()
    u.printf("Finished model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))