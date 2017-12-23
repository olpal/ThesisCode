import tensorflow as tf
import numpy as np

class CNN:
    """
    The Convolutional Network Class holds all structural information
    relating to the network
    """
    convolutional_layers = []
    #Kernals
    pooling_kernel = 2
    #Size determinators
    layer_filter_increment=16
    neuron_multiplier=0
    #strides
    pool_stride = 2

    def __init__(self, inData, keep_rate, convolutional_layer_count, image_size, number_of_classes, neuron_multiplier, convolutional_filter, cnn=True):
        """
        Constructor for the CNN class
        :param inData: A matrix of data values that will be processed, primarily used for its shape
        :param keep_rate: A float representing a normalization parameter in the fully connected network
        :param convolutional_layer_count: Integer representing the number of convolutional layers
        :param image_size: Integer representing the size of the image
        :param number_of_classes: Integer representing the number of classes
        :param neuron_multiplier: Integer representing the number of neurons to include in the fully connected layer
        :param convolutional_filter: Integer representing the size of the convolutional kernel
        """
        # Scalar variables
        self.image_size = image_size
        self.number_of_classes = number_of_classes
        self.neuron_multiplier = neuron_multiplier
        self.convolutional_kernel = convolutional_filter
        #Shape data
        self.data = inData
        self.shape_data(self.data)
        self.cnn=cnn
        #Build convolutional layers
        previous_convolutional_layer=self.data
        for i in range(convolutional_layer_count):
            current_convolutional_layer = self.create_convolutional_layer(input_layer=previous_convolutional_layer,
                                                                    filters=(self.layer_filter_increment*(i+1)),
                                                                    convolutional_kernel=self.convolutional_kernel,
                                                                    pool_kernel=self.pooling_kernel,
                                                                    stride=self.pool_stride)
            previous_convolutional_layer = current_convolutional_layer
            self.convolutional_layers.append(current_convolutional_layer)
        if cnn:
            #Terminate network
            self.neuron_count = (self.neuron_multiplier * (self.return_final_convolutional_layer().get_shape()[3]).value)
            self.terminate_network_nn(keep_rate)
        else:
            self.terminate_network_fcn()

    def terminate_network_nn(self,keep_rate):
        """
        This function creates a fully connected network to terminate the CNN
        :param keep_rate: Regularization term for the fully connected network
        :return: none
        """
        # Create fully connected layer
        self.full_network = self.create_fully_connected(self.return_final_convolutional_layer(), keep_rate)

    def terminate_network_fcn(self):
        layer_1 = tf.layers.conv2d_transpose(
            inputs=self.return_final_convolutional_layer(),
            kernel_size=self.convolutional_kernel,
            strides=2,
            filters=self.return_final_convolutional_layer().shape[3],
            padding='SAME',
            activation=tf.nn.relu)
        layer_2 = tf.layers.conv2d_transpose(
            inputs=layer_1,
            kernel_size=self.convolutional_kernel,
            strides=(self.image_size/int(layer_1.shape[1]))/2,
            filters=(self.return_final_convolutional_layer().shape[3]/2),
            padding='SAME',
            activation=tf.nn.relu)
        layer_3 = tf.layers.conv2d_transpose(
            inputs=layer_2,
            kernel_size=self.convolutional_kernel,
            strides=(self.image_size/int(layer_2.shape[1])),
            filters=1,
            padding='SAME',
            activation=tf.nn.relu)
        self.upsmple = layer_3

    def shape_data(self, data):
        """
        This function reshapes the data matrix
        :param data: The data matrix to reshape
        :return: None
        """
        self.data = tf.reshape(data, shape=[-1, self.image_size, self.image_size, 1])

    def conv2d(self, input_layer, filters, kernel):
        """
        This function creates an activation layer
        :param input_layer: The previous layer to link this layer to, normally the prior max pooling layer
        :param filters: Number of filter maps the layer will have
        :param kernel: The size of the convolutional filter
        :return: the newly constructed convolutional layer
        """
        return tf.layers.conv2d(inputs=input_layer, filters=filters, kernel_size=[kernel, kernel],  padding="same", activation=tf.nn.relu)

    def maxpool2d(self, input_layer, pool_size, stride):
        """
        Creates a max pooling layer
        :param input_layer: Previous layer to link the max pooling layer to, normally an activation layer
        :param pool_size: Size of the pooling kernel
        :param stride: The number of pixels to move the kernel on each iteration
        :return: The newly constructed max pooling layer
        """
        return tf.layers.max_pooling2d(inputs=input_layer, pool_size=[pool_size, pool_size], strides=stride)

    def create_convolutional_layer(self, input_layer, filters, convolutional_kernel, pool_kernel, stride):
        """
        This function creates a convolutiona layer
        :param input_layer: The previous layer to link this layer to
        :param filters: Number of filter maps the layer will have
        :param convolutional_kernel: Integer representing the size of the convolutional filter
        :param pool_kernel: Integer representing the size of the pooling kernal
        :param stride: Integer representing how many pixels the kernal moves on each iteration
        :return:
        """
        convolutional_layer = self.conv2d(input_layer=input_layer, filters=filters, kernel=convolutional_kernel)
        convolutional_layer = self.maxpool2d(input_layer=convolutional_layer, pool_size=pool_kernel, stride=stride)
        return convolutional_layer

    def create_fully_connected(self, layer, keep_rate):
        """
        Creates a fully connected network
        :param layer: The final convolutional layer
        :param keep_rate: Regularization term to simulate dead neurons
        :return: Returns a fully connected network
        """
        shape = layer.get_shape()
        flat_layer = tf.reshape(layer, [-1, int(shape[1]) * int(shape[2]) * int(shape[3])])
        dense = tf.layers.dense(inputs=flat_layer, units=self.neuron_count, activation=tf.nn.relu)
        dropout_layer = tf.layers.dropout(inputs=dense, rate=keep_rate)
        return tf.layers.dense(inputs=dropout_layer, units=self.number_of_classes)

    def return_final_convolutional_layer(self):
        """
        Returns the final convolutional layer
        :return: Returns the last convolutional layer
        """
        return self.convolutional_layers[len(self.convolutional_layers)-1]

    def return_network(self):
        """
        Gets the convolutional network
        :return: The convolutional network existing in this class
        """
        if self.cnn:
            return self.full_network
        else:
            return self.upsmple
