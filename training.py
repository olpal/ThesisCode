import tensorflow as tf
import cnn as cn
import visualize as vs
import dataset as ds
import parameters as hp
import sys
import util as u
import datetime as dt
import csv
import numpy as np

version = 1.0

u.printf ("Starting model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

"""Variables"""
u.printf ("Initializing list variables...")
training_losses = []
training_accuracies = []
testing_accuracies = []

u.printf ("Creating model parameters")
params = hp.ModelParameters()

"""Assign passed in variables if they exist"""
if len(sys.argv) > 1:
    u.printf ("Arguments found, loading...")
    params.dataset_location = sys.argv[1]
    params.output_dir = sys.argv[2]
    params.training_batch_size = int(sys.argv[3])
    params.learning_rate = float(sys.argv[4])
    params.dropout = float(sys.argv[5])
    params.epochs = int(sys.argv[6])
    params.convolutional_layer_count = int(sys.argv[7])
    params.neuron_multiplier = int(sys.argv[8])
    params.convolutional_filter = int(sys.argv[9])
    params.cnn = sys.argv[10].lower() == 'true'
    params.random_seed = int(sys.argv[11])
    params.current_fold = int(sys.argv[12])

"""Data sets"""
u.printf ("Building dataset...")
data_set = ds.DataSet(params.dataset_location, params.output_dir)
labels, data, splits = data_set.load_datasets(params.random_seed)
params.generate_model_paths()

"""Session Variables"""
u.printf ("Initializing Neural Network...")
if params.cnn:
    x = tf.placeholder('float32')
    y = tf.placeholder('float32')
else:
    x = tf.placeholder('float32', (None, params.image_size, params.image_size, 1))
    y = tf.placeholder('float32', (None, params.image_size, params.image_size, 1))
keep_prob = tf.placeholder(tf.float32)

"""Neural Network Variables"""
cnn = cn.CNN(x, keep_prob, params.convolutional_layer_count, params.image_size,
                               params.number_of_classes, params.neuron_multiplier, params.convolutional_filter, params.cnn)
prediction = cnn.return_network()

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(params.learning_rate).minimize(cost)

"""Overall Accuracy"""
correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(prediction)), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""All true Accuracy"""
all_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
accuracy_all_true = tf.reduce_mean(all_true)

"""Examination code"""
prediction_truth = correct_prediction
prediction_true_values = tf.nn.sigmoid(prediction)
prediction_round_values = tf.round(tf.nn.sigmoid(prediction))
cost_examination = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)


"""Train the neural network"""
def train_network(sess, train_data, train_labels, iteration):
    """
    This function is used to train a neural network
    :param sess: A tensor flow session
    :param train_data: A list of training data files
    :param train_labels: A list of training mask files
    :param iteration: The K-fold iteration
    :return:
    """
    sess.run(tf.global_variables_initializer())
    current_epoch = 1
    u.printf ("Beginning model training...")
    while current_epoch <= params.epochs:
        mini_batch_x, mini_batch_y = data_set.next_batch(train_labels, train_data, params.training_batch_size)
        mini_batch_x = data_set.load_image_data(mini_batch_x,0,params.training_batch_size, params)
        mini_batch_y = data_set.load_image_data(mini_batch_y,0,params.training_batch_size, params, binary=True)
        sess.run(optimizer, feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})

        if current_epoch % params.training_display == 0:
            training_accuracy, training_loss, predictions, costs = sess.run([accuracy, prediction_round_values, correct_prediction, prediction_true_values], feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})
            training_accuracy,training_accuracy2, training_loss = sess.run(
                    [accuracy, accuracy_all_true, cost], feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: 1.0})
            u.printf ("Iteration: {}, Training Accuracy: {} - {} Training Loss: {}".format(current_epoch, training_accuracy, training_accuracy2, training_loss))
            #training_losses[iteration-1].append((training_loss,current_epoch))
            #training_accuracies[iteration-1].append((training_accuracy,current_epoch))

        current_epoch += 1


def test_network(sess, test_data, test_labels):
    """
    This function is used to conduct testing on a neural network
    :param sess: A Tensorflow session
    :param test_data: A collection of testing image file names
    :param test_labels: A collection of testing image mask file names
    """

    start_index = 0
    total_elements = len(test_labels)
    denominator = total_elements
    accuracy_values = []
    image_accuracy_values = []
    while start_index < total_elements:

        """Data gathering"""
        test_data_real = data_set.load_image_data(test_data, start_index, params.testing_batch_size, params)
        test_labels_real = data_set.load_image_data(test_labels, start_index, params.testing_batch_size, params, binary=True)

        """Model execution"""
        accuracy_pred, all_accuracy, predictions = sess.run([accuracy, accuracy_all_true,prediction_round_values],
                                                            feed_dict={x: test_data_real, y: test_labels_real, keep_prob: 1.0})

        """Accuracy Calculation"""
        for i in xrange(len(test_labels_real)):
            label_array = np.array(test_labels_real[i]).reshape((params.image_size,params.image_size))
            prediction_array = np.array(predictions[i]).reshape((params.image_size,params.image_size))
            accuracy_values.append(vs.calculate_test_accuracy(prediction_array,label_array))
            image_accuracy_values.append(all_accuracy)

        if start_index % params.testing_display == 0:
            u.printf ('Tested {} data items'.format(start_index))

        start_index += params.testing_batch_size

    """Testing output"""
    if total_elements >= params.testing_batch_size:
        denominator = float(total_elements/params.testing_batch_size)

    accuracy_array = np.array(accuracy_values).sum(0)
    u.printf("Test Accuracy: {} Image Accuracy: {} F1: {} Recall: {} Precision: {}".format((accuracy_array[0] / total_elements),
                                                                                          ((sum(image_accuracy_values) / denominator)/100),
                                                                                          (accuracy_array[1] / total_elements),
                                                                                          (accuracy_array[2] / total_elements),
                                                                                          (accuracy_array[3] / total_elements)))

    u.printf("Debug: Accuracy{} TP: {} TN: {} FP: {} FN: {}".format(accuracy_pred,accuracy_array[4],accuracy_array[5],accuracy_array[6],accuracy_array[7]))


def generate_visuals(sess, test_data, test_labels):
    """
    This function is used to generate a series of visualizations
    :param sess: A Tensorflow session
    :param test_data: A collection of testing image file names
    :param test_labels: A collection of testing image mask file names
    :return:
    """
    """Data gathering"""
    u.printf ("Generating visuals...")
    test_labels_real = data_set.load_image_data(test_labels, 0, params.visualize_images, params, binary=True)
    test_data_real= data_set.load_image_data(test_data, 0, params.visualize_images, params)

    """Integrity check"""
    params.check_visualize_image(len(test_labels_real))

    """Visualize"""
    vs.visualize_map(sess, test_data_real, test_labels_real, cnn.image_size, data_set.dp,
                     prediction_true_values, x, y, keep_prob, params)
    vs.visualize_data(sess, test_data_real, test_labels_real, prediction_round_values, x, y, keep_prob,
                      data_set.dp, params)



def save_model(sess):
    """
    This function is used to save all related model data to a series of files
    :param sess: The tensorflow session to save
    :return: None
    """
    model_saver = tf.train.Saver()
    model_saver.save(sess, params.saved_model_file_path)


def save_variables(test_data):
    """
    This function is used to save all related variables to a text file
    :return: None
    """
    try:
        with open(params.saved_variable_file_path, 'w') as csvfile:
            writer = csv.writer(csvfile,delimiter=",")
            writer.writerow([params.number_of_classes,params.neuron_multiplier,params.convolutional_filter,params.convolutional_layer_count])
        with open(params.saved_images_file_name, 'w') as csvfile:
            writer = csv.writer(csvfile,delimiter=",")
            for i in xrange(params.visualize_images):
                writer.writerow(test_data[i])
    except:
        u.printf("Unable to write to variable file: {}".format(params.saved_variable_file_path))


if __name__ == '__main__':
    u.printf("Executing version {} of the code".format(version))
    u.printf ("Beginning model execution with the following settings\n" \
          "Convolutional-Layers:{} Epoch-Count:{} Image-Size:{} Batch-Size:{} Learning-Rate:{} Drop-Out:{} "
              "Display-Interval:{} Neuron Multiplier:{} Convolutional Filter:{} CNN:{} Random-Seed:{}" \
          .format(params.convolutional_layer_count, params.epochs, params.image_size, params.training_batch_size,
                  params.learning_rate, params.dropout, params.training_display, params.neuron_multiplier, params.convolutional_filter,
                  params.cnn, params.random_seed))

    with tf.Session() as sess:
        iteration=0
        visual_paths=[]
        for train_indices, test_indices in splits:
            """This exists to work around 4 hour run limit"""
            if iteration != params.current_fold:
                iteration += 1
                continue
            u.printf("Running {} fold...".format(iteration))

            train_network(sess, np.take(data, train_indices), np.take(labels, train_indices), iteration)
            test_network(sess, np.take(data, test_indices), np.take(labels, test_indices) )

            break
        generate_visuals(sess, np.take(data, test_indices), np.take(labels, test_indices) )

        if params.save_model:
            save_model(sess)
            save_variables(np.take(data, test_indices))


    u.printf ("Finished model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
