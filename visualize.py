import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_map(sess, test_images, test_labels, image_size, dataset_params, prediction_true_values, x, y, keep_prob, model_params):
    """
    This function is used to generate a series of visual maps
    :param sess: A tensorflow session to use
    :param test_images: An array of test images
    :param test_labels: An array of test binary maps
    :param image_size: The size of the images
    :param dataset_params: A dataset parameters object
    :param prediction_true_values: A cost function to run
    :param x: Tensorflow images parameter
    :param y: Tensorflow label data parameter
    :param keep_prob: Regularization term
    :param model_params: Model parameters object
    :return: None
    """

    for position in range(model_params.visualize_images):
        image = test_images[position:position + 1]
        label = test_labels[position:position + 1]
        map_list = []
        map_response = sess.run([prediction_true_values], feed_dict={x: image, y: label, keep_prob: 1.})
        if model_params.cnn:
            map_list.append(np.reshape(map_response,[image_size, image_size]))
        else:
            map_list.append(map_response[0][0])

        map_visualized = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), map_list))

        for visual, original in zip(map_visualized, image):
            plt.figure(figsize=(3.56, 3.56), dpi=100)
            plt.axis('off')
            plt.imshow(1 - np.resize(original, [image_size, image_size]), cmap='gray_r')
            plt.imshow(np.resize(visual, [image_size, image_size]), cmap='jet', alpha=0.35, interpolation='none', vmin=0, vmax=1)
            cmap_file = '{}/map_{}_jet.{}'.format(dataset_params.output_dir, position, dataset_params.output_file_ext)
            plt.savefig(cmap_file)
            plt.close()
        #plt.figure(figsize=(3.56, 3.56), dpi=100)
        plt.imsave('{}/map_{}_original.{}'.format(dataset_params.output_dir, position, dataset_params.output_file_ext), np.reshape(image,(model_params.image_size,model_params.image_size)), cmap='gray', dpi=72)


def visualize_data(sess, test_images, test_labels, prediction_round_values, x, y, keep_prob, dataset_params, model_params):
    """
    This function is used to produce output files in the form of raw text
    :param sess: A tensorflow session to use
    :param test_images: An array of test images
    :param test_labels: An array of test binary maps
    :param prediction_round_values: A cost function to run that is rounded afterwars
    :param x: Tensorflow images parameter
    :param y: Tensorflow label data parameter
    :param dataset_params: A dataset parameters object
    :param model_params: Model parameters object
    :return:
    """
    for position in range(model_params.visualize_images):
        image = test_images[position:position + 1]
        label = test_labels[position:position + 1]

        predictions = sess.run([prediction_round_values], feed_dict={x: image, y: label, keep_prob: 1.})
        plt.imsave('{}/map_{}_label.pdf'.format(dataset_params.output_dir, position), np.reshape(label, (model_params.image_size,model_params.image_size)), cmap='gray', dpi=72)
        plt.clf()
        plt.imsave('{}/map_{}_prediction.pdf'.format(dataset_params.output_dir, position), np.reshape(predictions, (model_params.image_size,model_params.image_size)), cmap='gray', dpi=72)
        plt.clf()

def plot_model_training(training_values, data_label, parameters):
    """
    This function is used to create a series of graphs representing training efficiency
    :param training_values: The training values to plot
    :param data_label: The type of graph being produced
    :param parameters: DataSet parameters to use for determining output director
    :return: None
    """
    plt.title("Model {} Graph".format(data_label))
    plt.plot(zip(*training_values)[0], '-b', label="Training data", ls=':')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("{}".format(data_label))
    plt.savefig("{}/{}.{}".format(parameters.output_dir,data_label,parameters.output_graph_ext))
    plt.clf()


def calculate_test_accuracy(predctions, labels):
    """
    This function is used to calculate testing accuracies
    :param predctions: A numpy array of predicted binary values
    :param labels: A numpy array of true binary values
    :return: A tuple containing 4 floats: accuracy, precision, recall, f1
    """
    """Metrics"""
    true_positive = np.count_nonzero(predctions * labels)
    true_negative = np.count_nonzero((predctions - 1) * (labels - 1))
    false_positive = np.count_nonzero(predctions * (labels - 1))
    false_negative = np.count_nonzero((predctions - 1) * labels)

    """Calculations"""
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    accuracy = float(true_positive+true_negative)/float(true_positive+true_negative+false_positive+false_negative)
    if true_positive+false_positive > 0:
        precision = float(true_positive)/float(true_positive+false_positive)
    if true_positive+false_negative > 0:
        recall = float(true_positive)/float(true_positive+false_negative)
    if recall > 0 or precision > 0:
        f1 = float(2*((float(precision) * float(recall)) / (precision + recall)))

    return (accuracy, precision, recall, f1, true_positive, true_negative, false_positive, false_negative)
