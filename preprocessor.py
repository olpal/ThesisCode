import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys
import numpy as np
from scipy import ndimage
from PIL import Image
import util as u
import datetime as dt
from skimage import draw
import csv

xsize=256
ysize=256
xoverlap=80
yoverlap=80
image_count=1
imageCell=1
mask_threshold=50

#inputDirectory="/scratch/aolpin/thesis/results/log/2017-10-25--00-25-28-853857160/"
#outputDirectory="/scratch/aolpin/thesis/dataset2/"
#datasetDirectory="/scratch/aolpin/thesis/dataset3/"

inputDirectory="/Users/aolpin/Documents/School/thesis/datatest/"
outputDirectory="/Users/aolpin/Documents/School/thesis/datatest/data/"
datasetDirectory="/Users/aolpin/Documents/School/thesis/datatest/dataout/"

data_files=True
overlay_images=True
write_original=False
mode=1


def plot_image_rotate(slices, slices_rotated, in_image, in_image_mask, file_name):
    """
    This function is used to slice up an image and generate a series of smaller images
    :param slices: The coordinates for slices to be taken
    :param slices_rotated: The coordinates for slices of the rotated image
    :param in_image: The image to slice up
    :param in_image_mask: The image mask to slice up
    :return: None
    """
    np.set_printoptions(threshold=np.inf)
    global imageCell
    #Rotation position
    image_process = in_image
    mask_process = in_image_mask
    position=1
    while position <= 4:
        imageSlices = slices
        #If it has been rotated 90 or 270 degrees
        if position % 2 ==0:
            imageSlices = slices_rotated

        if write_original == True:
            plt.imshow(image_process, cmap="gray")
            plt.imsave('{}original/{}_{}.png'.format(outputDirectory,file_name.replace(".bmp",""),imageCell), image_process, cmap='gray')

        for imageSlice in imageSlices:
            saveFileName = ("img_{}".format(imageCell))
            imageCell += 1

            sliced_image = image_process[imageSlice[0]:imageSlice[1], imageSlice[2]:imageSlice[3]]
            sliced_image_mask = mask_process[imageSlice[0]:imageSlice[1], imageSlice[2]:imageSlice[3]]

            plt.imshow(sliced_image,cmap="gray")
            plt.imsave('{}images/{}_{}.png'.format(outputDirectory,file_name.replace(".bmp",""),saveFileName), sliced_image, cmap='gray')
            plt.clf()

            if data_files:
                labelfile = open('{}labels/{}_{}.txt'.format(outputDirectory,file_name.replace(".bmp",""),saveFileName), 'w')
                reshaped = np.reshape(sliced_image_mask, (65536,))
                labelfile.write("{}\n".format((np.array_str(reshaped, max_line_width=1000000))))
                labelfile.close()
            else:
                plt.imshow(sliced_image_mask, cmap="gray")
                plt.imsave('{}masks/{}_{}.png'.format(outputDirectory,file_name.replace(".bmp",""),saveFileName), sliced_image_mask, cmap='gray')
                plt.clf()

            if overlay_images:
                if data_files:
                    sliced_image = np.reshape(sliced_image, (65536,))
                    for i in xrange(65536):
                        if reshaped[i] == 1:
                            sliced_image[i] = 1
                    plt.imsave('{}overlays/{}_overlay.png'.format(outputDirectory, saveFileName),
                               np.reshape(sliced_image, (256, 256)), cmap='gray')
                else:
                    plt.imshow(sliced_image, cmap="gray")
                    plt.imshow(sliced_image_mask, cmap="jet", alpha=0.5)
                    plt.savefig('{}overlays/{}_overlay.png'.format(outputDirectory,file_name.replace(".bmp",""),saveFileName))
                    plt.clf()

       #Rotate the matrices by 90 degrees
        image_process = image_process.swapaxes(-2, -1)[..., ::-1]
        mask_process = mask_process.swapaxes(-2, -1)[..., ::-1]
        position+=1

def slice_image(inimage):
    """
    This function is used to generate a series of slices
    :param inimage: The image that slices will be generated for
    :return: an array of tuples of slices where the values are: ystart, yend, xstart, xend
    """
    slices = []
    shape = inimage.shape
    yStartPos=0
    finalY=False
    while True:
        finalX=False
        xStartPos = 0
        while True:
            slices.append((yStartPos, (yStartPos + ysize), xStartPos, (xStartPos + xsize)))
            xStartPos = (xStartPos + (xsize-xoverlap))
            if (xStartPos+xsize) > shape[1] and finalX:
                break
            elif (xStartPos+xsize) > shape[1]:
                xStartPos = (shape[1] - xsize)
                finalX=True
        yStartPos = (yStartPos+(ysize - yoverlap))
        if (yStartPos+ysize) > shape[0] and finalY:
            break
        elif (yStartPos+ysize) > shape[0]:
            yStartPos = (shape[0] - ysize)
            finalY = True

    return slices

def rename_files():
    """
    This function is used to rename the extention on images
    :return: None
    """
    for file in glob.glob('{}{}/*_.bmp'.format(inputDirectory,'images')):
        os.rename(file,file.replace("_.bmp",".bmp"))

def load_images(input_file_name):
    """
    This function loads an image and a mask file
    :param input_file_name: The name of the image file to load
    :return: an array of image data, an array of mask data
    """
    image = plt.imread(input_file_name)

    return image

def load_masks(mask_file_name):
    """
    This function loads an image and a mask file
    :param mask_file_name: The name of the mask file to load
    :return: an array of image data, an array of mask data
    """

    image_mask = Image.open(mask_file_name).convert('L')

    img_mask_data = np.asarray(image_mask)
    image_threshold = (img_mask_data > mask_threshold) * 1.0

    return image_threshold

def load_data(data_file_name):
    array = ''
    with open(data_file_name, 'r') as label_file:
        for line in label_file:
            data = line.replace("[", "")
            data = data.replace("]", "")
            data = data.replace("\n", "")
            array += data
    return np.fromstring(array, dtype=int, sep=' ')

def rename_match():
    """
    This fuction is used to rename a collection of individual series of images into a single unified collection
    :return: None
    """
    image_files = glob.glob('{}{}/*.png'.format(outputDirectory, 'images'))
    index=1
    for i in xrange(len(image_files)):

        image_file_name="{}{}/img_{}.{}".format(datasetDirectory,"images",index,"png")
        if data_files:
            label_name = (image_files[i].replace("images", "labels")).replace("png","txt")
            label_file_name = "{}{}/img_{}.{}".format(datasetDirectory, "labels", index, "txt")
        else:
            label_name = image_files[i].replace("images", "masks")
            label_file_name="{}{}/img_{}.{}".format(datasetDirectory,"masks",index,"png")
        os.rename(image_files[i],image_file_name)
        os.rename(label_name, label_file_name)
        index+=1

def bounding_boxes():
    """
    This function uses the global variable csv_values to generate a series
    of bounding boxes and writes the data to a text file
    """
    np.set_printoptions(threshold=np.inf)

    if len(files) == 0:
        return

    for input_file in files:
        file_name = os.path.basename(input_file)
        csv_file_name = "{}csv/{}".format(inputDirectory, file_name.replace(".bmp", ".csv"))

        if not os.path.isfile(csv_file_name):
            print("{} does not exist as a file".format(csv_file_name))
            continue

        csv_values = read_csv(csv_file_name)
        in_image =  plt.imread(input_file)
        csv_array = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=int)
        for val in csv_values:
            distance = int(float(val[2]))
            row = int(float(val[1]))
            column = int(float(val[0]))
            rr_left, cc_left = draw.line((row - distance), (column - distance), (row + distance), (column - distance))
            rr_right, cc_right = draw.line((row - distance), (column + distance), (row + distance), (column + distance))
            rr_top, cc_top = draw.line((row - distance), (column - distance), (row - distance), (column + distance))
            rr_bottom, cc_bottom = draw.line((row + distance), (column - distance), (row + distance), (column + distance))

            csv_array[rr_left, cc_left] = 1
            csv_array[rr_right, cc_right] = 1
            csv_array[rr_top, cc_top] = 1
            csv_array[rr_bottom, cc_bottom] = 1

    labelfile = open('{}labels/{}'.format(outputDirectory, file_name.replace(".bmp", ".txt")), 'w')
    reshaped = np.reshape(csv_array, ((in_image.shape[0] * in_image.shape[1]),))
    labelfile.write("{}\n".format((np.array_str(reshaped, max_line_width=1000000))))
    labelfile.close()

def read_csv(filename):
    """
    This function is used to read all lines of a csv file that contains x,y and radius coordinates
    :param filename: The name of the image to look for a csv file for
    :return: none
    """
    csv_values = []
    try:
        with open(filename, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                csv_values.append(row)
    except IOError as e:
        print("No csv value found for file {}".format(filename))

    return csv_values

def produce_images():
    file_count=1
    for input_file in files:
        file_name = os.path.basename(input_file)
        mask_file_name = "{}masks/{}".format(inputDirectory, file_name.replace(".bmp", "_mask.jpg"))
        label_file_name = "{}labels/{}".format(inputDirectory, file_name.replace(".bmp", ".txt"))

        if not os.path.isfile(mask_file_name) and not data_files:
            print("{} does not exist as a file".format(mask_file_name))
            continue

        if not os.path.isfile(label_file_name) and data_files:
            print("{} does not exist as a file".format(label_file_name))
            continue

        image = load_images(input_file)
        if data_files:
            image_mask = load_data(label_file_name)
            image_mask = np.reshape(image_mask,(image.shape[0],image.shape[1]))
        else:
            image_mask = load_masks(mask_file_name)

        imageSlices = slice_image(image)
        imageSliceRotated = slice_image(ndimage.rotate(image,90))
        plot_image_rotate(imageSlices, imageSliceRotated, image, image_mask, file_name)
        if file_count % 10 == 0:
            print ("Processed {} files".format(file_count))
        file_count+=1

if __name__ == '__main__':
    u.printf("Started data processing at: {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    files = []
    if len(sys.argv) > 1:
        print ("Arguments found, loading...")
        mode = int(sys.argv[1])

    if mode == 1:
        files = glob.glob('{}images/*.bmp'.format(inputDirectory))
        produce_images()
        print ("Found {} files".format(len(files)))
    elif mode == 2:
        files.append(sys.argv[2])
        produce_images()
    elif mode ==3:
        rename_match()
    elif mode ==4:
        files = glob.glob('{}images/*.bmp'.format(inputDirectory))
        bounding_boxes()
    elif mode ==5:
        files.append(sys.argv[2])
        bounding_boxes()

    u.printf("Finished data processing at: {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
