from appJar import gui
import parameters as hp
import usemodel as um
import imageprocessing as im
import os

def check_variables(model_directory, input_image, output_directory):
    """
    This function is used to validate all UI fields
    :param model_directory: The path to the model directory to check
    :param input_image: The path to an image to check
    :param output_directory: The path to the output directory to check
    :return: Boolean indicating if validation succeeded
    """
    train_params = hp.ModelParameters()
    variable_full_path = '{}/{}'.format(model_directory,train_params.saved_variable_file_name)
    model_full_path = '{}/{}.index'.format(model_directory,train_params.saved_model_file_name)
    if not os.path.exists(variable_full_path):
        app.setLabel("info", "Unable to locate variable file")
        return False
    if not os.path.exists(model_full_path):
        app.setLabel("info", "Unable to locate model file")
        return False
    if not os.path.exists(input_image):
        app.setLabel("info", "Unable to input image")
        return False
    if not os.path.exists(output_directory):
        app.setLabel("info", "Unable to locate output directory")
        return False
    return True

def run_model(model_directory,input_image,output_directory):
    """
    This function is used to run the model and create a prediction map
    :param model_directory: The directory containing saved model files
    :param input_image: The image to conduct predictions on
    :param output_directory: The directory to save prediction data to
    :return: None
    """
    model = um.Model(model_directory,input_image,output_directory)
    model.execute()
    del model

def run_image_processing(input_file,original_image,output_directory,mode,image_prefix,
                         image_postfix,export_data,export_overlay):
    """
    This function is used to run image processing on a file
    :param input_file: The file to use for image processing
    :param original_image: The original image predictions were done for
    :param output_directory: The directory to save data to
    :param mode: The image processing mode to conduct
    :param image_prefix: String to add before saved output filename
    :param image_postfix: String to add after saved output filename
    :param export_data: Boolean indicating whether to save data file
    :param export_overlay: Boolean indicating whether to save overlay image
    :return: None
    """
    image = im.ImageProcessing()
    image.load_parameters(input_file,original_image,output_directory,mode,image_prefix,image_postfix,export_data,export_overlay)
    image.execute()
    del image


def run_code(btn):
    """
    Executes the prediction and image processing algorithms called by the execute button
    :param btn: The execute button that calls this function
    :return: None
    """
    exec_params = hp.ExecutionParameters()
    app.setLabel("info","Checking variables...")
    if not check_variables(app.getEntry("Model_Directory"),app.getEntry("Input_Image"),app.getEntry("Output_Directory")):
        return
    app.setLabel("info", "Variables valid, running model...")
    run_model(app.getEntry("Model_Directory"), app.getEntry("Input_Image"), app.getEntry("Output_Directory"))
    app.setLabel("info", "Saving image data...")
    input_file = '{}/{}'.format(app.getEntry("Output_Directory"),exec_params.prediction_file_name)
    run_image_processing(input_file,app.getEntry("Input_Image"), app.getEntry("Output_Directory"),1,app.getEntry("image_pre"),
                         app.getEntry("image_post"),app.getSpinBox("Export_Data"),app.getSpinBox("Export_Overlay"))
    app.setLabel("info", "Process complete")

app = gui("Run Model", "400x400")

app.addLabel("l1","Input Image",0,0,3)
app.addFileEntry("Input_Image",1,0,3)
app.addLabel("l2","Model Directory",2,0,3)
app.addDirectoryEntry("Model_Directory",3,0,3)
app.addLabel("l3","Output Directory",4,0,3)
app.addDirectoryEntry("Output_Directory",5,0,3)

app.addLabel("l8","Image Prefix",6,0)
app.addEntry("image_pre",7,0)
app.addLabel("l9","Image Postfix",6,2)
app.addEntry("image_post",7,2)

app.addLabel("l7","Generate Options",8,0,3)
app.addLabel("l5","Image Data File",9,0)
app.addSpinBox("Export_Data", ["True", "False"],10,0)
app.addLabel("l6","Image Overlay",9,2)
app.addSpinBox("Export_Overlay", ["True", "False"],10,2)
app.addLabel("info","Waiting for variable selection...",11,0,3)
app.addButton("Execute",run_code,12,1)
app.go()