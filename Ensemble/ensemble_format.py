import generateXML
import numpy as np
import cv2 as cv
import glob
import os
import subprocess
from lxml import etree
import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("prediction_dir", help="The directory that contains the folders with the model names, each having 3 subfolders detection_bbox, generalization_bbox and sequence_bbox(must end with /)")
parser.add_argument("output_dir", help="The directory which will contain the ensemble preditions(must end with /)")
parser.add_argument("image_dir", help="The directory that contains the folders Detection, Detection_sequence and Generalization, which contain the images(must end with /)")
parser.add_argument("ensemble_type", help="The ensembling strategy(affirmative, consensus or unanimous)")
parser.add_argument("threshold", help="if 2 bounding boxes overlap with an iou value higher than this threshold(0-1) their ensemble is taken as the boudning box", type=float)


args = parser.parse_args()

'''output_directory = "ead_test_output/"
predictions_dir = "ead_model_predictions/"
image_directory = "ead_test_images/"
ensemble_output_dir = output_directory + "ensemble_output/"
'''

ensemble_threshold = args.threshold
prediction_folder_names = ["detection_bbox", "generalization_bbox", "sequence_bbox"]
test_image_folder_names = ["Detection", "Generalization", "Detection_sequence" ]
ensemble_types = ["affirmative", "consensus", "unanimous"]
output_directory = args.output_dir

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

predictions_dir = args.prediction_dir
image_directory = args.image_dir
temp_dir = "temp/"
ensemble_type = args.ensemble_type

#input checks
if ensemble_type not in ensemble_types:
    print("The ensemble type should be either affirmative, consensus or unanimous")
    exit(-1)

for i in range(len(prediction_folder_names)):
    prediction_type = prediction_folder_names[i]
    for model_folder in glob.glob(predictions_dir + "*/"):
        folder = model_folder + prediction_type + "/"
        if not os.path.exists(folder):
            print("Prediction folder does not contain required directories: detection_bbox, generalization_bbox, sequence_bbox")
            exit(-1)

if not os.path.exists(image_directory):
    print("Image directory does not exist")
    exit(-1)

for image_folder in test_image_folder_names:
    folder = image_directory + "/" + image_folder + "/"

    if not os.path.exists(folder):
        print("Image folder does not contain required directories: Detection, Generalization, Detection_sequence")
        exit(-1)


if not os.path.exists(predictions_dir):
    print("Prediction directory does not exist")
    exit(-1)


def ead_to_eod(ead_file_path, ead_image_path, save_path):

    file_name = ead_file_path.split('/')[-1]

    image = cv.imread(ead_image_path, cv.IMREAD_COLOR)
    im_height, im_width, im_depth = image.shape
    boxes = []
    with open(ead_file_path, "r") as f:
        for l in f:
            ln = l.strip('\n')
            line = ln.split(' ')
            clsnum, confidence, x1, y1, x2, y2 = line


            box = [clsnum, x1, y1, x2, y2, confidence]
            boxes.append(box)
    a = generateXML.generateXML(file_name, save_path[:-1], im_width, im_height, im_depth, boxes)
    with open(save_path + file_name.split('.')[0] + ".xml", "w") as e:
        e.write(a)


def main():

    for i in range(len(prediction_folder_names)):
        ensemble_output_dir = output_directory
        prediction_type = prediction_folder_names[i]
        test_image_folder = test_image_folder_names[i]

        for model_folder in tqdm.tqdm(glob.glob(predictions_dir + "*/"), desc="From EAD to EOD format: " + prediction_type):

            folder = model_folder + prediction_type
            folder_name = folder.split("/")[-2]
            ensemble_output_dir = ensemble_output_dir + folder_name + "_"

            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)

            if not os.path.exists(temp_dir + folder_name):
                os.mkdir(temp_dir + folder_name)

            for file in glob.glob(folder+ "/" + "*.txt"):
                image_file_name = file.split('/')[-1].split('.')[0] + ".jpg"
                ead_to_eod(file, image_directory + test_image_folder + "/" + image_file_name, temp_dir + folder_name + "/")
        print("Ensembling...")
        cmd = ['python', "main.py", "-d", temp_dir, "-o", ensemble_type, "-t", str(ensemble_threshold)]
        subprocess.Popen(cmd).wait()
        ensemble_output_dir = ensemble_output_dir + ensemble_type + "_" + str(ensemble_threshold).split('.')[-1] + "/"
        if not os.path.exists(ensemble_output_dir):
            os.mkdir(ensemble_output_dir)

        if not os.path.exists(ensemble_output_dir + prediction_type):
            os.mkdir(ensemble_output_dir + prediction_type)

        for file in tqdm.tqdm(glob.glob(temp_dir + "output/*.xml"), desc="From EOD to EAD format:" + prediction_type):
            new_pred_file = ensemble_output_dir + prediction_type + "/" + file.split('/')[-1].split('.')[0] + ".txt"

            with open(new_pred_file, "w") as f:
                file_content_str = ""
                folder_name = folder.split("/")[-2]

                doc = etree.parse(file)
                filename = doc.getroot()  # find the root
                objects = filename.findall("object")
                for j in range(len(objects)):
                    name = objects[j].find("name").text
                    ymax = float(objects[j].find("bndbox").find("ymax").text)
                    ymin = float(objects[j].find("bndbox").find("ymin").text)
                    xmax = float(objects[j].find("bndbox").find("xmax").text)
                    xmin = float(objects[j].find("bndbox").find("xmin").text)
                    prob = "{0:.2f}".format(float(objects[j].find("confidence").text))
                    file_content_str += name + " " + str(prob) + " " + str(xmin) + " " + str(ymin) + " " + str(
                        xmax) + " " + str(ymax) + "\n"
                f.write(file_content_str)
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    return

    '''for folder in tqdm.tqdm(glob.glob(predictions_dir + "*/")):
        folder_name = folder.split("/")[-2]
        if not os.path.exists(output_directory + folder_name):
            os.mkdir(output_directory + folder_name)

        for file in glob.glob(folder + "*.txt"):
            image_file_name = file.split('/')[-1].split('.')[0] + ".jpg"
            ead_to_eod(file, image_directory + image_file_name, output_directory + folder_name + "/")

    cmd = ['python', "main.py", "-d", output_directory, "-o", "consensus"]
    subprocess.Popen(cmd).wait()

    if not os.path.exists(ensemble_output_dir):
            os.mkdir(ensemble_output_dir)


    for file in tqdm.tqdm(glob.glob(output_directory + "output/*.xml")):
        new_pred_file = ensemble_output_dir + file.split('/')[-1].split('.')[0] + ".txt"

        with open(new_pred_file, "w") as f:
            file_content_str = ""
            folder_name = folder.split("/")[-2]

            doc = etree.parse(file)
            filename = doc.getroot()  # find the root
            objects = filename.findall("object")
            for j in range(len(objects)):
                name = objects[j].find("name").text
                ymax = float(objects[j].find("bndbox").find("ymax").text)
                ymin = float(objects[j].find("bndbox").find("ymin").text)
                xmax = float(objects[j].find("bndbox").find("xmax").text)
                xmin = float(objects[j].find("bndbox").find("xmin").text)
                prob = "{0:.2f}".format(float(objects[j].find("confidence").text))
                file_content_str += name + " " + str(prob) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n"
            f.write(file_content_str)
    return'''

if __name__ == "__main__":
    main()