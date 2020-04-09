import cv2
import glob
import numpy as np
import tqdm
import os
import subprocess
import shutil

#parametreler
output_dir = "/home/ws2080/Desktop/data/training/predicted_tta/"
image_path = "/home/ws2080/Desktop/data/training/test/"

temp_dir = "tmp/"
temp_augmented_image_dir = temp_dir + "augmented_images/"
temp_prediction_dir = temp_dir + "predictions/"
temp_prediction_original_form_dir = temp_dir + "original_image_predictions/"
ensemble_type = "affirmative"
current_model = "faster" # "retinanet" "cascade"

if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

if not os.path.exists(temp_augmented_image_dir):
    os.mkdir(temp_augmented_image_dir)

if not os.path.exists(temp_prediction_dir):
    os.mkdir(temp_prediction_dir)

if not os.path.exists(temp_prediction_original_form_dir):
    os.mkdir(temp_prediction_original_form_dir)


'''
bounding_box_path = "test_bbox/"
frame_path = "frames/"
predicted_bb_path = "GT_predictions/predicted/"
train_path = "train/"
eliminated_prediction_path = "eliminated_predictions/"
svm_eliminated_prediction_path = "svm_eliminated_predictions/"
augmented_train_path = "augmented_train/"
'''
def get_coordinates(x, y, w, h):
    x1 = (x - w / 2.)
    x2 = ((x - w / 2.) + w)
    y1 = (y - h / 2.)
    y2 = ((y - h / 2.) + h)

    return x1, x2, y1, y2

def get_cartesian(im_width, im_height, x1, x2, y1, y2):
    x1_c = int(np.round(x1 - im_width/2))
    x2_c = int(np.round(x2 - im_width / 2))
    y1_c = int(np.round(-(y1 - im_height/2)))
    y2_c = int(np.round(-(y2 - im_height/2)))
    return x1_c, x2_c, y1_c, y2_c

def get_original_coordinates(im_width, im_height, x1, x2, y1, y2):
    x1_o = int(np.round(x1 + im_width/2))
    x2_o = int(np.round(x2 + im_width / 2))
    y1_o = int(np.round(im_height/2 - y1))
    y2_o = int(np.round(im_height/2 - y2))

    return x1_o, x2_o, y1_o, y2_o


def rotate(x, y, angle=90):
    rad = np.pi * angle / 180
    x_rot = x * np.cos(rad) - y * np.sin(rad)
    y_rot = x * np.sin(rad) + y * np.cos(rad)

    return x_rot, y_rot


def rotate_box_normalize(im_width, im_height, x1, x2, y1, y2, degree=90):

    x1_c, x2_c, y1_c, y2_c = get_cartesian(im_width, im_height, x1, x2, y1, y2)

    p1_x, p1_y = rotate(x1_c, y1_c, degree)
    p4_x, p4_y = rotate(x2_c, y2_c, degree)

    if degree % 180 != 0:
        new_width, new_height = im_height, im_width
    else:
        new_width, new_height = im_width, im_height

    p1_x, p4_x, p1_y, p4_y = get_original_coordinates(new_width, new_height, p1_x, p4_x, p1_y, p4_y)

    return p1_x, p4_x, p1_y, p4_y, new_height, new_width


def flip_vertical(im_width, im_height, x1, x2, y1, y2):
    x1_c, x2_c, y1_c, y2_c = get_cartesian(im_width, im_height, x1, x2, y1, y2)

    x1_c = -x1_c
    x2_c = -x2_c

    x1_c, x2_c, y1_c, y2_c = get_original_coordinates(im_width, im_height, x1_c, x2_c, y1_c, y2_c)

    return x1_c, x2_c, y1_c, y2_c

im_sizes = {}

def augment_test(test_path, degrees=[0], flip=True):
    image_path = test_path
    test_image = cv2.imread(image_path)
    im_height, im_width, _ = test_image.shape
    im_sizes[image_path.split("/")[-1]] = (im_height, im_width)
    root_filename = image_path.split('/')[-1].split('.')[0]

    #create the folders
    for i in degrees:
        folder = temp_augmented_image_dir + str(i) + "&flipped/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        folder = temp_augmented_image_dir + str(i) + "&notflipped/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        folder = temp_prediction_dir + str(i) + "&flipped/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        folder = temp_prediction_dir + str(i) + "&notflipped/"
        if not os.path.exists(folder):
            os.mkdir(folder)

    #save rotated and flipped images
    for i in range(len(degrees)):
        cur_degree = degrees[i]
        rotation_count = int(np.round(degrees[i] / 90))
        image = test_image
        for j in range(rotation_count):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(temp_augmented_image_dir + str(cur_degree) + "&notflipped/" + root_filename + ".jpg", image)
        if flip:
            cv2.imwrite(temp_augmented_image_dir + str(cur_degree) + "&flipped/" + root_filename + ".jpg", cv2.flip(image, flipCode=1))

    return

def make_predictions():
    # Some basic setup:
    # Setup detectron2 logger
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common libraries
    import numpy as np
    import glob
    import os
    import cv2
    import random
    from google.colab.patches import cv2_imshow

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.structures import BoxMode

    from detectron2.config import get_cfg

    class_names = ["specularity", "saturation", "artifact", "blur", "contrast", 
                   "bubbles", "instrument", "blood"]
    
    model_pth=""
    if current_model == "retinanet":    
        model_path = "/home/ws2080/Desktop/codes/detectron/model_retinanet/output_32"
        model_pth = "model_0059999.pth"
    elif current_model == "faster":    
        model_path = "/home/ws2080/Desktop/codes/detectron/model_faster_rcnn_R_50_FPN_3x/output_9"
        model_pth = "model_0139999.pth"
    elif current_model == "cascade":    
        model_path = "/home/ws2080/Desktop/codes/detectron/model_cascade_mask_rcnn_R_50_FPN_3x/output_24"
        model_pth = "model_0059999.pth"
        
    cfg = get_cfg()

    cfg.merge_from_file(model_path+"/config.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(model_path, model_pth)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1   

    cfg.DATASETS.TEST = ("ead_validation_1",)

    predictor = DefaultPredictor(cfg)
    
    #make predictions
    for image_folder in glob.glob(temp_augmented_image_dir+"*/"):
        prediction_output_dir = temp_prediction_dir + image_folder.split("/")[-2] + "/"
        #prediction image_folderin içindeki resimler için yapılıp prediction_output_dir'in içine atılmalı
        for im_path in glob.glob(image_folder + "*.jpg"):
            im_name = im_path.split("/")[-1].split(".")[0]
            saved_image_path = prediction_output_dir + im_name + ".txt"
            # Burası sana kalmış abi, predict edip saved_image_path olarak kaydetmek lazım sadece
            
            im = cv2.imread(im_path)
            outputs = predictor(im)   

            total_detection = len(outputs["instances"])

            temp_detection_list = []
            detections = outputs["instances"]

            for i in range(total_detection):
                temp_detection = class_names[int(detections.pred_classes[i])]+" "+ str(float(detections.scores[i]))+" "+str(float(detections.pred_boxes.tensor[i,0]))+" "+ str(float(detections.pred_boxes.tensor[i,1])) +" "+str(float(detections.pred_boxes.tensor[i,2]))+" "+ str(float(detections.pred_boxes.tensor[i,3]))
                temp_detection_list.append(temp_detection)

            with open(saved_image_path, 'w') as f:
                    for item in temp_detection_list:
                        f.write("%s\n" % item)       
                        
            #print("Saved->" + saved_image_path)

def main():
    degrees = [0, 90, 180, 270]

    for filename in tqdm.tqdm(glob.glob(image_path + '*.jpg')):
        augment_test(filename, degrees=degrees)

    make_predictions()

    for prediction_folder in glob.glob(temp_prediction_dir + "*/"):
        folder_name = prediction_folder.split("/")[-2]

        if not os.path.exists(temp_prediction_original_form_dir + folder_name):
            os.mkdir(temp_prediction_original_form_dir + folder_name)

        aug_type = prediction_folder.split("/")[-2].split(".")[0].split("&")
        deg = int(aug_type[0])
        is_flipped = aug_type[1]



        for pred_file in glob.glob(prediction_folder + "*.txt"):
            im_name = pred_file.split("/")[-1].split(".")[0]
            im_height, im_width = im_sizes[im_name + ".jpg"]
            if deg % 180 != 0:
                new_width, new_height = im_height, im_width
            else:
                new_width, new_height = im_width, im_height
            new_content = ""
            with open(pred_file, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')
                    clsnum, confidence, x1, y1, x2, y2 = line
                    x1 = int(float(x1))
                    x2 = int(float(x2))
                    y1 = int(float(y1))
                    y2 = int(float(y2))

                    if is_flipped == "flipped":
                        x1, x2, y1, y2 = flip_vertical(new_width, new_height, x1, x2, y1, y2)

                    x1n, x2n, y1n, y2n, hn, wn = rotate_box_normalize(new_width, new_height, x1, x2, y1, y2, (360-deg) % 360)
                    x1n = round(x1n, 2)
                    x2n = round(x2n, 2)
                    y1n = round(y1n, 2)
                    y2n = round(y2n, 2)

                    new_content += clsnum + " " + confidence + " " + str(min(x1n, x2n))+".00" + " " + str(min(y1n, y2n))+".00" + " " + str(max(x1n, x2n))+".00" + " " + str(max(y1n, y2n))+".00" + "\n"
            with open(temp_prediction_original_form_dir + folder_name + "/" + im_name + ".txt", "w") as f:
                f.write(new_content)
    cmd = ["python", "ensemble_format_eval.py", temp_prediction_original_form_dir, output_dir, image_path, ensemble_type, str(0.5)]
    subprocess.Popen(cmd).wait()
#    try:
#        shutil.rmtree(temp_dir)
#    except OSError as e:
#        print("Error: %s - %s." % (e.filename, e.strerror))
    return

if __name__ == "__main__":
    main()
