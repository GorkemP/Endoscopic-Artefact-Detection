import glob
import numpy as np
import argparse
import shutil

import time

parser = argparse.ArgumentParser()
parser.add_argument("prediction_dir", help="Predictions directory (must end with /)")
parser.add_argument("output_dir", help="Output dİrectory (must end with /)")

args = parser.parse_args()

prediction_folder_path = args.prediction_dir
output_folder_path = args.output_dir

predictionPaths = glob.glob(prediction_folder_path+'*.txt')

class_names = ["specularity", "saturation", "artifact", "blur", "contrast", 
               "bubbles", "instrument", "blood"]

IoUs_path = "files/class_iou_1_5.txt"

with open(IoUs_path) as f:
    contents = f.readlines()

class_thresholds_8 = {"specularity": 0.159,
                      "saturation": 0.194,
                      "artifact": 0.217,
                      "blur": 0.103, 
                      "contrast":0.137,                      
                      "bubbles": 0.2, 
                      "instrument": 0.219, 
                      "blood": 0.119
                        }
#class_thresholds_8[target_list[-1][0]]
# class_thresholds[target_list[-1][0]][contents[counter][0]]
hard_threshold = 0.4
class_thresholds  = {}
temp_dict = {}

for i in range(len(contents)):
        
    main_class = i // 8
    sub_class = i-8*main_class
        
    content = float(contents[i].split(" ")[1][0:7])
    temp_dict[class_names[sub_class]]=content
    
    if (i%8 == 7):
        class_thresholds[class_names[main_class]]= temp_dict
        temp_dict = {}

def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

total_time=0
for predictionPath in predictionPaths:
    txt_file_name = predictionPath.split("/")[-1]
    target_list = []

    contents = np.loadtxt(predictionPath, dtype=str)
    
    start = time.time()

    if (contents.ndim == 1) or (len(contents) == 0):
        shutil.copy(predictionPath, output_folder_path)
        continue
        
    contents = contents[np.argsort(contents[:, 1])][::-1]
    
    while (len(contents) > 0):
        target_list.append(contents[0])
        contents = np.delete(contents, 0, 0)
        
        contents_size = len(contents)
        counter = 0
        while (counter < contents_size):
            if (IoU(target_list[-1][2:].astype(float), contents[counter][2:].astype(float)) > hard_threshold):
                contents = np.delete(contents, counter, 0)
                counter = counter-1
                contents_size = contents_size-1
            counter = counter+1
    
    total_time += time.time()-start

    temp_detection_list = []
    for i in range(len(target_list)):
            temp_detection = target_list[i][0]+" "+ target_list[i][1]+" "+target_list[i][2]+" "+ target_list[i][3] +" "+target_list[i][4]+" "+ target_list[i][5]
            temp_detection_list.append(temp_detection)   
            
    with open(output_folder_path+txt_file_name, 'w') as f:
        for item in temp_detection_list:
            f.write("%s\n" % item)

print("Average inference time: "+str(total_time/len(predictionPaths)))
