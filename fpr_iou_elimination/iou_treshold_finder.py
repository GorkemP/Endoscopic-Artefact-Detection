import cv2
import numpy as np
import PIL
from matplotlib import pyplot as plt
import glob
import os
from scipy.stats import iqr as iqr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="folder path (not end with /)")
parser.add_argument("train_path", help="traib path (must end with /)")
parser.add_argument("iqr_value", help="iqr")

args = parser.parse_args()
path = args.path
train_path = args.train_path
iqr_value = args.iqr_value


bounding_box_path = train_path
frame_path = train_path
class_file_name = path + '/class_list.txt'


def create_folder(path, name):
    try:
        os.makedirs(path + name)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)


def getCoordinates(x, y, w, h):
    x1 = (x - w / 2.)
    x2 = ((x - w / 2.) + w)
    y1 = (y - h / 2.)
    y2 = ((y - h / 2.) + h)

    return x1, x2, y1, y2


def getBoxValues(x1, y1, w1, h1, x2, y2, w2, h2):
    x1a, x2a, y1a, y2a = getCoordinates(x1, y1, w1, h1)
    x1b, x2b, y1b, y2b = getCoordinates(x2, y2, w2, h2)

    boxA = [x1a, x2a, y1a, y2a]
    boxB = [x1b, x2b, y1b, y2b]

    return boxA, boxB


def calculateIOU(item1, item2):
    x1 = item1[0]
    y1 = item1[1]
    w1 = item1[2]
    h1 = item1[3]

    x2 = item2[0]
    y2 = item2[1]
    w2 = item2[2]
    h2 = item2[3]

    boxA, boxB = getBoxValues(x1, y1, w1, h1, x2, y2, w2, h2)

    xA = max(boxA[0], boxB[0])
    xB = min(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
    boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    if interArea == 0:
        return 0

    iou = (interArea / float(boxAArea + boxBArea - interArea))

    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def float_number(s):
    return float(s)


def readTXT(txt_name, image_name):
    classnumber = []
    coordinates = []
    with open(txt_name, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(' ')
            clsnum, x, y, w, h = line

            x = float_number(x)
            y = float_number(y)
            w = float_number(w)
            h = float_number(h)

            image_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            m, n = image_gray.shape

            real_x = int(n * x)
            real_w = int(n * w)
            real_y = int(m * y)
            real_h = int(m * h)

            classnumber.append(clsnum)
            coordinates.append([real_x, real_y, real_w, real_h])
            # coordinates.append([x,y,w,h])
    return classnumber, coordinates


def main():
    
    create_folder(path, '/iou_values')
    create_folder(path, '/figures')

    txt_name = []
    image_name = []

    for filename in sorted(glob.glob(bounding_box_path + '*.txt')):
        txt_name.append(filename)

    for filename in sorted(glob.glob(frame_path + '*.jpg')):
        image_name.append(filename)

    assert len(txt_name) == len(image_name)

    classnumber = []
    coordinates = []
    
    for i in range(len(txt_name)):
        clsn, coor = readTXT(txt_name[i], image_name[i])
        classnumber.append(clsn)
        coordinates.append(coor)

    print('cls :', len(classnumber))
    print('coor :', len(coordinates))

    intersection_of_same_class = [[] for i in range(8)]
    index = -1
    for element in coordinates:
        lookup = []
        index += 1
        for i in range(0, len(element)):
            item1 = element[i]
            for j in range(0, len(element)):
                item2 = element[j]
                if i == j or (i, j) in lookup or (j, i) in lookup or classnumber[index][i] != classnumber[index][j]:
                    continue
                lookup.append((i, j))
                a = calculateIOU(item1, item2)
                if a > 0:
                    intersection_of_same_class[int(classnumber[index][i])].append(a)

    class_list = []
    iqr_array = []

    for item in intersection_of_same_class:
        if len(item) == 0:
            iqr_array.append((0, 1))
            continue
        s = np.array(item, dtype="float64")
        data = np.sort(s, kind="quicksort")
        # First quartile (Q1)
        Q1 = np.percentile(data, 25, interpolation='midpoint')

        # Third quartile (Q3)
        Q3 = np.percentile(data, 75, interpolation='midpoint')

        # Interquaritle range (IQR)
        IQR = Q3 - Q1
        lower_bound = Q1 - float(iqr_value) * IQR
        upper_bound = Q3 + float(iqr_value) * IQR
        i = iqr(data)
        iqr_array.append((lower_bound, upper_bound))
    
    with open('iqr_log_iou.txt', 'w') as f:
        for item in iqr_array:
            f.write(str(item) + '\n')
    
    

if __name__ == "__main__":
    main()
