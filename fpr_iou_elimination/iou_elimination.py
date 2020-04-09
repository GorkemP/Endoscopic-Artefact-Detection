import cv2
import PIL
import glob
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("prediction_dir", help="Predictions directory(must end with /)")
parser.add_argument("output_dir", help="Output Dİrectory(must end with /)")
parser.add_argument("iqr_txt", help="IQR txt(must be .txt)")

args = parser.parse_args()

predicted_bb_path = args.prediction_dir
eliminated_prediction_path = args.output_dir
iqr_path = args.iqr_txt
gt_bboxes = []
elimination_counts_name = 'eliminated_counts.txt'



treshold = dict()
# treshold = {'specularity' : 0.5, 'saturation' : 0.4, 'artifact' : 0.5, 'blur' : 0.5, 'contrast' : 0.1, 'bubbles' : 0.35, 'instrument' : 0.15, 'blood' : 0.2}
classes = ['specularity',
           'saturation',
           'artifact',
           'blur',
           'contrast',
           'bubbles',
           'instrument',
           'blood'
           ]


def create_folder(path, name):
    try:
        os.makedirs(path + name)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)


def str_to_tuple(string):
    splitted = string.split(",")
    left = splitted[0]
    right = splitted[1]
    left = left[1:]
    right = right[:-1]
    return (float(left), float(right))


with open(iqr_path, "r") as r:
    line = r.read()
    line = line.split("\n")
    for l in range(len(classes)):
        treshold[classes[l]] = str_to_tuple(line[l])


def calculateIOU(boxA, boxB):
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


class_elimination_count = {}
for k in classes:
    class_elimination_count[k] = [0, 0]


def read_predicted(txt_name):
    eliminated_file = eliminated_prediction_path + txt_name.split("/")[-1].split('.')[0] + ".txt"
    p = []
    lookup = []
    deletelist = []
    linelist = []
    with open(txt_name, 'r') as f:
        with open(eliminated_file, "w") as e:
            for l in f:
                ln = l.strip('\n')
                linelist.append(ln)
                line = ln.split(' ')
                clsnum, confidence, x1, y1, x2, y2 = line
                p.append([clsnum, confidence, float(x1), float(y1), float(x2), float(y2)])
            
            for i in range(len(p)):
                class_elimination_count[clsnum][0] += 1
                clsnum, confidence, x1, y1, x2, y2 = p[i]
                for j in range(len(p)):
                    comp_clsnum, comp_confidence, comp_x1, comp_y1, comp_x2, comp_y2 = p[j]
                    if i == j or clsnum != comp_clsnum or (i, j) in lookup or (j, i) in lookup:
                        continue
                    lookup.append((i, j))
                    iou = calculateIOU([x1, x2, y1, y2], [comp_x1, comp_x2, comp_y1, comp_y2])
                    if treshold[clsnum][0] > iou or iou > treshold[clsnum][1]:
                        class_elimination_count[clsnum][1] += 1
                        if p[i][1] < p[j][1]:
                            deletelist.append(i)
                        else:
                            deletelist.append(j)

            for k in range(len(p)):
                if not k in deletelist:
                    e.write(linelist[k])
                    e.write('\n')


def getCoordinates(x, y, w, h):
    x1 = (x - w / 2.)
    x2 = ((x - w / 2.) + w)
    y1 = (y - h / 2.)
    y2 = ((y - h / 2.) + h)

    return x1, x2, y1, y2


def main():
    eliminated_prediction_name = eliminated_prediction_path.split('/')[0]
    create_folder('', eliminated_prediction_name)
    txt_name = []
    for filename in sorted(glob.glob(predicted_bb_path + '*')):
        txt_name.append(filename)

    for item in txt_name:
        read_predicted(item)

    with open(elimination_counts_name, "w") as f:
        for i in class_elimination_count.keys():
            f.write("{}--> Occurence: {}, Elimination: {}, Elimination Percentage: {}\n".format(i,
                                                                                                class_elimination_count[
                                                                                                    i][0],
                                                                                                class_elimination_count[
                                                                                                    i][1], 100 * (
                                                                                                        class_elimination_count[
                                                                                                            i][1] /
                                                                                                        class_elimination_count[
                                                                                                            i][0])))


if __name__ == "__main__":
    main()
