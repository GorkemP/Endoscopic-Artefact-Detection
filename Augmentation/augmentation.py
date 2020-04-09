import cv2
import glob
import numpy as np
import tqdm

train_path = "/home/ws2080/Desktop/data/training/train/"
augmented_train_path = "/home/ws2080/Desktop/data/training/train_augmented/"

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
    '''
    :param im_width:
    :param im_height:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param degree:
    convert the bounding boxes into the standard cartesian coordinate values where
    x gets higher in the right direction and y gets higher in the upper direction
    :return:
    '''
    x1_c, x2_c, y1_c, y2_c = get_cartesian(im_width, im_height, x1, x2, y1, y2)

    p1_x, p1_y = rotate(x1_c, y1_c, degree)
    p2_x, p2_y = rotate(x2_c, y1_c, degree)
    p3_x, p3_y = rotate(x1_c, y2_c, degree)
    p4_x, p4_y = rotate(x2_c, y2_c, degree)

    if degree % 180 != 0:
        new_width, new_height = im_height, im_width
    else:
        new_width, new_height = im_width, im_height

    x1_o, x2_o, y1_o, y2_o = get_original_coordinates(new_width, new_height, p1_x, p4_x, p1_y, p4_y)




    x_normalized = (x1_o + x2_o) / (2 * new_width)
    y_normalized = (y1_o + y2_o) / (2 * new_height)
    w_normalized = abs(x1_o - x2_o) / new_width
    h_normalized = abs(y1_o - y2_o) / new_height



    assert x_normalized >= 0 and x_normalized <= 1
    assert y_normalized >= 0 and y_normalized <= 1
    assert w_normalized >= 0 and w_normalized <= 1
    assert h_normalized >= 0 and h_normalized <= 1

    return x_normalized, y_normalized, w_normalized, h_normalized


def flip_vertical(im_width, im_height, x1, x2, y1, y2):
    x1_c, x2_c, y1_c, y2_c = get_cartesian(im_width, im_height, x1, x2, y1, y2)

    x1_c = -x1_c
    x2_c = -x2_c

    x1_o, x2_o, y1_o, y2_o = get_original_coordinates(im_width, im_height, x1_c, x2_c, y1_c, y2_c)

    x_normalized = (x1_o + x2_o) / (2 * im_width)
    y_normalized = (y1_o + y2_o) / (2 * im_height)
    w_normalized = abs(x1_o - x2_o) / im_width
    h_normalized = abs(y1_o - y2_o) / im_height

    return x_normalized, y_normalized, w_normalized, h_normalized


def augment_gt(gt_path, degrees=[0, 90, 180, 270], flip=True):
    image_path = train_path + gt_path.split("/")[-1].split('.')[0] + ".jpg"
    gt_image = cv2.imread(image_path)
    im_height, im_width, _ = gt_image.shape

    root_filename = augmented_train_path + image_path.split('/')[-1].split('.')[0]

    #save rotated and flipped images
    for i in range(len(degrees)):
        rotation_count = int(np.round(degrees[i] / 90))
        image = gt_image
        for j in range(rotation_count):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(root_filename + "_" + str(degrees[i]) + "_not_flipped.jpg", image)
        if flip:
            cv2.imwrite(root_filename + "_" + str(degrees[i]) + "_flipped.jpg", cv2.flip(image, flipCode=1))

    # initilize the contents of the files
    file_contents = {}
    for i in range(len(degrees)):
        not_flipped_name = root_filename + "_" + str(degrees[i]) + "_not_flipped.txt"
        file_contents[not_flipped_name] = ""
        if flip:
            flipped_name = root_filename + "_" + str(degrees[i]) + "_flipped.txt"
            file_contents[flipped_name] = ""

    with open(gt_path, "r") as f:
        content = f.read()
        lines = content.split("\n")
        for line in lines:
            line = line.split(' ')
            if len(line) < 5:
                continue
            clsnum, x, y, w, h = line
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            real_x = int(np.round(im_width * x))
            real_w = int(np.round(im_width * w))
            real_y = int(np.round(im_height * y))
            real_h = int(np.round(im_height * h))

            # get the real values of the corners of the bb's
            x1, x2, y1, y2 = get_coordinates(real_x, real_y, real_w, real_h)

            x1 = int(np.round(x1))
            x2 = int(np.round(x2))
            y1 = int(np.round(y1))
            y2 = int(np.round(y2))

            # for each degree specified
            for i in range(len(degrees)):
                # rotate the box and normalize it
                file_name = root_filename + "_" + str(degrees[i]) + "_not_flipped.txt"
                x_n, y_n, w_n, h_n = rotate_box_normalize(im_width, im_height, x1, x2, y1, y2, degree=degrees[i])
                line_content = clsnum + " " + str(x_n) + " " + str(y_n) + " " + str(w_n) + " " + str(h_n) + "\n"
                file_contents[file_name] += line_content

                #flip the normalized bb values
                if flip:
                    if degrees[i] % 180 != 0:
                        new_width, new_height = im_height, im_width
                    else:
                        new_width, new_height = im_width, im_height

                    r_x = int(np.round(new_width * x_n))
                    r_w = int(np.round(new_width * w_n))
                    r_y = int(np.round(new_height * y_n))
                    r_h = int(np.round(new_height * h_n))

                    x1_r, x2_r, y1_r, y2_r = get_coordinates(r_x, r_y, r_w, r_h)

                    x1_r = int(np.round(x1_r))
                    x2_r = int(np.round(x2_r))
                    y1_r = int(np.round(y1_r))
                    y2_r = int(np.round(y2_r))

                    x_n, y_n, w_n, h_n = flip_vertical(new_width, new_height, x1_r, x2_r, y1_r, y2_r)
                    file_name = root_filename + "_" + str(degrees[i]) + "_flipped.txt"
                    line_content = clsnum + " " + str(x_n) + " " + str(y_n) + " " + str(w_n) + " " + str(h_n) + "\n"
                    file_contents[file_name] += line_content

    #write the contents to the files
    for file_name in file_contents.keys():
        with open(file_name, "w") as file:
            file.write(file_contents[file_name])

    return

def main():
    for filename in tqdm.tqdm(sorted(glob.glob(train_path + '*.txt'))):
        augment_gt(filename)
    return

if __name__ == "__main__":
    main()
