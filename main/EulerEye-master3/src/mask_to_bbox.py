"""
mask_to_bbox.py
----------------
Convert fcn probability mask into bounding boxes

@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   02/20/2020
"""
import cv2 as cv
import numpy as np

# Define Kernels as global variables
STRUCT_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)  # 4-connection
STRUCT_SQUARE = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)  # 8-connection
HORIZONTAL_KER_3 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)  # horizontal kernel
HORIZONTAL_KER_5 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)  # horizontal kernel
VERTICAL_KER_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)  # vertical kernel


def erode(image, ker, r):
    return cv.erode(image, ker, iterations=r)


def dilate(image, ker, r):
    return cv.dilate(image, ker, iterations=r)


def open(image, ker, r):
    return cv.morphologyEx(image, cv.MORPH_OPEN, ker, iterations=r)


def close(image, ker, r):
    return cv.morphologyEx(image, cv.MORPH_CLOSE, ker, iterations=r)


def morphological_process(image):
    # binarize the given mask image using Otsu's algorithm
    # Original Mathematica implementation is adding 0.15 * 255 from auto-thresholding
    thres, _ = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    new_thres = int(thres + 0.15 * 255)
    #new_thres = min(200, int(thres + 0.25 * 255))
    max_val = 255
    binary_img = np.array((image > new_thres) * max_val, dtype=np.uint8)
    step1 = open(binary_img, STRUCT_SQUARE, 2)
    step2 = open(step1, HORIZONTAL_KER_3, 14)
    #step3 = open(step2, VERTICAL_KER_5, 4)
    return step2
    #return binary_img


def process_single_mask(mask_path, w, h):
    # read in mask as csv file
    image = np.genfromtxt(mask_path, delimiter=",")
    # convert the probability mask to the right scale f:[0,1] -> [0,255]
    mask = image * 255
    # define the channel to int8 to avoid "error: (-215) _src.type() == CV_8UC1 in function equalizeHist"
    mask = mask.astype(np.uint8)
    morpho = morphological_process(mask)
    labels = label_components(morpho)
    bboxes_resized = get_bboxes(labels)
    bboxes = convert_bbox_to_original(bboxes_resized, w, h)
    return bboxes


def label_components(image):
    w, h = len(image), len(image[0])
    labels = [[0 for _ in range(h)] for _ in range(w)]
    visited = [[0 for _ in range(h)] for _ in range(w)]

    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    cnt = 0
    # loop over all un-visited foreground pixels
    for x in range(w):
        for y in range(h):
            if image[x][y] == 1 and visited[x][y] == 0:
                visited[x][y] = 1
                cnt += 1
                # BFS: Start flooding from (x,y)
                queue = [(x, y)]
                labels[x][y] = cnt
                while queue:
                    (qx, qy) = queue.pop(0)
                    for i in range(4):
                        nx, ny = qx + dx[i], qy + dy[i]
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and image[nx][ny] == 1
                            and visited[nx][ny] == 0
                        ):
                            queue.append((nx, ny))
                            labels[nx][ny] = cnt
                            visited[nx][ny] = 1
    return labels


def get_bboxes(mask_labels):
    mask = np.array(mask_labels)
    labels = list(np.unique(mask))
    bboxes = []
    for label in labels:
        if label:  # ignore the case when label = 0
            pos = np.array(list(map(list, list(zip(*np.where(mask == label))))))
            y_min = pos[:, 0].min()
            y_max = pos[:, 0].max()
            x_min = pos[:, 1].min()
            x_max = pos[:, 1].max()
            bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes


def convert_bbox_to_original(bboxes, w, h):
    ratio_w, ratio_h = w / 512, h / 512
    bboxes_original_size = []
    for box in bboxes:
        [x_min, y_min, x_max, y_max] = box
        new_x_min = int(x_min * ratio_w)
        new_x_max = int(x_max * ratio_w)
        new_y_min = int(y_min * ratio_h)
        new_y_max = int(y_max * ratio_h)
        bboxes_original_size.append([new_x_min, new_y_min, new_x_max, new_y_max])
    return bboxes_original_size


def process_single_mask(mask_path, w, h, bbox_fname):
    # read in mask as csv file
    image = np.genfromtxt(mask_path, delimiter=",")
    # convert the probability mask to the right scale f:[0,1] -> [0,255]
    mask = image * 255
    # define the channel to avoid "error: (-215) _src.type() == CV_8UC1 in function equalizeHist"
    mask = mask.astype(np.uint8)
    morpho = morphological_process(mask)
    morpho[morpho > 0] = 1
    labels = label_components(morpho)
    bboxes_resized = get_bboxes(labels)
    bboxes = convert_bbox_to_original(bboxes_resized, w, h)
    np.savetxt(bbox_fname, bboxes, delimiter=",")
    return bboxes
