"""
bbox_post_processing.py
----------------
Post Processing Functions for Bounding Boxes:

@author Zhang, Tianyu (ttizha@amazon.com)
@author Biswas, Raj (rajbisr@amazon.com)
@date   06/08/2020
"""

import numpy as np

def get_read_order_thresholds(bboxes, image_width):
    """
    Calculates the thresholds for converting bboxes of segementation model to a read order
    Args:
        bboxs(np.array): 2D-array for bbox
        image_width(int): the image width

    Returns:
        thresh (int): a threshold to evaluate whether two bboxs are adjacent
        multiple_col_thresh (float): a threshold to evaluate whether a bbox is a cross-col bbox
    """
    bboxes = np.array(bboxes)
    modal_width = np.argmax(np.bincount(np.round(bboxes[:, 2] - bboxes[:, 0], decimals=-2)))
    multiple_col_thresh = (modal_width * 1.1) / image_width
    thresh = np.round(modal_width * 0.8)
    return thresh, multiple_col_thresh

def sort_bbox_in_read_order(bboxes, thresh, multiple_col_thresh, image_w):
    """
    Sort bboxes of segementation models to a read order
    Args:
        bboxs(np.array): 2D-array for bbox
        thresh(int): a threshold to evaluate whether two bboxs are adjacent
        multiple_col_thresh(float): a threshold to evaluate whether a bbox is a cross-col bbox
        image_w, image_h(int): the image size

    Returns:
        final_read_order_bbox(np.ndarray): 2D-array for bbox
    """

    # make a mapping with bbox and its top-left point(filter out cross multiple cols bbox)
    multiple_col_bbox = []
    top_left_2_bbox = {}
    bboxs = np.copy(bboxes)
    for bbox in bboxs:
        if abs(bbox[2] - bbox[0]) > multiple_col_thresh * image_w:
            multiple_col_bbox.append(bbox)
            continue
        top_left_2_bbox[(bbox[0], bbox[1])] = bbox
    if len(top_left_2_bbox) == 0: #every box is considered a multiple col
        for bbox in bboxs:
            top_left_2_bbox[(bbox[0], bbox[1])] = bbox
        
    top_left_ponits = list(top_left_2_bbox.keys())
    
    # sort bbox with two keys
    top_left_points_sorted = sorted(top_left_ponits, key=lambda x: (x[0], x[1]))

    # find most left boundary for each columns:
    left_boundary = [top_left_points_sorted[0][0]]
    for i in range(1, len(top_left_points_sorted)):
        if abs(top_left_points_sorted[i][0] - left_boundary[-1]) > thresh:
            left_boundary.append(top_left_points_sorted[i][0])
        else:
            left_boundary[-1] = min(top_left_points_sorted[i][0], left_boundary[-1])

    revised_top_left_2_origin = {}
    # set all of the bbox with most left boundary:
    left_align_points = []
    for (x, y) in top_left_points_sorted:
        for l_b in left_boundary:
            if abs(x - l_b) < thresh:
                # Added condition - Raj - Case where x satisies the condition for multiple l_b
                if (x, y) not in revised_top_left_2_origin.values():
                    left_align_points.append((l_b, y))
                    revised_top_left_2_origin[(l_b, y)] = (x, y)

    # resort the bbox with aligned left boundary:
    read_order_points = sorted(left_align_points, key=lambda x: (x[0], x[1]))

    # modify every bbox on left boundary and get most right boundary
    read_order_bboxs_left_aligned = []
    right_boundary = [top_left_2_bbox[revised_top_left_2_origin[read_order_points[0]]][2]]
    for revised_top_left in read_order_points:
        bbox = top_left_2_bbox[revised_top_left_2_origin[revised_top_left]]
        bbox[0] = revised_top_left[0]
        read_order_bboxs_left_aligned.append(bbox)
        if abs(bbox[2] - right_boundary[-1]) > thresh * 2 / 3:  # hard code here modified later
            right_boundary.append(bbox[2])
        else:
            right_boundary[-1] = max(bbox[2], right_boundary[-1])
    read_order_bboxs_left_aligned = sorted(read_order_bboxs_left_aligned, key=lambda x: (x[0], x[1]))

    # modified right boundary
    read_order_bboxs_right_aligned = []
    for bbox_left_aligned in read_order_bboxs_left_aligned:
        for r_b in right_boundary:
            if abs(bbox_left_aligned[2] - r_b) < thresh * 2 / 3:
                bbox_left_aligned[2] = r_b
                read_order_bboxs_right_aligned.append(bbox_left_aligned)
    read_order_bboxs_right_aligned = sorted(read_order_bboxs_left_aligned, key=lambda x: (x[0], x[1]))

    # convert right boundary to the left boundary for next colums,
    # convert bottom boundary to the top boundary for next block
    read_order_small_bbox = [read_order_bboxs_right_aligned[0]]
    for i in range(1, len(read_order_bboxs_right_aligned)):
        bbox = read_order_bboxs_right_aligned[i]

        if abs(bbox[0] - read_order_small_bbox[-1][0]) < thresh / 5 and abs(
                bbox[1] - read_order_small_bbox[-1][3]) < thresh / 4:  # in the same column
            bbox[0] = read_order_small_bbox[-1][0]
            bbox[1] = read_order_small_bbox[-1][3]
            bbox[2] = read_order_small_bbox[-1][2]
            read_order_small_bbox.append(bbox)
        elif abs(bbox[0] - read_order_small_bbox[-1][0]) < thresh / 10:
            bbox[0] = read_order_small_bbox[-1][0]
            bbox[2] = read_order_small_bbox[-1][2]
            read_order_small_bbox.append(bbox)
        else:  # cross columns
            # Added condition - Raj - Only correcting if bbox is the immediate next column
            if abs(bbox[0] - read_order_small_bbox[-1][2]) < thresh / 5:
                bbox[0] = read_order_small_bbox[-1][2]
            read_order_small_bbox.append(bbox)

    # add the corss columns blocks
    for i, larger_bbox in enumerate(multiple_col_bbox):
        for l_b in left_boundary:
            if abs(multiple_col_bbox[i][0] - l_b) < thresh * 2 / 3:
                multiple_col_bbox[i][0] = l_b
        for r_b in right_boundary:
            if abs(multiple_col_bbox[i][2] - r_b) < thresh * 2 / 3:
                multiple_col_bbox[i][2] = r_b
    if len(top_left_2_bbox) == len(bboxs):
        final_bbox = read_order_small_bbox
    else:
        final_bbox = read_order_small_bbox + multiple_col_bbox
    final_read_order_bbox = sorted(final_bbox, key=lambda x: (x[0], x[1]))

    # let top block hit top, let bot block hit bottom/ find top boundary, bot boundary
    # for bbox in final_read_order_bbox
    return np.array(final_read_order_bbox)

def merge_small_boxes(bboxes, height_threshold, dist_threshold):
    """
    Merges larger bboxes to smaller ones
    Args:
        bboxs(np.array): 2D-array for bbox
        height_threshold(float): a threshold to evaluate whether two bboxs are adjacent
        dist_threshold(float): a threshold to evaluate whether a bbox is a cross-col bbox

    Returns:
        bboxes (np.ndarray): 2D-array for bbox
    """
    i = 0
    while (i < bboxes.shape[0]):
        height = bboxes[i, 3] - bboxes[i, 1]
        prev_flag = 0
        next_flag = 0
        if i != 0:
            # Checking if the previous box is aligned with the current box
            prev_align = (bboxes[i - 1, 0] == bboxes[i, 0]) and (bboxes[i - 1, 2] == bboxes[i, 2])
        if i != (bboxes.shape[0] - 1):
            # Checking if the next box is aligned with the current box
            next_align = (bboxes[i + 1, 0] == bboxes[i, 0]) and (bboxes[i + 1, 2] == bboxes[i, 2])
        if height < height_threshold:
            # Condition for first box
            if i == 0:
                dist_next = ((bboxes[i + 1, 1] - bboxes[i, 3]) < dist_threshold)
                if next_align and dist_next:
                    next_flag = 1
            # Condition for last box
            elif i == (bboxes.shape[0] - 1):
                dist_prev = ((bboxes[i, 1] - bboxes[i - 1, 3]) < dist_threshold)
                if prev_align and dist_prev:
                    prev_flag = 1
            else:
                dist_prev = ((bboxes[i, 1] - bboxes[i - 1, 3]) < dist_threshold)
                dist_next = ((bboxes[i + 1, 1] - bboxes[i, 3]) < dist_threshold)
                if (next_align and dist_next) and (prev_align and dist_prev):
                    height_prev = bboxes[i - 1, 3] - bboxes[i - 1, 1]
                    height_next = bboxes[i + 1, 3] - bboxes[i + 1, 1]
                    if height_prev >= height_next:
                        next_flag = 1
                    else:
                        prev_flag = 1
                elif prev_align and dist_prev:
                    prev_flag = 1
                elif next_align and dist_next:
                    next_flag = 1
            if prev_flag:
                bboxes[i - 1, 3] = bboxes[i, 3]
                bboxes = np.delete(bboxes, (i), axis=0)
                i = i - 1
                continue
            elif next_flag:
                bboxes[i + 1, 1] = bboxes[i, 1]
                bboxes = np.delete(bboxes, (i), axis=0)
                continue
            else:
                i += 1
        else:
            i += 1
    return bboxes

def post_process_bboxes(bboxes, image_width):
    """
    Combines all post processing functions
    Args:
        bboxs(np.array): 2D-array for bbox
        image_width (np.int): image width
    Returns:
        bboxes (np.array): 2D-array for bbox
    """

    thresh, multiple_col_thresh = get_read_order_thresholds(bboxes, image_width)
    new_bboxes = sort_bbox_in_read_order(bboxes, thresh, multiple_col_thresh, image_width)
    final_bboxes = merge_small_boxes(new_bboxes, 400, 100)
    return final_bboxes