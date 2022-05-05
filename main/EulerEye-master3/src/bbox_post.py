"""
bbox_post_processing.py
----------------
Post Processing Functions for Bounding Boxes:

@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   07/18/2020
"""
import numpy as np
import math

from shapely.geometry import Polygon, Point


def _check_boundary(bbox, w, h):
    [xmin, ymin, xmax, ymax] = bbox
    if xmax > w:
        print("xmax", xmax)
        xmax = w - 5
    if ymax > h:
        print("ymax", ymax)
        ymax = h - 5
    return [xmin, ymin, xmax, ymax]


def adjust_from_boundary(bboxes, w, h):
    new_bboxes = []
    for bbox in bboxes:
        bbox = _check_boundary(bbox, w, h)
        new_bboxes.append(bbox)
    print("bounding boxes out of boundary, so need to re-adjust")
    return new_bboxes


def _bbox_is_valid(bbox):
    [xmin, ymin, xmax, ymax] = bbox
    if xmin >= xmax:
        print(xmin, xmax)
        return False
    elif ymin >= ymax:
        print(ymin, ymax)
        return False
    return True


def check_bboxes_are_valid(bboxes):
    final_bbox = []
    for bbox in bboxes:
        if _bbox_is_valid(bbox):
            final_bbox.append(bbox)
        else:
            print("{} is not valid".format(bbox))
    return final_bbox


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
    modal_width = np.argmax(np.bincount(np.round(bboxes[:, 2] - bboxes[:, 0], decimals=-2)))
    multiple_col_thresh = (modal_width * 1.1) / image_width
    thresh = np.round(modal_width * 0.8)
    return thresh, multiple_col_thresh


class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class UnionFind:
    def __init__(self, nodes):
        self.father = {}
        for i in range(len(nodes)):
            self.father[nodes[i].label] = nodes[i].label

    def find(self, x):
        if self.father[x] == x:
            return x
        else:
            self.father[x] = self.find(self.father[x])
            return self.father[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y


class UndirectedGraph:
    """
    @param: nodes: a array of Undirected graph node
    @return: a connected set of a Undirected graph
    """

    def connectedSet(self, nodes):
        uf = UnionFind(nodes)
        for node in nodes:
            for neighbor in node.neighbors:
                uf.union(node.label, neighbor.label)

        hash = {}
        for node in nodes:
            root_label = uf.find(node.label)
            if root_label not in hash:
                hash[root_label] = []
            hash[root_label].append(node.label)

        res = []
        for _, node in hash.items():
            res.append(node)

        return list(map(sorted, list(map(set, res))))


###################################################################################################
#                                     BOUNDING-BOXES CONVERSION
###################################################################################################
def convert_bbox_xyxy_to_ABCD(box):
    """
    convert a bounding box into ABCD four vertices in counter clockwise
    :param box:
    :return: a list of four vertices left top, left bottom, right top, right bottom
    """
    [xmin, ymin, xmax, ymax] = box
    A = [xmin, ymin]
    B = [xmin, ymax]
    C = [xmax, ymax]
    D = [xmax, ymin]
    return [A, B, C, D]


def convert_bboxes_xyxy_to_ABCD(bboxes):
    res = []
    for b in bboxes:
        res.append(convert_bbox_xyxy_to_ABCD(b))
    return res


def convert_bbox_ABCD_to_xyxy(bbox_ABCD):
    try:
        [A, _, C, _] = bbox_ABCD
        [xmin, ymin] = A
        [xmax, ymax] = C
        return [xmin, ymin, xmax, ymax]
    except:
        print("ERROR")


def convert_bboxes_ABCD_to_xyxy(bboxes_ABCD):
    res = []
    for bbox in bboxes_ABCD:
        res.append(convert_bbox_ABCD_to_xyxy(bbox))
    return res


###################################################################################################
#                                BOUNDING-BOXES CONNECTIVITY HELPERS
###################################################################################################
def euclidean_distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def check_point_in_range(p1, p2, r):
    if euclidean_distance(p1, p2) < r:
        return True
    else:
        return False


def clustering_bboxes(bboxes, r):
    """
    :param bboxes:
    :param r:
    :return:
    """
    connected_components = _build_connected_components_from_undirected_graph(bboxes, r)
    (index_map, index_map_inverse, bboxes_abcd) = _build_building_blocks(bboxes)
    # each index set should have one single mean
    substitution_rule = []
    for index_set in connected_components:
        cluster = []
        for index in index_set:
            (i, j) = index_map[index]
            coord = bboxes_abcd[i][j]
            cluster.append(coord)
        new_vertices = np.mean(cluster, axis=0).astype(int).tolist()
        substitution_rule.append(new_vertices)

    for i in range(len(substitution_rule)):
        index_set = connected_components[i]
        new_index = substitution_rule[i]
        for index in index_set:
            (i, j) = index_map[index]
            bboxes_abcd[i][j] = new_index

    new_bboxes = convert_bboxes_ABCD_to_xyxy(bboxes_abcd)
    return new_bboxes


def _build_building_blocks(bboxes):
    """
    Helper functions that build the basic building blocks
    :param bboxes:
    :return: a tuple of (index_map, index_map_inverse, bboxes_abcd)
    """
    n = len(bboxes)
    bboxes_abcd = convert_bboxes_xyxy_to_ABCD(bboxes)  # this doesn't change the order
    # need to build a mapping to map the index to the (i, j) 2d index, and also an inverse map to map it back
    index_map = {}
    index_map_inverse = {}
    counter = 0
    for i in range(n):
        for k in range(4):
            index_map[counter] = (i, k)
            index_map_inverse[(i, k)] = counter
            counter += 1
    return (index_map, index_map_inverse, bboxes_abcd)


def _build_connected_components_from_undirected_graph(bboxes, r):
    """
    :param bboxes: xyxy
    :param r: threshold for euclidean distance (num of pixels)
    :return:
    """
    graph = []
    graph_single = []
    n = len(bboxes)
    (index_map, index_map_inverse, bboxes_abcd) = _build_building_blocks(bboxes)
    # brute force
    for i in range(n):
        box = bboxes[i]
        box_abcd = convert_bbox_xyxy_to_ABCD(box)
        for k in range(4):
            p = box_abcd[k]
            for j in range(i + 1, n):
                vertices = bboxes_abcd[j]
                for v in range(0, len(vertices)):
                    if check_point_in_range(p, vertices[v], r=r) and p != vertices[v]:
                        edge1 = [i, k]
                        edge2 = [j, v]
                        graph.append([edge1, edge2])
                        # look up
                        e1 = index_map_inverse[(i, k)]
                        e2 = index_map_inverse[(j, v)]  # check
                        graph_single.append([e1, e2])
                        graph_single.append([e2, e1])

    g = {}
    for edge in graph_single:
        (e1, e2) = edge
        if e1 in g.keys():
            g[e1].append(e2)
        else:
            g[e1] = [e2]

    data = []
    for v, nbs in g.items():
        node = UndirectedGraphNode(v)
        node.neighbors = [UndirectedGraphNode(n) for n in nbs]
        data.append(node)
    ug = UndirectedGraph()
    connected_components = ug.connectedSet(data)
    return connected_components


###################################################################################################
#                                BOUNDING-BOXES CONNECTIVITY HELPERS
###################################################################################################
def is_intersect(box1, box2):
    """
    Two axes aligned boxes (of any dimension) overlap if and only if the projections to all axes overlap.
    The projection to an axis is simply the coordinate range for that axis.
    :param box1: bounding box 1
    :param box2: bounding box 2
    :return:
    """
    [x1_min, y1_min, x1_max, y1_max] = box1
    [x2_min, y2_min, x2_max, y2_max] = box2
    if x1_min < x2_max and x2_min < x1_max and y1_min < y2_max and y2_min < y1_max:
        return True
    else:
        return False


def polygon_within(bboxes):
    """
    :param bboxes: xyxy bboxes
    :return: a list of indices (i, j) where bboxes[i] is in bboxes[j]
    """
    n = len(bboxes)
    bboxes_ABCD = convert_bboxes_xyxy_to_ABCD(bboxes)
    subsets = []
    for i in range(n):
        for j in range(n):
            if i != j:
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bboxes_ABCD[i]
                p1 = Point(x1, y1)
                p2 = Point(x2, y2)
                p3 = Point(x3, y3)
                p4 = Point(x4, y4)
                poly = Polygon(bboxes_ABCD[j])
                if p1.within(poly) and p2.within(poly) and p3.within(poly) and p4.within(poly):
                    subsets.append((i, j))
    return subsets


def undirected_graph_with_iou(bboxes):
    n = len(bboxes)
    bboxes_ABCD = convert_bboxes_xyxy_to_ABCD(bboxes)
    graph = []
    iou_dict = {}
    for i in range(n):
        polygon1 = Polygon(bboxes_ABCD[i])
        for j in range(i + 1, n):
            polygon2 = Polygon(bboxes_ABCD[j])
            iou = get_polygon_iou(polygon1, polygon2)
            if iou != 0:
                graph.append([i, j])
                iou_dict[(i, j)] = iou
    return graph, iou_dict


"""
polygon IOU is not useful right now, might be useful later
because we care about the intersection area / each of the polygon area
"""


def get_polygon_iou(polygon1_coords, polygon2_coords):
    """
    Calculate intersection over union overlap between a pair of polygons (coords can take more than 4 points, but
    expected input is quadrilaterals)
    Parameters:
        polygon1_coords: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        polygon2_coords:  [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    Returns:
        iou_overlap : overlap between boxes
    """
    polygon1 = Polygon(polygon1_coords)
    polygon2 = Polygon(polygon2_coords)
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0
    overlap_area = polygon1.intersection(polygon2).area
    total_area = polygon1.union(polygon2).area
    if total_area == 0:
        return 0
    return overlap_area / total_area


###################################################################################################
#                                  REMOVE OVERLAPPING BOUNDING-BOXES
###################################################################################################
def get_overlap_ratio_on_smaller_bboxes(polygon1_coords, polygon2_coords):
    """
    Calculate intersection over union overlap between a pair of polygons (coords can take more than 4 points, but
    expected input is quadrilaterals)
    Parameters:
        polygon1_coords: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        polygon2_coords:  [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    Returns:
        iou_overlap : overlap between boxes
    """
    polygon1 = Polygon(polygon1_coords)
    polygon2 = Polygon(polygon2_coords)
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0
    overlap_area = polygon1.intersection(polygon2).area
    poly1_area = polygon1.area
    poly2_area = polygon2.area

    if poly1_area < poly2_area:
        return 1, overlap_area / poly1_area
    else:
        return 0, overlap_area / poly2_area


def undirected_graph_with_overlap_ratio(bboxes, r):
    n = len(bboxes)
    bboxes_ABCD = convert_bboxes_xyxy_to_ABCD(bboxes)
    graph = []
    iou_dict = {}
    for i in range(n):
        polygon1 = Polygon(bboxes_ABCD[i])
        for j in range(i + 1, n):
            polygon2 = Polygon(bboxes_ABCD[j])
            flag, area_ratio = get_overlap_ratio_on_smaller_bboxes(polygon1, polygon2)
            if area_ratio > r:
                if flag == 1:
                    graph.append([i, j])
                else:
                    graph.append([j, i])
                iou_dict[(i, j)] = area_ratio
    return graph, iou_dict


def _delete(expr, pos):
    """
    delete the element at position n in ex
    :param expr: a list of elements
    :param pos: position / index to be deleted
    :return: a new list
    """
    res = np.array([i for j, i in enumerate(expr) if j not in pos])
    return [l.tolist() for l in res]


def remove_overlapping_bboxes(bboxes, r):
    """
    :param bboxes: xyxy bboxes
    :param r: ratio threshold
    :return: new bboxes
    """
    # consider one page's bounding boxes as a graph
    # where the node is box, and edge means their overlapping threshold is greater than a given threshold
    # We need to test a good threshold
    # build the weighted graph with weight being the overlapping ratio on the smaller box
    undirected_weighted_graph, area_ratio = undirected_graph_with_overlap_ratio(bboxes, r)
    if undirected_weighted_graph == []:
        return bboxes
    else:
        indices_to_delete = np.array(undirected_weighted_graph)[:, 0].tolist()
        new_bboxes = _delete(bboxes, indices_to_delete)
        return new_bboxes


###################################################################################################
#                                  MERGE VERTICAL BOUNDING-BOXES
###################################################################################################
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
    read_order_bboxs_left_aligned = sorted(
        read_order_bboxs_left_aligned, key=lambda x: (x[0], x[1])
    )

    # modified right boundary
    read_order_bboxs_right_aligned = []
    for bbox_left_aligned in read_order_bboxs_left_aligned:
        for r_b in right_boundary:
            if abs(bbox_left_aligned[2] - r_b) < thresh * 2 / 3:
                bbox_left_aligned[2] = r_b
                read_order_bboxs_right_aligned.append(bbox_left_aligned)
    read_order_bboxs_right_aligned = sorted(
        read_order_bboxs_left_aligned, key=lambda x: (x[0], x[1])
    )

    # convert right boundary to the left boundary for next colums,
    # convert bottom boundary to the top boundary for next block
    read_order_small_bbox = [read_order_bboxs_right_aligned[0]]
    for i in range(1, len(read_order_bboxs_right_aligned)):
        bbox = read_order_bboxs_right_aligned[i]

        if (
            abs(bbox[0] - read_order_small_bbox[-1][0]) < thresh / 5
            and abs(bbox[1] - read_order_small_bbox[-1][3]) < thresh / 4
        ):  # in the same column
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

    final_read_order_bbox = sorted(
        read_order_small_bbox + multiple_col_bbox, key=lambda x: (x[0], x[1])
    )

    # let top block hit top, let bot block hit bottom/ find top boundary, bot boundary
    # for bbox in final_read_order_bbox
    return np.array(final_read_order_bbox)


def vertical_merge(bboxes):
    """
    Merge vertically adjacent bboxes
    """
    bboxes = sort_bbox_in_read_order(bboxes, 250, 0.3, 5000)
    x_min_max_2_bbox = {}
    bboxes_list = []
    for bbox in bboxes:
        if (bbox[0], bbox[2]) not in x_min_max_2_bbox:
            x_min_max_2_bbox[(bbox[0], bbox[2])] = [list(bbox)]
        else:
            x_min_max_2_bbox[(bbox[0], bbox[2])].append(list(bbox))

    for key, val in x_min_max_2_bbox.items():
        bboxes_list += merge_same_col_box(val)

    return np.array(bboxes_list), len(bboxes_list), len(bboxes)


def merge_same_col_box(bboxes_same_col):
    sorted_bboxes = sorted(bboxes_same_col)
    bbox_list = [sorted_bboxes[0]]
    for i in range(1, len(sorted_bboxes)):
        if abs(sorted_bboxes[i][1] - bbox_list[-1][3]) < 50:
            bbox_list[-1][3] = sorted_bboxes[i][3]
        else:
            bbox_list.append(sorted_bboxes[i])
    return bbox_list
