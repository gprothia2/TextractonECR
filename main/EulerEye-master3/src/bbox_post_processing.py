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
from graph import UndirectedGraphNode, UndirectedGraph


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
    connected_components = build_connected_components_from_undirected_graph(bboxes, r)
    (index_map, index_map_inverse, bboxes_abcd) = _build_building_blocks(bboxes)
    print((index_map, index_map_inverse, bboxes_abcd))
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
            for j in range(i + 1, n):
                index_map[counter] = (i, k)
                index_map_inverse[(i, k)] = counter
                counter += 1
                index_map[counter] = (j, k)
                index_map_inverse[(j, k)] = counter
                counter += 1
    return (index_map, index_map_inverse, bboxes_abcd)


def build_connected_components_from_undirected_graph(bboxes, r):
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
                for v in vertices:
                    if check_point_in_range(p, v, r=r) and p != v:
                        edge1 = [i, k]
                        edge2 = [j, k]
                        graph.append([edge1, edge2])
                        # look up
                        e1 = index_map_inverse[(i, k)]
                        e2 = index_map_inverse[(j, k)]
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
    # poly1_area = polygon1.area
    # poly2_area = polygon2.area
    # avoid dividing by zero
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
    res = [i for j, i in enumerate(expr) if j not in pos]
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
    indices_to_delete = np.array(undirected_weighted_graph)[:, 0].tolist()
    new_bboxes = _delete(bboxes, indices_to_delete)
    return new_bboxes
