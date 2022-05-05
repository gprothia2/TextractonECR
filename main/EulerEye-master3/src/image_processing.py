"""
image_processing.py
----------------
Image Processing Functions:
@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   06/01/2020
"""

from PIL import Image
import numpy as np


def image_trim(image, roi):
    """
    :param image: numpy array
    :param roi: [x_min, y_min, x_max, y_max]
    :return:
    """
    if type(roi) == np.ndarray:
        [x_min, y_min, x_max, y_max] = [int(x) for x in roi]
        return image[y_min:y_max, x_min:x_max]
    else:
        [x_min, y_min, x_max, y_max] = [int(x) for x in roi]
        return image[y_min:y_max, x_min:x_max]


def save_img(image, path):
    pil_img = Image.fromarray(image)
    pil_img.save(path)
