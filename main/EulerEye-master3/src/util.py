"""
util.py
----------------
Holds utility functions
@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   11/26/2019
@edit by Yang, Guang (yaguan@amazon.com)
@date   07/19/2021
"""
import os
import json
import logging
import numpy as np
import shutil
from PIL import Image

import torch

def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()


class bcolors:
    PINK = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def create_experiment_directory(root):
    # create experiment folder based on root folder and time
    input_dir = os.path.join(root, "demo_image")
    print(root)
    experiment_dir = os.path.join(root, "output")
    mask_dir = os.path.join(experiment_dir, "pred_mask")
    bbox_raw_dir = os.path.join(experiment_dir, "bbox_raw")
    bbox_dir = os.path.join(experiment_dir, "bbox")
    cropped_img_dir = os.path.join(experiment_dir, "cropped_image")
    ocr_plain_dir = os.path.join(experiment_dir, "ocr_plain_text")
    ocr_coord_dir = os.path.join(experiment_dir, "ocr_coords")
    ocr_vis_dir = os.path.join(experiment_dir, "visualization")

    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)
    if not os.path.isdir(bbox_raw_dir):
        os.mkdir(bbox_raw_dir)
    if not os.path.isdir(bbox_dir):
        os.mkdir(bbox_dir)
    if not os.path.isdir(cropped_img_dir):
        os.mkdir(cropped_img_dir)
    if not os.path.isdir(ocr_plain_dir):
        os.mkdir(ocr_plain_dir)
    if not os.path.isdir(ocr_coord_dir):
        os.mkdir(ocr_coord_dir)
    if not os.path.isdir(ocr_vis_dir):
        os.mkdir(ocr_vis_dir)
    return (
        input_dir,
        mask_dir,
        bbox_raw_dir,
        bbox_dir,
        cropped_img_dir,
        ocr_plain_dir,
        ocr_coord_dir,
    )

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 4)
    
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_checkpoint(state, is_best, is_train_best, checkpoint, params):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    weight = params.weighted_loss
    batch_size = params.batch_size
    lr = params.learning_rate
    filepath = os.path.join(checkpoint, 'last_{}_lr_{}_batch_{}.pth.tar'.format(weight, lr, batch_size))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_{}_lr_{}_batch_{}.pth.tar'.format(weight, lr, batch_size)))
    if is_train_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'train_best_{}_lr_{}_batch_{}.pth.tar'.format(weight, lr, batch_size)))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def bb_is_inside(box_small, box_large, thres=0.1):
    """
    Determine the small bbox is within the large bbox 
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_small[0], box_large[0])
    yA = max(box_small[1], box_large[1])
    xB = min(box_small[2], box_large[2])
    yB = min(box_small[3], box_large[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box_small_area = (box_small[2] - box_small[0] + 1) * (box_small[3] - box_small[1] + 1)
    ios = interArea / float(box_small_area)
    return ios>thres


def dist_of_two_point(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
