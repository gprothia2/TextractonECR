import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mxnet as mx
import gluoncv
import math
import numpy as np


def visualization_image_with_box(image_path, bboxes, show_order, save_img):
    image = mx.image.imread(image_path).asnumpy()
    image_name = "_".join(image_path.split("/")[4:])
    plt.figure(figsize=(60, 60))
    ax1 = plt.subplot(1, 2, 1)
    # title_font_size = 10
    # label_font_size = 6
    # color = 'darkorange'
    if show_order:
        gluoncv.utils.viz.plot_bbox(image, bboxes, labels=np.arange(len(bboxes)), ax=ax1)
    else:
        gluoncv.utils.viz.plot_bbox(image, bboxes, ax=ax1)
    if save_img:
        plt.savefig("output/{}".format(image_name))


def get_cropped_image_keys(file_key, file_paths, crop_image_prefix):
    """
    file_key: str
    example: crop_image_prefix = '/home/ec2-user/SageMaker/crop_image_06_16/'
    """
    cropped_imgs = []

    for f in file_paths:
        if f.startswith(file_key):
            crop_image_full_path = crop_image_prefix + f
            cropped_imgs.append(crop_image_full_path)
    return cropped_imgs


def plot_image(file_path):
    img = mpimg.imread(file_path)
    plt.imshow(img)
    plt.axis("off")


def parse_to_graph(text, num_in_line=None):
    words = text.split(" ")
    n_words = len(words)
    if not num_in_line:
        num_in_line = 5
    res = []
    for i in range(n_words):
        res.append(words[i])
        if i % num_in_line == 0 and i > 0:
            res.append("\n")

    return " ".join(res)


def show_crop_image_with_words(textract_path, img_key, file_paths):
    textract_words = get_textract_words(textract_path)
    img_paths = get_cropped_image_keys(img_key, file_paths)
    n = len(img_paths)
    fontsize = 20
    columns = 3
    rows = math.floor(n / columns)
    fig = plt.figure(figsize=(rows * 10, columns * 10))
    axs = []
    num = columns * rows + 1
    for i in range(1, num):
        img = plt.imread(img_paths[i - 1])
        # create subplot and append to ax
        axs.append(fig.add_subplot(rows, columns, i))
        plt.imshow(img)
        plt.subplots_adjust(wspace=0, hspace=0.1)  # 调整子图间距
        plt.axis("off")
        # plot texts
        text = textract_words[i - 1]
        graph = parse_to_graph(text, 10)
        axs[-1].set_title(graph, fontsize=fontsize, fontweight="bold", backgroundcolor="silver")
    plt.show()
    image_path = "visualization_{}.jpg".format(img_key)
    plt.savefig(image_path)
