"""
pipeline.py
----------------
This file holds the main and RunEvents

@author Zhu, Wenzhen (wenzhu@amazon.com), edit by Yang, Guang (yaguan@amazon.com)
@date   11/16/2020

* Finalize Script Design
    * argument: image_path, root, coords
    * intermediate result and final result will be saved at different place
    * pre-trained model, hard-coded
"""

# Import Python Packages
import argparse
import time
import os
import json
from PIL import Image
import numpy as np
import pandas as pd
import boto3

# Import our modules
from util import *
#from mask_to_bbox import process_single_mask
from inference import get_ordered_segments, reorder_word
# from bbox_post_processing_original import post_process_bboxes
# from bbox_post import (
#     remove_overlapping_bboxes,
#     clustering_bboxes,
#     vertical_merge,
#     adjust_from_boundary,
#     check_bboxes_are_valid,
# )
from image_processing import image_trim, save_img
from textract import (
    textract_text_only,
    translate_word_bboxes,
    textract_bbox_text,
    flatten,
)
from data_loading import upload_to_s3
from file_util import write_file, save_texts, save_dict_to_json
from constants import S3_BUCKET_NAME,S3_RAW_IMAGE_FOLDER,TEXTRACT_LIMIT_SIZE,FCN_MODEL  

OCR_OUTPUT_HEADER = ["word", "segment_id", "confidence","x_min", "y_min", "x_max", "y_max"]
OUTPUT_SEP = ','

###################################################################################################
#                            BBOX POST-PROCESSING PIPELINE
###################################################################################################
def _get_raw_bboxes(raw_bbox_fname):
    df = pd.read_csv(raw_bbox_fname, header=None)
    pred_bboxes = df.to_numpy()
    return pred_bboxes


def post_processing_module(image, raw_bboxes, w, h):
    """
    :param image: numpy (image data
    :param raw_bbox_fname: str
    :return: a list of bbox
    """
    less_bboxes = remove_overlapping_bboxes(raw_bboxes, r=0.8)
    if len(less_bboxes) > 6:
        post_bboxes = post_process_bboxes(image, less_bboxes)
        clustered_bboxes = clustering_bboxes(post_bboxes, r=70)
    else:
        clustered_bboxes = clustering_bboxes(less_bboxes, r=70)
    ## Merge vertical
    if len(clustered_bboxes) > 16:
        try:
            bboxes, _, _ = vertical_merge(clustered_bboxes)
        except:
            print("vertical merge failed")
            bboxes = clustered_bboxes
    else:
        bboxes = clustered_bboxes

    # added an extra step to make sure the final bboxes are within image
    adjusted = adjust_from_boundary(bboxes, w, h)
    result_bboxes = check_bboxes_are_valid(adjusted)
    return result_bboxes


###################################################################################################
#                               TEXTRACT PIPELINE
###################################################################################################

#@TODO: Fix the get_key in a more robust way
def _get_key(img_name):
    return img_name.split("/")[-1].split(".")[0]


def textract_pipeline(image_path, bboxes, combined_bboxes, texts_path, json_path, opt):
    # 1. trim image
    img_key = _get_key(img_name=image_path)
    root_path = "/".join(image_path.split("/")[:-2])
    image = Image.open(image_path)
    image_data = np.asarray(image)

    if opt == 1:
        ocr_lines = []
        for i, bbox in enumerate(combined_bboxes):
            print(bcolors.BLUE + "Running Textract on segment {}...".format(i+1) + bcolors.ENDC)
            x, y, xp, yp = bbox
            w_sub, h_sub = xp-x, yp-y
            prefix, postfix = img_key, str(i).zfill(3)
            sub_image = image_trim(image_data, bbox)
            crop_img_path = "{}/output/cropped_image/{}_{}.jpg".format(root_path, prefix, postfix)
            save_img(sub_image, crop_img_path)
            crop_img_size = os.stat(crop_img_path).st_size
            if crop_img_size < TEXTRACT_LIMIT_SIZE:
                s3_path = "cropped_image/{}_{}.jpg".format(prefix, postfix)
                upload_to_s3(crop_img_path, S3_BUCKET_NAME, s3_path)
                lines = textract_text_only(bucket=S3_BUCKET_NAME, document=s3_path, w=w_sub, h=h_sub, segment_id=i)
                ocr_lines.append(lines)
            else:
                print("File size over limit after cropping")
        write_file(result_file=texts_path, lines=ocr_lines)

    else:
        global_word_dicts = []
        for i, bbox in enumerate(combined_bboxes):
            print(bcolors.BLUE + "Running Textract on segment {}...".format(i+1) + bcolors.ENDC)
            x, y, xp, yp = bbox
            w_sub, h_sub = xp-x, yp-y
            prefix, postfix = img_key, str(i).zfill(3)
            sub_image = image_trim(image_data, bbox)
            crop_img_path = "{}/output/cropped_image/{}_{}.jpg".format(root_path, prefix, postfix)
            save_img(sub_image, crop_img_path)
            crop_img_size = os.stat(crop_img_path).st_size
            if crop_img_size < TEXTRACT_LIMIT_SIZE:
                s3_path = "cropped_image/{}_{}.jpg".format(prefix, postfix)
                upload_to_s3(crop_img_path, S3_BUCKET_NAME, s3_path)
                local_word_dicts = textract_bbox_text(
                    bucket=S3_BUCKET_NAME, document=s3_path, w=w_sub, h=h_sub, segment_id=i
                )
                translated_word_dicts = translate_word_bboxes(local_word_dicts, x, y)
                global_word_dicts.extend(translated_word_dicts)        
            else:
                print("File size over limit after cropping")
                
        global_word_dicts_ordered = reorder_word(global_word_dicts, bboxes)        
        lines = []
        #lines.append(OUTPUT_SEP.join(OCR_OUTPUT_HEADER))
        for word in global_word_dicts_ordered:
            line = OUTPUT_SEP.join([word['word'], 
                                    #str(word['segment']),
                                    str(word['read_order_segment_id']),
                                    str(word['confidence']),
                                    str(word['x_min']), 
                                    str(word['y_min']), 
                                    str(word['x_max']), 
                                    str(word['y_max'])])
            lines.append(line)  
        save_texts(lines, texts_path)
        #also ave as json str
        save_dict_to_json(global_word_dicts, json_path)

def pipeline(root_dir, img_path, option, device):
    """
    :param img_path: string
    :param opt: int 1 / 2
    :return: ocr result
    """
    # set up directory
    (
        input_dir,
        mask_dir,
        bbox_raw_dir,
        bbox_dir,
        cropped_img_dir,
        ocr_plain_dir,
        ocr_coord_dir,
    ) = create_experiment_directory(root_dir)
    #print("mask is saved at " + bcolors.YELLOW + mask_dir + bcolors.ENDC)
    print("bounding box is saved at " + bcolors.YELLOW + bbox_dir + bcolors.ENDC)
    print("cropped images are saved at " + bcolors.YELLOW + cropped_img_dir + bcolors.ENDC)
    if option == 1:
        print("OCR task is plain text, saved at " + bcolors.YELLOW + ocr_plain_dir + bcolors.ENDC)
    else:
        print(
            "OCR task is with coordinates, saved at "
            + bcolors.YELLOW
            + ocr_coord_dir
            + bcolors.ENDC
        )
    t0 = time.time()

    # a file on the S3
    # upload new data to s3
    s3_file = S3_RAW_IMAGE_FOLDER + img_path.split("/")[-1]
    # upload_data_to_s3(path, bucket, prefix)
    # specify the model using for inference
    model_file = os.path.join(root_dir, "model", FCN_MODEL)
    img_fname = os.path.join(input_dir, img_path)
    file_format = '.' + img_fname.split("/")[-1].split(".")[-1]
    key =  _get_key(img_name=img_fname)
    
#     # raw data downloaded from the S3
#     output_file_name, output = get_fcn_mask(s3_file, input_dir, model_file, mask_dir)
#     img_fname = os.path.join(input_dir, img_path)
#     # @TODO: This is a dangerous way to get key. Some image has weird format as "$dir/xxxx.yy.png"
#     file_format = '.' + img_fname.split("/")[-1].split(".")[-1]
#     key = img_fname.split("/")[-1].split(file_format)[0]
#     image = Image.open(img_fname)
#     w, h = image.size
#     t1 = time.time()
#     img_to_mask_time = t1 - t0
#     mask_fname = "{}/{}.csv".format(mask_dir, key)
#     raw_bbox_path = "{}/{}/{}/{}.csv".format(root_dir, "output", "bbox_raw", key)
#     raw_bboxes = process_single_mask(mask_fname, w, h, bbox_fname=raw_bbox_path)
#     t2 = time.time()
#     mask_to_bbox_time = t2 - t1
#     image_data = np.asarray(image)
#     bboxes = post_processing_module(image_data, raw_bboxes, w, h)
    
    bboxes, combined_bboxes = get_ordered_segments(s3_file, input_dir, model_file, bbox_dir, device)
    print(bcolors.GREEN + "Segments created!" + bcolors.ENDC)
    #np.savetxt("{}/{}.csv".format(bbox_dir, key), bboxes, delimiter=",", fmt="%i")
    t1 = time.time()
    create_segment_time = t1 - t0
    
    if option == 1:
        ocr_res_path = "{}/{}.txt".format(ocr_plain_dir, key)
    else:
        ocr_res_path = "{}/{}.txt".format(ocr_coord_dir, key)
    print(bcolors.BLUE + "Cropping image and running Textract..." + bcolors.ENDC)
    
#     try:
#         textract_pipeline(img_fname, bboxes, ocr_res_path, ocr_res_path.replace('txt','json'), option)
#     except Exception as e:
#         print(str(e))
    textract_pipeline(img_fname, bboxes, combined_bboxes, ocr_res_path, ocr_res_path.replace('txt','json'), option)
    #textract_pipeline(img_fname, combined_bboxes, ocr_res_path, ocr_res_path.replace('txt','json'), option)
    
    t2 = time.time()
    textract_time = t2 - t1
    return (create_segment_time, textract_time)


class RunEvents:
    """
    Holds the utility functions to run experiments
    """

    def __init__(self, img, root, coords, gpu):
        if not coords:
            coords = 1
            print(
                bcolors.YELLOW
                + "Coordinates flag is not given, set to plain text as default..."
                + bcolors.ENDC
            )
        if not img:
            print("Image file path is not given, please specify the path...")
        if not root:
            root = "/home/ec2-user/SageMaker/ancestry_script_demo/"
            print("Root path is not given, root is set as " + root)
            
        use_cuda = gpu >= 0 and torch.cuda.is_available()
        if use_cuda:
            device = torch.device('cuda:%d' % gpu) 
        else:
            device = torch.device('cpu') 
            
        if coords == 1:
            opt = 1
        else:
            opt = 2

        try:
          dynamodb = boto3.resource('dynamodb')
          table = dynamodb.Table('AncestoryLog')
          (
              create_segment_time,
              textract_time,
          ) = pipeline(root, img, opt, device)
          total_time = create_segment_time + textract_time
          print(bcolors.GREEN + "Total time cost is %s" % total_time + bcolors.ENDC)
          starttime = str(time.time())
          response = table.put_item(
             Item = {
                  'FileName': img,
                  'Insideendtime': starttime
               }
               )
        except Exception as exp:
          img_file = img
          endtime = str(time.time())
          response = table.put_item(
                 Item = {
                        'FileName': img,
                        'Insideerrortime': endtime,
                        'error': exp
                   }
               )


def main():
  try:
    parser = argparse.ArgumentParser("video pre-process pipeline")
    parser.add_argument("--img", action="store", dest="img", default="", type=str)
    parser.add_argument("--root", action="store", dest="root", type=str)
    parser.add_argument("--coords", action="store", dest="coords", default=1, type=int)
    parser.add_argument("--gpu", type=int, default=-1,
                help="gpu device to use")
    args = parser.parse_args()

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('AncestoryLog')
    starttime = str(time.time())
    img_file = args.img
    response = table.put_item(
           Item = {
                  'FileName': img_file,
                  'starttime': starttime
             }
             )
    RunEvents(img=args.img, root=args.root, coords=args.coords, gpu=args.gpu)
    starttime = str(time.time())
    response = table.put_item(
           Item = {
                  'FileName': img_file,
                  'endtime': starttime
             }
             )
  except Exception as exp:
    img_file = args.img
    endtime = str(time.time())
    response = table.put_item(
           Item = {
                  'FileName': img_file,
                  'errortime': endtime,
                  'error': exp
             }
             )


if __name__ == "__main__":
    """
    The main function called when pipeline_script.py is run
    from the command line:
    python pipeline.py --img 007759480.png --root /home/ec2-user/SageMaker/ancestry-demo --gpu 0
    """
    main()
