import os
from PIL import Image
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import numpy as np


from util import bb_is_inside, dist_of_two_point, load_checkpoint
import torch
import torchvision.transforms as transforms
import boto3
import net

from bbox_post_processing_original import post_process_bboxes
from bbox_post import (
    remove_overlapping_bboxes,
    clustering_bboxes,
    vertical_merge,
    adjust_from_boundary,
    check_bboxes_are_valid,
)

def is_vertically_aligned(bbox1, bbox2, thresh=0.05):
    '''
    @Param: 
        bbox1 and bbox2: two input bounding boxes to be checked for vertical alignement
            Bounding boxes are in (x_min, y_min, x_max, y_max) format
        thresh: misalignment tolerance with respect to the average width of bbox1 and bbox2
    '''
    w1 = bbox1[2] - bbox1[0]
    w2 = bbox2[2] - bbox2[0]
    w = (w1 + w2)/2
    return ((abs(bbox1[0] - bbox2[0]) < thresh*w) and (abs(bbox1[2] - bbox2[2]) < thresh*w))


def is_horizontally_aligned(bbox1, bbox2, thresh=0.05):
    '''
    @Param: 
        bbox1 and bbox2: two input bounding boxes to be checked for horizontal alignement
            Bounding boxes are in (x_min, y_min, x_max, y_max) format
        thresh: misalignment tolerance with respect to the average width of bbox1 and bbox2
    '''
    h1 = bbox1[3] - bbox1[1]
    h2 = bbox2[3] - bbox2[1]
    h = (h1 + h2)/2
    return ((abs(bbox1[1] - bbox2[1]) < thresh*h) and (abs(bbox1[3] - bbox2[3]) < thresh*h))


def combine_group(group):
    '''
    @Param: 
        group: a list of bounding boxes
    @Return: a single bounding box which computed from the
        combination of the input bounding boxes
    '''
    x_min = 10000
    y_min = 10000
    x_max = 0
    y_max = 0
    for box in group:
        x_min = min(x_min, box[0])
        y_min = min(y_min, box[1])
        x_max = max(x_max, box[2])
        y_max = max(y_max, box[3])
        
    return (x_min, y_min, x_max, y_max)

def area(bbox):
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])

def combine_bboxes(bboxes_original, max_area=None, thresh=0.05):
    '''
    @Param:
        bboxes: The list of all of the image bounding boxes
        max_area: The area of the largest bounding box taht Textract can digest
    
    @Return:
        A list of super boxes computed from vertically aligned bounding boxes
    '''
    bboxes = bboxes_original.copy()
    super_boxes = []
    while len(bboxes) > 0:
        group = []
        pivot = bboxes.pop(0)
        for bbox in bboxes:
            if is_vertically_aligned(bbox, pivot, thresh=thresh):
                combined = combine_group([bbox, pivot])
                if (max_area is not None) and (area(combined) > max_area):
                    break
                else:
                    pivot = combined
                    group.append(bbox)
        super_boxes.append(pivot)
        for bbox in group:
            bboxes.remove(bbox)
    return super_boxes

def get_segement_id(word_box, segments):
    """
    Get the article id of each word based on the bbox coordinates 
    """
    all_candidates = []
    for idx, box in enumerate(segments):
        if bb_is_inside(word_box, box):
            all_candidates.append(idx)
    if len(all_candidates)>0:
        return max(all_candidates)
    else:
        return None
    
def get_closest_segment(word_box, segments): 
    """
    Find the closest segment for each word based on the bbox coordinates if cannot find the bbox within
    """
    close_dist = 10e4                                        
    word_center = ((word_box[0]+ word_box[2])/2, (word_box[1]+ word_box[3])/2)
    for idx, box in enumerate(segments):
        box_center =  ((box[0]+ box[2])/2, 
                       (box[1]+ box[3])/2)
                                            
        box_upper_left = (box[0], box[1])
        box_lower_right = (box[2], box[3])
        dist_to_upper_left = dist_of_two_point(word_center, box_upper_left)
        dist_to_lower_right = dist_of_two_point(word_center, box_lower_right)
        dist = (dist_to_upper_left+dist_to_lower_right)/2
        if dist <  close_dist:
            close_dist = dist
            segment = idx
    return segment

def reorder_word(texts_w_coordinates_dict, segments):
    """
    Main function to reorder the word based on the order of the new segments 
    """
    len_of_not_found_word = 0
    for word in texts_w_coordinates_dict:
        word_box = (word['x_min'], word['y_min'], word['x_max'], word['y_max'])
        segment  =  get_segement_id(word_box, segments)
        if segment is None:
            #logging.info('Did not find corresponding segment for word {}'.format(word['word']))
            #logging.info('try find the closest segment...')
            print('Did not find corresponding segment for word {}'.format(word['word']))
            print('try find the closest segment...')
            #TODO: needs further improvement and decide whether to include these words
            segment = get_closest_segment(word_box, segments)
            #logging.info('Assigning word {} to segment {} ...'.format(word['word'], segment))
            print('Assigning word {} to segment {} ...'.format(word['word'], segment))
            word['read_order_segment_id'] = segment
            len_of_not_found_word +=1
        else:
            word['read_order_segment_id'] = segment
    #logging.info('Did not find corresponding segment for {} words'.format(len_of_not_found_word))
    print('Did not find corresponding segment for {} words'.format(len_of_not_found_word))
    texts_w_coordinates_dict.sort(key=lambda x: (x['read_order_segment_id']))
    return texts_w_coordinates_dict

def post_processing_module(raw_bboxes, w, h):
    """
    :param image: numpy (image data
    :param raw_bbox_fname: str
    :return: a list of bbox
    """
    less_bboxes = remove_overlapping_bboxes(raw_bboxes, r=0.8)
    if len(less_bboxes) > 6:
        #reduce bboxes
        post_bboxes = post_process_bboxes(less_bboxes, w)
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

def close_gaps(raw_bboxes, w, h, threshold =0.001, width_ratio_to_pad=0.0125, height_ratio_to_pad=0.005):
    new_bboxes = []
    #simply expand a few pixels 
    for idx, box in enumerate(raw_bboxes):
        pixels_to_pad_horizontal = int(width_ratio_to_pad * w)
        pixels_to_pad_vertical = int(height_ratio_to_pad * h)
        #pixels_to_pad_horizontal = 50
        #pixels_to_pad_vertical = 50
        expand_box = (max(box[0]- pixels_to_pad_horizontal, 0), 
                      max(box[1]- pixels_to_pad_vertical, 0),
                      min(box[2]+ pixels_to_pad_horizontal, w),
                      min(box[3]+ pixels_to_pad_vertical, h))
        
        new_bboxes.append(expand_box)
    #TODO: more complicated way to close gaps 
    # identify bbox idxes belong to same horizontal group 
    # close horizontal gaps 
    # identify bbox idxes belong to same vertical group 
    # close vertical gaps
    
#     for idx, box in enumerate(raw_bboxes):
#         current_x_max = box[2]
#         current_y_max = box[3]

#         reach_to_x_end = (w - current_x_max) < threshold*w
#         if idx > 0:
#             last_y_max = new_bboxes[idx-1][3]
            
#         reach_to_y_end = current_y_max > last_y_max
#         if not reach_to_x_end:
#             if box[0] > new_bboxes[idx-1][2]:  #to the right the previous bbox
#                 expand_box = (new_bboxes[idx-1][2], box[1], box[2], box[3])
#                 new_bboxes.append(expand_box)
#             else: #to the left the previous bbox 
                
                
#             need_to_close_gap = True
#             next_x_start = current_x_max
            
#         else:
#             new_bboxes.append(box)
    return new_bboxes
            
    
def get_image_from_s3(s3_dir, input_dir):
    print("Downloading data from S3...")
    s3_dir = s3_dir.split("/")
    bucket = s3_dir[0]
    key_file = "/".join(s3_dir[1:])
    # check local path
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    local_file = os.path.join(input_dir, s3_dir[-1])
    boto3.resource("s3").Bucket(bucket).download_file(key_file, local_file)
    print("Done: the file {} is downloaded".format(local_file))
    return local_file


def upload_data_to_s3(path, bucket, prefix):
    # @TODO: Tianyu designed the beginning of the pipeline, we need to agree what process is the easiest for user
    """

    # original data location in local instance, take one image as an example
    raw_data_dir = '../tianyu/FCN_Newspaper/data/gt_data/Images_PNG/1700-1899/'
    # S3 target bucket path and file prefix
    bucket = 'ancestry-demo'
    prefix = 'raw-image-data'
    # upload data into target bucket on Amazon S3
    upload_data_to_s3(raw_data_dir, bucket, prefix)
    """
    print("Uploading data...")
    for file in [os.path.join(path, f) for f in os.listdir(path)]:
        target_file = prefix + "/" + file.split("/")[-1]
        boto3.Session().resource("s3").Bucket(bucket).upload_file(file, target_file)
    print("Done: {} files uploaded to {}".format(len(os.listdir(path)), bucket + "/" + prefix))


def resize_image(image_file, grid_size):
    # default is 512 x 512, wont add argument(w and h) in this function
    IMG_W, IMG_H = grid_size, grid_size
    # /home/ec2-user/SageMaker/test_02_25/EulerEye/demo_image/007759480.png
    image_dir = "/".join(image_file.split("/")[:-1])
    file_name = image_file.split("/")[-1]
    if '.' not in file_name:
        file_name = file_name + '.png'
    image = Image.open(image_file)
    original_w, original_h = image.size
    image = image.resize((IMG_W, IMG_H))
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    resized_iamge = os.path.join(image_dir, "resized_" + file_name)
    rgb_image.save(resized_iamge)

    return (original_w, original_h), resized_iamge


def image_to_tensor(resize_image):
    image_transformer = transforms.Compose([transforms.ToTensor()])
    image = Image.open(resize_image)
    image_tensor = image_transformer(image)[0] #only take the grayscale 
    image_tensor = image_tensor.unsqueeze(0) #expand on the channel dim 
    image_tensor = image_tensor.unsqueeze(0) #expand on the batch size dim
    return image_tensor

def image_to_tensor_3_channel(resize_image):
    image_transformer = transforms.Compose([transforms.ToTensor()])
    image = Image.open(resize_image)
    image_tensor = image_transformer(image) 
    image_tensor = image_tensor.unsqueeze(0) #expand on the channel dim 
    #image_tensor = image_tensor.unsqueeze(0) #expand on the batch size dim
    return image_tensor


def load_checkpoint(checkpoint, model):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint


def get_fcn_mask(s3_file, input_dir, model_file, output_dir):
    """
    Predict a mask from the image on S3.
    Args:
        s3_file(string): image file on s3
        input_dir(string): image path on local instance
        model_file(string): FCN_model path
        output_dir(string): mask path on local instance
    """
    # download data from s3
    image_file = get_image_from_s3(s3_file, input_dir)

    # resize the data to 512 x 512
    original_size, resized_iamge = resize_image(image_file)

    # convert image to torch tensor
    image_tensor = image_to_tensor_3_channel(resized_iamge)
    # load fcn model
    model = net.FCN()

    load_checkpoint(model_file, model)

    # get output
    output = model(image_tensor)

    # save output
    image_id = s3_file.split("/")[-1].split(".")[0]

    # check output path on local instance
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output = output.data.cpu().numpy()
    output_file_name = os.path.join(output_dir, image_id + ".csv")
    np.savetxt(output_file_name, output.reshape(512, 512), delimiter=",")
    print("Done: image mask is generated and saved in {}".format(output_file_name))
    return output_file_name, output

def save_segments(bboxes, bbox_dir, key):
    np.savetxt("{}/{}.csv".format(bbox_dir, key), bboxes, delimiter=",", fmt="%i")

def get_ordered_segments(s3_file, input_dir, model_file, output_dir, device, grid_size=512, threshold=0.001):
    """Get the ordered segments by applying inference 

    Args:


    """
    # download data from s3
    image_file = get_image_from_s3(s3_file, input_dir)

    # resize the data to 512 x 512
    original_size, resized_iamge = resize_image(image_file, grid_size)
    
    original_w, original_h = original_size
    
     # convert image to torch tensor
    image_tensor = image_to_tensor(resized_iamge)
    
    # load fcn model
    model = net.FCN().to(device)

    load_checkpoint(model_file, model)
    model.eval()
    # get output
    output = model(image_tensor.float().to(device))

    _pred_mask = output[0,:,:,:].cpu().detach().numpy().reshape(grid_size, grid_size)
    _pred_mask_binary = _pred_mask > 0.5
    label_image = label(_pred_mask_binary,connectivity=2)
    raw_bboxes = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= int(grid_size*grid_size*threshold):
            raw_bboxes.append((int(region.bbox[1]/grid_size*original_w),
                           int(region.bbox[0]/grid_size*original_h),
                           int(region.bbox[3]/grid_size*original_w), 
                           int(region.bbox[2]/grid_size*original_h)))
                          
    raw_bboxes.sort(key=lambda x: (x[1], x[0]))
    
    #post-process bbox
    #bboxes = post_processing_module(raw_bboxes, original_w, original_h)
    bboxes = close_gaps(raw_bboxes, original_w, original_h)
    #bboxes = clustering_bboxes(raw_bboxes, r=200)
    bboxes.sort(key=lambda x: (x[1], x[0]))
    
    image_id = s3_file.split("/")[-1].split(".")[0]
    save_segments(raw_bboxes, output_dir+"_raw", image_id)
    save_segments(bboxes, output_dir, image_id)
    
    #combine bboxes 
    combined_bboxes = combine_bboxes(bboxes)
    save_segments(combined_bboxes, output_dir, image_id+"_combined")
    return bboxes, combined_bboxes

