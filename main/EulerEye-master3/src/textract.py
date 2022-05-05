"""
pipeline.py
----------------
end-to-end pipeline to wrap the high level functions

@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   08/13/2020
"""
import boto3
import numpy as np


def initiate_file(file_name):
    file = open("%s" % (file_name), "a")
    file.seek(0)
    file.truncate()
    file.flush()
    return file


def write_file(result_file, lines):
    file = initiate_file(result_file)
    for line in lines:
        file.write(" ".join(line) + "\n")
    file.flush()


def save_texts(texts, path):
    with open(path, "w") as f:
        for text in texts:
            f.write(text)
            f.write("\n")


def textract_text_only(bucket, document, w, h, segment_id):
    """
    :param bucket:
    :param document:
    :return:
    """
#     textract = boto3.client("textract")
#     response = textract.detect_document_text(
#         Document={"S3Object": {"Bucket": bucket, "Name": document}}
#     )
#     # Detect columns and print lines
#     columns = []
#     lines = []
#     for item in response["Blocks"]:
#         if item["BlockType"] == "LINE":
            
#             lines.append((item["Geometry"]["BoundingBox"]["Top"], 
#                           item["Geometry"]["BoundingBox"]["Left"], 
#                           item["Text"]))         

#     lines.sort(key=lambda x: (x[0], x[1]))
#     res = []
#     for line in lines:
#         res.append(line[2])
    
    words_dict = textract_bbox_text(bucket, document, w, h, segment_id)
    res = ' '.join([k['word'] for k in words_dict])
    return res


def textract_bbox_text(bucket, document, w, h, segment_id):
    """
    :param bucket: s3 bucket name where file is uploaded to 
    :param document: s3_path + file_name
    :return: a list of dict with words and local coordinates 2d tuple of bounding boxes and words
    """
    # Detect text in the document
    client = boto3.client("textract")
    # Process using S3 object
    response = client.detect_document_text(
        Document={"S3Object": {"Bucket": bucket, "Name": document}}
    )
    # Get the text blocks
    blocks = response["Blocks"]
    words_dict = []
    #this returns line by line scan from top to bottom, left to right 
    for block in blocks:
        if block['BlockType'] == "WORD":
            words_dict.append(
                {
                    'word': block["Text"],
                    'segment': segment_id, 
                    'confidence':  block["Confidence"],
                    'x_min': int(w*(block["Geometry"]["BoundingBox"]["Left"])),
                    'y_min': int(h*(block["Geometry"]["BoundingBox"]["Top"])),
                    'x_max': int(w*(block["Geometry"]["BoundingBox"]["Left"] + block["Geometry"]["BoundingBox"]["Width"])),
                    'y_max': int(h*(block["Geometry"]["BoundingBox"]["Top"] + block["Geometry"]["BoundingBox"]["Height"])),
                }
            )

    return words_dict


flatten = lambda l: [item for sublist in l for item in sublist]


def translate_word_bboxes(local_word_dicts, x, y):
    """
    Translate the local bounding boxes back to global
    Parameters
    ----------
    local_word_dicts: local word dict with coordinates 
    x: global x_min
    y: global y_min

    Returns
    -------
    """
    res = []
    for w in local_word_dicts:
        w['x_min'] += x
        w['x_max'] += x
        w['y_min'] += y
        w['y_max'] += y
        res.append(w)
    return res
