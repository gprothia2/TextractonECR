"""
text_analysis.py
----------------
Computing OCR Accuracy and pre-process texts and string

@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   04/18/2020
"""

# Global Variables
BUCKET_NAME = "ancestry-us-west-1"
ocr_path = "/home/ec2-user/SageMaker/ocr/"
SM_PATH = "/home/ec2-user/SageMaker/"

import re
import datetime, time
from collections import OrderedDict
import tqdm
from jiwer import wer
import editdistance
import Levenshtein as lev
import edit_distance
import pickle as pkl
import itertools

from data_loading import read_text_from_s3
from file_util import initiate_file

############################################################################
#                                UTILITIES                                 #
############################################################################

flatten = lambda l: [item for sublist in l for item in sublist]


def getImageKey(jp2_image):
    _image = jp2_image.split("/")
    year, image_id = _image[4], _image[5].split(".")[0]
    return (year, image_id)


############################################################################
#                        BUILD RESULT TO MAPPING                           #
############################################################################
def clean_pred(samples):
    """
    samples: a list of string, each string corresponds to a segment
    """
    res = []
    for sample in samples:
        res.append(clean_string(sample).split())
    return res


def load_raw_file(file_path):
    res = []
    with open(file_path, "r") as f:
        for line in f:
            res.append(line)
    return res


def get_gt_words(file_path):
    gt_sample = load_raw_file(file_path)
    text_gt = " ".join(gt_sample)
    clean_gt = clean_string(text_gt)
    return clean_gt.split()


def get_textract_words(file_path):
    pred_sample = load_raw_file(file_path)
    return clean_pred(pred_sample)


def delete_duplicates(l):
    return list(dict.fromkeys(l))


def build_block_indices_mapping(gt_words, block):
    indices = OrderedDict()
    for ind, word in enumerate(block):
        if word in gt_words:
            index = [i for i, x in enumerate(gt_words) if x == word]
            if ind in indices:
                indices[ind] += index
            else:
                indices[ind] = []
                indices[ind] += index
        else:
            indices[ind] = []
    for ind in indices:
        indices[ind] = delete_duplicates(indices[ind])
        # remove the high-frequency words
        if len(indices[ind]) > 3:
            indices[ind] = []
    return indices


############################################################################
#               APPROACH 1: LONGEST INCREASING SUBSEQUENCE                 #
############################################################################


def clean_indices_mapping(block_ind_to_gt_ind):
    final_mapping = OrderedDict()
    pivot_block = -1
    pivot_gt = -1
    for block_ind in block_ind_to_gt_ind:
        gt_ind = block_ind_to_gt_ind[block_ind]
        if len(gt_ind) == 1:
            pivot_block = block_ind
            pivot_gt = gt_ind[0]

    for block_ind in block_ind_to_gt_ind:
        gt_ind = block_ind_to_gt_ind[block_ind]
        diff_min = 1000
        if gt_ind:
            for gt_i in gt_ind:
                if block_ind < pivot_block:  # if index is in the left side of pivot 100, 101
                    # select the index closest to
                    index_diff = pivot_block - block_ind
                    guess_gt_index = pivot_gt - index_diff
                else:
                    index_diff = block_ind - pivot_block
                    guess_gt_index = pivot_gt + index_diff

                real_index_diff_abs = abs(gt_i - guess_gt_index)
                if real_index_diff_abs < diff_min:
                    diff_min = real_index_diff_abs
                    final_mapping[block_ind] = gt_i
    return final_mapping


def longest_increasing_subsequence(nums):
    n = len(nums)
    dp = [0 for _ in range(n)]
    seq = []
    for i, val in enumerate(nums):
        for j in range(i):
            if 0 < val - nums[j] < n:
                dp[i] = max(dp[i], dp[j] + 1)
    tmp = max(dp)
    ind = []
    for i in range(n - 1, -1, -1):
        if dp[i] == tmp:
            ind.append(i)
            tmp -= 1
    ind.reverse()

    for i in ind:
        seq.append(nums[i])
    return seq


############################################################################
#                          APPROACH 2: USE PIVOT                           #
############################################################################


def clean_indices_mapping_2(block_ind_to_gt_ind):
    final_mapping = OrderedDict()

    for block_ind in block_ind_to_gt_ind:
        gt_ind = block_ind_to_gt_ind[block_ind]
        if len(gt_ind) == 1:
            pivot_block = block_ind
            pivot_gt = gt_ind[0]
            print("----- pivot -----")
            print(pivot_block, pivot_gt)
            print("-----------------")
            for block_ind in block_ind_to_gt_ind:
                gt_ind = block_ind_to_gt_ind[block_ind]
                diff_min = 1000
                if gt_ind:
                    for gt_i in gt_ind:
                        if (
                            block_ind < pivot_block
                        ):  # if index is in the left side of pivot 100, 101
                            # select the index closest to
                            index_diff = pivot_block - block_ind
                            guess_gt_index = pivot_gt - index_diff
                        else:
                            index_diff = block_ind - pivot_block
                            guess_gt_index = pivot_gt + index_diff

                        real_index_diff_abs = abs(gt_i - guess_gt_index)
                        if real_index_diff_abs < diff_min:
                            diff_min = real_index_diff_abs
                            final_mapping[block_ind] = gt_i
        else:
            index_list = flatten(list(block_ind_to_gt_ind.values()))
            pivots = longest_increasing_subsequence(index_list)
            for block_ind in block_ind_to_gt_ind:
                gt_ind = block_ind_to_gt_ind[block_ind]
                if gt_ind:
                    if pivots[0] in gt_ind:
                        pivot = pivots[0]
                        pivot_index = block_ind

    return final_mapping


############################################################################
#                     APPROACH 4: TWO POINTERS                             #
############################################################################

# PROBLEMS:
# It's not obvious to define when we should move pointers in different ways


def get_range_from_left(mapping):
    n = len(mapping)
    left, right = 0, len(mapping) - 1
    candidates_list = []
    while left < right:
        begin_indices = mapping[left]
        end_indices = mapping[right]
        # Enumerate all combinations
        range_candidate = [
            list(zip(x, end_indices))
            for x in itertools.permutations(begin_indices, len(end_indices))
        ]
        candidates = delete_duplicates(flatten(range_candidate))
        for r in candidates:
            if r[1] > r[0] and r[1] - r[0] < 1.2 * n:
                print("FIND!!!!", r)
                pivot_left, pivot_right = left, right
                index_diff = right - left
                real_right = r[1] + index_diff + 1
                return (pivot_left, real_right)
            else:
                print("Our assumption failed")
        candidates_list += candidates
        right -= 1
    return (-1, -1)


def get_range_from_2_directions(mapping):
    n = len(mapping)
    left, right = 0, len(mapping) - 1
    candidates_list = []
    while left < right:
        begin_indices = mapping[left]
        end_indices = mapping[right]
        # Enumerate all combinations
        range_candidate = [
            list(zip(x, end_indices))
            for x in itertools.permutations(begin_indices, len(end_indices))
        ]
        candidates = delete_duplicates(flatten(range_candidate))
        for r in candidates:
            if r[1] > r[0] and r[1] - r[0] < 1.2 * n:
                print("FIND!!!!", r)
                index_left_diff = n - left
                index_right_diff = right
                real_right = r[1] + index_left_diff + 1
                real_left = r[0] - index_right_diff
                return (real_left, real_right)
            else:
                print("Our assumption failed")
        candidates_list += candidates
        left += 1
        right -= 1
    return (-1, -1)


############################################################################
#                        APPROACH 3: HashMap with Score                    #
############################################################################
def enumerate_mapping(mapping):
    """
    :param mapping:
    :return: find a good one
    """
    scores = {}
    pivot_pair = {}
    for word_ind in mapping:
        gt_ind = mapping[word_ind]
        # 1. Search for the single one
        if len(gt_ind) == 1:
            candidate = fill_mapping(mapping, word_ind, gt_ind[0])
            score = candidate_score(mapping, candidate)
            scores[score] = candidate
            pivot_pair[score] = (word_ind, gt_ind[0])
        elif len(gt_ind) == 2:
            candidate_1 = fill_mapping(mapping, word_ind, gt_ind[0])
            score = candidate_score(mapping, candidate_1)
            scores[score] = candidate_1
            pivot_pair[score] = (word_ind, gt_ind[0])
            candidate_2 = fill_mapping(mapping, word_ind, gt_ind[1])
            score = candidate_score(mapping, candidate_2)
            scores[score] = candidate_2
            pivot_pair[score] = (word_ind, gt_ind[1])
        else:
            # print("word index = ", word_ind, 'gt index length != 1 & 2')
            # print('gt index', gt_ind)
            continue
    if scores:
        #         print('scores', scores)
        #         print('pivot pair', pivot_pair)
        seq = max(scores.items(), key=lambda k: k[0])[1]
        (i_min, i_max) = (seq[0], seq[-1])
        try:
            (ind_min, ind_max) = adjust_range((i_min, i_max), mapping)
        except:
            print((i_min, i_max))
            (ind_min, ind_max) = (seq[0], seq[-1])
    else:
        print("Error in mapping...")
        (ind_min, ind_max) = (-1, -1)
    return (ind_min, ind_max)


def closest(candidates, target):
    if candidates:
        closest_elem = candidates[
            min(range(len(candidates)), key=lambda i: abs(candidates[i] - target))
        ]
        # print("closest", closest_elem)
        return closest_elem
    else:
        print("candidates is empty")
        print(candidates, target)


def adjust_range(pred_range, mapping):
    (x_min, x_max) = pred_range
    new_min, new_max = -1, -1
    left, right = 0, len(mapping) - 1
    while left < right:
        if mapping[left]:
            if x_min in mapping[left]:
                new_min = x_min
            else:
                closest_x_min = closest(mapping[left], target=x_min)
                new_min = closest_x_min
        else:
            left += 1

        if mapping[right]:
            if x_max in mapping[right]:
                new_max = x_max
            else:
                closest_x_max = closest(mapping[right], target=x_max)
                new_max = closest_x_max
        else:
            right -= 1

        if new_min != -1 and new_max != -1:
            if left > 0:
                new_min = new_min - left
            if right < len(mapping) - 1:
                diff = len(mapping) - 1 - right
                new_max = new_max + diff
            break
    return (new_min, new_max)


def candidate_score(mapping, candidate):
    """
    :param mapping: (11, [111, 407]),
             (12, [112, 1611]),
             (13, [3368]),
    :param candidate:
    :return:
    """
    counter = 0
    n = len(mapping)
    for i in range(n):
        if mapping[i]:
            # print('mapping', mapping[i], 'candidate', candidate[i])
            closest_elem = closest(mapping[i], candidate[i])
            if candidate[i] in mapping[i] or abs(candidate[i] - closest_elem) < 3:
                counter += 1
    return counter


def fill_mapping(mapping, block_index, pivot_index):
    """
    :param mapping:
    :param block_index: word index in block
    :param pivot_index: word index in gt
    :return:
    >>> fill_mapping(mapping, 7, 2534)
    """
    res = []
    for i in range(len(mapping)):
        diff = pivot_index - block_index + i
        res.append(diff)
    return res


############################################################################
#                             WORDS BLOCK's RANGE                          #
############################################################################
def get_range(textract_words, gt_words):
    n = len(textract_words)
    ranges = []
    for i in range(n):
        block = textract_words[i]
        mapping = build_block_indices_mapping(gt_words, block)
        ranges.append(enumerate_mapping(mapping))
    return ranges


############################################################################
#                             READ ORDER ACCURACY                          #
############################################################################
def define_ocr_block_order(block_ranges):
    n = len(block_ranges)
    order = {}
    for i in range(n):
        order[i] = block_ranges[i][0]
    return order


def get_block_order(gt_words, textract_words):
    block_ranges = get_range(textract_words, gt_words)
    orders = define_ocr_block_order(block_ranges)
    order_map = sorted(orders.items(), key=lambda k: k[1])
    order = [x[0] for x in order_map]
    return order


def get_segment_order_accuracy(gt_words, textract_words):
    n = len(textract_words)
    order = get_block_order(gt_words, textract_words)
    # min_num_of_swaps = min_swaps(order)
    num_not_in_order = count_not_in_order(order)
    return 1 - num_not_in_order / n


def compute_seg_order_accuracy_by_index(index):
    with open(SM_PATH + "FCN_Newspaper/output_image/logistics/id_2_path.pkl", "rb") as f:
        id_to_path = pkl.load(f)
    image_file = id_to_path[index]
    image_key = getImageKey(image_file)
    # SM_PATH + '{}_{}.txt'.format(image_key[0], image_key[1])
    file_name = SM_PATH + "{}_{}.txt".format(image_key[0], image_key[1])
    textract_words = get_textract_words(file_name)
    gt_words = get_gt_words(file_name)
    return get_segment_order_accuracy(gt_words, textract_words)


def compute_seg_order_accuracy_by_file_name(file_name):
    textract_words = get_textract_words(file_name)
    gt_words = get_gt_words(file_name)
    return get_segment_order_accuracy(gt_words, textract_words)


def count_not_in_order(nums):
    """
    We define a position P is a peak if:

    A[P] > A[P-1] && A[P] > A[P+1]
    :param nums:
    :return:
    """
    counter = 0
    n = len(nums)
    for i in range(2, n - 1):
        if nums[i] - nums[i - 1] > 2:
            counter += 1
    return counter


def run_read_order_single_textract(file_name, res_file_path, textract_prefix=None):
    gt_file_name = ocr_path + "gt_label/" + file_name
    textract_file_name = ocr_path + textract_prefix + file_name
    gt_words = get_gt_words(gt_file_name)
    textract_words = get_textract_words(textract_file_name)
    page_score = get_segment_order_accuracy(gt_words, textract_words)
    line = file_name + "," + str(page_score) + "\n"
    with open(res_file_path, "a") as f:
        f.write(line)
    print(line)


def run_read_order_single_abbyy(file_name, res_file_path):
    abbyy_prefix = "abbyy_post_process_lower/"
    gt_file_name = ocr_path + "gt_label/" + file_name
    abbyy_file_name = ocr_path + abbyy_prefix + file_name
    gt_words = get_gt_words(gt_file_name)
    abbyy_words = partition(get_abbyy_words(abbyy_file_name), 100)
    page_score = get_segment_order_accuracy(gt_words, abbyy_words)
    line = file_name + "," + str(page_score) + "\n"
    with open(res_file_path, "a") as f:
        f.write(line)
    print(line)


############################################################################
#                      METRICS 1: EDIT DISTANCE                            #
############################################################################
def _get_gt_block(gt_words, index_range):
    (i_min, i_max) = index_range
    if (i_min, i_max) == (-1, -1):
        return ""
    else:
        return " ".join(gt_words[i_min : i_max + 1])


def get_gt_block(gt_words, textract_words):
    block_ranges = get_range(textract_words, gt_words)
    gt_blocks = []
    for r in block_ranges:
        gt_block = _get_gt_block(gt_words, r)
        gt_blocks.append(gt_block)
    return gt_blocks


def get_textract_block(textract_words):
    textract_blocks = []
    for block in textract_words:
        textract_blocks.append(" ".join(block))
    return textract_blocks


def compute_edit_distance(textract_words, gt_words):
    textract_blocks = get_textract_block(textract_words)
    gt_blocks = get_gt_block(gt_words, textract_words)
    n = len(textract_blocks)
    scores = []
    edit_score_sum = 0
    for i in range(n):
        score = editdistance.eval(gt_blocks[i], textract_blocks[i])
        scores.append(score)
        edit_score_sum += score
    return scores, edit_score_sum


def compute_edit_distance_percentage(textract_words, gt_words):
    textract_blocks = get_textract_block(textract_words)
    gt_blocks = get_gt_block(gt_words, textract_words)
    gt_page = " ".join(gt_blocks)
    total_char_len = len(gt_page)
    n = len(textract_blocks)
    scores = []
    edit_score_sum = 0
    for i in range(n):
        gt, textract = gt_blocks[i], textract_blocks[i]
        char_len = len(gt)
        score = editdistance.eval(gt, textract)
        if char_len != 0:
            err_score = 1 - score / char_len
            scores.append(err_score)
            weight = char_len / total_char_len
            print(weight)
            if weight > 1:
                print("impossible")
            edit_score_sum += err_score * char_len / total_char_len
        else:
            scores.append(0)

    return scores, edit_score_sum


def run_edit_distance_single(file_name):
    abbyy_prefix = "abbyy_post_process_lower/"
    gt_file_name = ocr_path + "gt_label/" + file_name
    abbyy_file_name = ocr_path + abbyy_prefix + file_name
    gt_words = get_gt_words(gt_file_name)
    abbyy_words = partition(get_abbyy_words(abbyy_file_name), 100)
    block_score, page_score = compute_edit_distance_percentage(abbyy_words, gt_words)
    line = file_name + "," + str(page_score)
    print(line)


def run_edit_distance(ocr_file_names, textract_file_path, gt_file_path, num_iter):
    """

    :param ocr_file_names:
    :param textract_file_path:
    :param gt_file_path:
    :param num_iter:
    :return:
    """
    import tqdm

    accuracy = {}
    failed_indices = []
    for i in tqdm.tqdm(range(num_iter)):
        try:
            file_name = ocr_file_names[i]
            textract_file_name = textract_file_path + file_name
            gt_file_name = gt_file_path + file_name
            textract_words = get_textract_words(textract_file_name)
            gt_words = get_gt_words(gt_file_name)
            accuracy[i] = compute_edit_distance_percentage(textract_words, gt_words)
        except:
            failed_indices.append(i)


def run_edit_distance_single_textract(file_name, res_file_path):
    textract_prefix = "textract_post_process_raj_06_16/"
    gt_file_name = ocr_path + "gt_label/" + file_name
    textract_file_name = ocr_path + textract_prefix + file_name
    gt_words = get_gt_words(gt_file_name)
    textract_words = get_textract_words(textract_file_name)
    block_score, page_score = compute_edit_distance_percentage(textract_words, gt_words)
    line = file_name + ", " + str(page_score) + "\n"
    with open(res_file_path, "a") as f:
        f.write(line)
    print(line)


def get_abbyy_words(file_path):
    sample = load_raw_file(file_path)
    text = " ".join(sample)
    clean = clean_string(text)
    return clean.split()


def partition(lst, n):
    res = []
    for i in range(0, len(lst), n):
        res.append(lst[i : i + n])
    return res


def compute_abbyy_edit_distance(file_keys, num, res_file_name):
    abbyy_prefix = "abbyy_post_process_lower/"
    file = initiate_file(res_file_name)
    ed_score = {}
    failed_indices = []
    for i in tqdm.tqdm(range(num)):
        try:
            file_name = file_keys[i]
            abbyy_file_name = ocr_path + abbyy_prefix + file_name
            gt_file_name = ocr_path + "gt_label/" + file_name
            abbyy_words = partition(get_abbyy_words(abbyy_file_name), 100)
            gt_words = get_gt_words(gt_file_name)
            block_score, page_score = compute_word_crossing_score(gt_words, abbyy_words)
            ed_score[i] = (block_score, page_score)
            line = file_name + "," + str(page_score) + "\n"
            file.write(line)
            file.flush()
        except:
            failed_indices.append(i)

    return ed_score, failed_indices


############################################################################
#                      TEXT & STRING OPERATIONS                            #
############################################################################
def remove_punctuation(string):
    # punctuation marks
    punctuations = """!()-[]{};:'"\,<>./?@#$%^&*_~"""

    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string


def remove_dash(string):
    for x in string.lower():
        if x == "—":
            string = string.replace(x, " ")
    return string


def find_all_tags(string):
    """
    :param string : str
        some string
    :return
        a list of str type tag
    """
    tag_regex = re.compile(r"(<[a-z]*>)")
    all_tags = tag_regex.findall(string)
    return list(set(all_tags))


def find_file_name(string_files, all_strings, tag):
    filename_to_string = dict(zip(string_files, all_strings))
    file_names = []
    for f, string in filename_to_string.items():
        if tag in string:
            file_names.append(f)
    return file_names


def remove_tags(string):
    string = re.sub(r"(<[a-z]*>)", "", string)
    string = re.sub(r"(</[a-z]*>)", "", string)
    return string


def remove_headers(string):
    string = string.replace("<h1>", "")
    string = string.replace("</h1>", "")
    return string


def remove_line_separator(string):
    strings = string.replace("\r", "")  # Windows line separator
    return strings.replace("\n", "")  # Linux / Unix line separator


def remove_unknown(string):
    return string.replace("[?]", "")


def remove_consecutive_white_spaces(string):
    return " ".join(string.split())


def clean_string(string):
    return remove_consecutive_white_spaces(
        remove_dash(
            remove_unknown(
                remove_line_separator(
                    remove_punctuation(remove_headers(remove_tags(string.lower())))
                )
            )
        )
    )


def get_clean_res(file_name):
    data = read_text_from_s3(BUCKET_NAME, file_name)
    processed_data = clean_string(data)
    return processed_data


def get_raw_res(file_name):
    raw_data = read_text_from_s3(BUCKET_NAME, file_name)
    return raw_data


def process_abbyy(string):
    """

    :param string:
    :return:
    """
    return remove_line_separator(string)


############################################################################
#                          Error Analysis
############################################################################
def compute_file_length(file_key):
    """
    :param file_key:
    :return:
    """
    file_dict = generate_file_name_for_error(file_key)
    abbyy = file_dict["abbyy"]
    stringract = file_dict["textract"]
    ground_truth = file_dict["ground_truth"]

    abbyy_res = get_clean_res(abbyy)
    stringract_res = get_clean_res(stringract)
    gt_res = get_clean_res(ground_truth)

    abbyy_num_of_words = len(abbyy_res.split())
    stringract_num_of_words = len(stringract_res.split())
    ground_truth_num_of_words = len(gt_res.split())

    return {
        "abbyy": abbyy_num_of_words,
        "textract": stringract_num_of_words,
        "ground_truth": ground_truth_num_of_words,
    }


def compute_file_length_all(file_keys, length_file_name):
    counter = 0
    failed_jobs = []
    total_iteration = len(file_keys)
    time_begin = datetime.datetime.fromtimestamp(time.time())
    file = initiate_file(length_file_name)
    for key in file_keys:
        try:
            length_dict = compute_file_length(key)
            abbyy_len = length_dict["abbyy"]
            string_len = length_dict["stringract"]
            gt_len = length_dict["ground_truth"]
            line = key + " " + abbyy_len + " " + string_len + " " + gt_len + "\n"
            file.write(line)
        except:
            print(key + "is not done")
            failed_jobs.append(key)
        counter += 1
        time_end = datetime.datetime.fromtimestamp(time.time())
        print("Time elapsed: ", str(time_end - time_begin))
        print_progress_bar(counter, total_iteration)
    file.flush()
    return failed_jobs


############################################################################
#                          Error Computation
############################################################################
def error_metric(metric, ground_truth, hypothesis):
    switcher = {
        "wer": wer(hypothesis, ground_truth, standardize=True),
        "editdistance": editdistance.eval(ground_truth, hypothesis),
        "lev": lev.distance(ground_truth, hypothesis),
        "edit_distance": edit_distance.SequenceMatcher(
            a=ground_truth,
            b=hypothesis,
            action_function=edit_distance.highest_match_action,
        ).ratio(),
    }
    return switcher.get(metric, "nothing")


def get_error(file_key, option, metric):
    res_dict = generate_file_name_for_error(file_key)
    abbyy = res_dict["abbyy"]
    stringract = res_dict["stringract"]
    ground_truth = res_dict["ground_truth"]

    if option == "raw":
        abbyy_res = get_raw_res(abbyy)
        stringract_res = get_raw_res(stringract)
        ground_truth_res = get_clean_res(ground_truth)
    elif option == "clean":
        abbyy_res = get_clean_res(abbyy)
        stringract_res = get_clean_res(stringract)
        ground_truth_res = get_clean_res(ground_truth)
    else:
        raise ValueError("Error option is not specified, either raw or clean!")

    error_abbyy = error_metric(metric, ground_truth_res, abbyy_res)
    error_stringract = error_metric(metric, ground_truth_res, stringract_res)
    return (file_key, error_abbyy, error_stringract)


def run_error(file_keys, error_file_name, option, metric):
    counter = 0
    res = []
    failed_jobs = []
    total_iteration = len(file_keys)
    time_begin = datetime.datetime.fromtimestamp(time.time())
    [file_name, ext] = error_file_name.split(".")
    err_file_name = file_name + "_" + option + "_" + metric + "." + ext
    file = initiate_file(err_file_name)
    for key in file_keys:
        try:
            (_, abbyy_err, stringract_err) = get_error(key, option, metric)
            line = key + " " + str(abbyy_err) + " " + str(stringract_err) + "\n"
            print(line)
            res.append(line)
        except:
            print(key + "is not done")
            failed_jobs.append(key)
        counter += 1
        time_end = datetime.datetime.fromtimestamp(time.time())
        print("Time elapsed: ", str(time_end - time_begin))
        print_progress_bar(counter, total_iteration)
    for line in res:
        file.write(line)
    file.flush()
    return failed_jobs
