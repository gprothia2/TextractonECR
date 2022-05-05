def _get_gt_block(gt_words, index_range):
    (i_min, i_max) = index_range
    if (i_min, i_max) == (-1, -1):
        return ""
    else:
        return " ".join(gt_words[i_min:i_max])


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


def get_gt_file_key(key, ocr_files):
    for f in ocr_files:
        ocr_key = f.split("_")[-1]
        if key == ocr_key:
            return f
    else:
        return None


def get_key(gt_file_name):
    return gt_file_name.split("_")[-1]


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
        if score < 0:
            score = 0
        if char_len != 0:
            err_score = score / char_len
            scores.append(err_score)
        else:
            scores.append(0)
        edit_score_sum += err_score * char_len / total_char_len
    return scores, edit_score_sum


def compute_textract_edit_distance(file_name, ocr_files):
    try:
        gt_file_key = get_gt_file_key(file_name, ocr_files)
        if gt_file_key:
            gt_file_name = ocr_path + "gt_label/" + gt_file_key
            textract_words = get_textract_words("experiment/ocr_1/" + file_name)
            gt_words = get_gt_words(gt_file_name)
            block_score, page_score = compute_edit_distance_percentage(textract_words, gt_words)
            return (block_score, page_score)
        else:
            print(file_name + "not found")

    except:
        print("ground truth %s not found", file_name)
