SCALER = 65535
import xml.etree.ElementTree as ET

def _get_bbox_from_label_file(label_name, w, h):
    """
    Collect the bbox from the xml file

     Args:
        label_name: (str) label file name

    Returns:
        class_bboxs: (dict) key: numerical class name(converted by the class_to_int)
                            value: list of bboxs, converted to match image w and h 
    """
    # class_to_int = {'image':0,'article':1,'obit':2,'marriage':3,'other':4,'delete':5,'background':6}
    # WARN: for new dataset, we got only 1 class which is article
    class_to_int = {'article':1}
    class_bboxs = {}
    tree = ET.parse(label_name)
    root = tree.getroot()
    for child in root:
        if child.tag == 'particle':
            if child.attrib['classification'] not in class_to_int:
                continue
            _class = class_to_int[child.attrib['classification']]
            _bboxs = []
            for subchild in child.iter():
                if subchild.tag == 'subparticle':
                    bbox = [int(x) for x in subchild.attrib['rect'].split(',')]
                    x_min, y_min, x_max, y_max = bbox
                    #scale bbox 
                    x_min, y_min, x_max, y_max = int(x_min / SCALER * w), int(y_min / SCALER * h),\
                                                 int(x_max / SCALER * w), int(y_max / SCALER * h) 
                    
                    _bboxs.append([x_min, y_min, x_max, y_max])
            if _class in class_bboxs:
                class_bboxs[_class] = class_bboxs[_class] + _bboxs
            else:
                class_bboxs[_class] = _bboxs
    return class_bboxs