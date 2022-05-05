# EulerEye

A suite of processing modules to enhance the use of Amazon Textract.

> If you have any questions, feel free to ping us:`@yaguan`, `@ttizha`, and `@wenzhu`

If you have any detailed question to understand the code, you can find the author on top of each module.

## Install
The environment in SageMaker to use is PyTorch
```
/home/ec2-user/anaconda3/envs/pytorch_p36/bin/python
```

### Set up environment

All needed packages are in the `requirements.txt`, in terminal, use the following command line:
```
pip install -r requirements.txt
```

Note that calling Textract from SageMaker requires SageMaker notebook be in the same AZ 
(availability zone) with S3 bucket, 
so you also need to make sure your `S3_BUCKET_NAME` is defined as the same as the S3 bucket you created.
This can be easily modified at `src/constants.py`

## Demo
To try this pipeline, go to `src` directory and run the example command:

```sh
python src/pipeline.py --img $your_img_name.png --root /$your_path/EulerEye --coords 1
```

Example

```sh
python src/pipeline.py --img 007759480.png --root /home/ec2-user/SageMaker/EulerEye --coords 1
```

Explaination 3 arguments:
* `img` : given test image path
* `root` : given root folder path
* `coords` : 1 or 2,
    * 1: OCR results is plain texts with coordinates
    * 2: OCR results as plain texts only
  
### Structure Desgin:
```
ancestry-demo
├── src/
│     ├── bbox_post_processing.py  : post process to correct bounding boxes
│     ├── constant.py : holds hard-coded S3 bucket name
│     ├── data_loading.py : communicate with S3 bucket for textract
│     ├── fcn.py : mask generation
│     ├── feature_extractor.py : supporting module for FCN
│     ├── text_analysis : metrics + functions to analyze the text ground truth
│     ├── file_util.py : supporting textract and some other results writing
│     ├── graph.py : supporting data structures for wenzhen bboxes processing
│     ├── image_processing.py : image processing functions supporting textract pipeline
│     ├── mask_2_bbox.py : converting FCN mask into bounding boxes (raw)
│     ├── net.py : supporting fcn
│     ├── script.py (demo)
│     ├── text_analysis.py : support metrics evaluation + align gt / textract / abbyy
│     ├── metric_evaluation.py 
│     └── textract.py : support the pipeline high-level textract pipeline
├── model/ 
│     └── best_1.5.pth.tar
├── demo_image/ 
│     ├── resized_015495008.png
│     ├── 015495008.png
│     └── ...
├── viualization/
│     └── visual_demo.nb
├── output/ (note we use one single file key to map all the intermediate results and final result)
│     ├── pred_mask -> predicted FCN masks (`.csv`)
│     ├── bbox_raw -> raw bounding boxes from morphology operations (`.csv`)
│     ├── bbox -> post-processed bounding boxes (`.csv`)
│     ├── cropped_image -> cropped images based on `bbox` (`.png`)
│     ├── ocr_plain_text -> plain OCR results as text file. (`.txt`)
│     └── ocr_coords -> OCR results with coordinates (x, y) (`.txt`)
├── Code-Walk-Through-MLSL.ipynb
└── README.md
```
## Visualization Demo
* Please read visualization demo at [link](visualization/visualization-demo.ipynb)
* For details, please read `Code-Walk-Through-MLSL.ipynb` at [link](/Code-Walk-Through-MLSL.ipynb)

## Technical Knowledge Sharing
* [Connectivity and Mathematical Morphology on Binary and Grayscale Images](https://wiki.solutions-lab.ml.aws.dev/display/TK/Connectivity+and+Mathematical+Morphology+on+Binary+and+Grayscale+Images)
* [Engagement Summary](https://wiki.solutions-lab.ml.aws.dev/display/ES/Ancestry)

## API Design Details

`mask_to_bbox.py`
