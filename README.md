# Simple Video Object Detection using Opencv Dnn, Tensorflow, Pytorch
This project is a simple opencv, tensorflow, pytorch implementation of **Faster RCNN**, **Mask RCNN**, **YOLO**. 
The purpose of this project is to implement a simple object detection program using various frameworks.
This project does not have codes for training. It goes through following processes:

1. Reads video file
2. Detects objects with pretrained models (Trained on **COCO**)
3. Draws bounding boxes and labels.

<br>
While implementing, I referred to implementations below: 

* Opencv Dnn, Pytorch FasterRCNN - https://github.com/spmallick/learnopencv<br>
* Pytorch-Yolo - https://github.com/ayooshkathuria/pytorch-yolo-v3<br>
* Tensorflow, Tensorflow-Yolo - https://github.com/wizyoung/YOLOv3_TensorFlow<br>

## Requirements

* Python 3.6
* torch 1.0<br>
```pip install torch==1.0```
* torchvision 0.3.0
* opencv-python 4.1.0.25
* imutils 0.5.2<br>
```pip install torchvision opencv-python imutils```
* tensorflow 1.14.0
* tensorflow-gpu 1.1.0<br>
```pip install tensorflow tensorflow-gpu```
<br/>




##Pretrained Models

### 1. Tensorflow Object Detection API

1.1 Go to Tensorflow Object Detection API page
<br/>
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

<img src="readme/tensorflow_api.png" width="600px"/>

1.2 Find table as in image, download COCO pre-trained model.

1.3 In order to use pre-trained model in **Opencv**, convert `.pb` file to `.pbtxt`<br>
For example) if downloaded pretrained model is faster rcnn
```
$ python tf_text_graph_faster_rcnn.py --input /path/to/.pb --output /path/to/.pbtxt --config /path/to/pipeline.config
```

### 2. Yolo
1.1 Go to Yolo page.<br/>
https://pjreddie.com/darknet/yolo/

1.2 Download pre-trained model(The project was tested only on Yolov3)

### 3. Pytorch pre-trained models

    .
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...



## Demo
#### FasterRCNN Opencv
```
python --video assets/cars.mp4 --pbtxt data/graph.pbtxt --frozen data/frozen_inference_graph.pb --conf 0.5
```
<br/>

#### MaskRCNN Opencv
```
python --video assets/cars.mp4 --pbtxt data/graph.pbtxt --frozen data/frozen_inference_graph.pb --conf 0.5 --mask 0.3
```
<br/>

#### Yolo Opencv
```
python --video assets/cars.mp4 --config data/yolov3.config --weight data/yolov3.weights --conf 0.5 --nms 0.4 --resol 416
```
<br/>

#### FasterRCNN Pytorch
```
python --video assets/cars.mp4 --conf 0.5
```
<br/>

#### Yolo Pytorch
```
python --video assets/cars.mp4 --config data/yolov3.config --weight data/yolov3.weights --conf 0.5 --nms 0.4 --resolution 416
```
<br/>