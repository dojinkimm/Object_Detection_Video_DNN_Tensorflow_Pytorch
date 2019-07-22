# Object Detection using Opencv Dnn, Tensorflow, Pytorch
This project is a simple opencv, tensorflow, pytorch implementation of **Faster RCNN**, **Mask RCNN**, **YOLO**. 
The purpose of this project is to implement a simple object detection program using various frameworks.
This project does not have codes for training. It goes through following processes:

1. Reads video file
2. Detects objects with pretrained models 
3. Draws bounding boxes and labels.

<br>
While implementing, I referred to implementations below: 

* Opencv Dnn, Pytorch FasterRCNN - https://github.com/spmallick/learnopencv<br>
* Pytorch-Yolo - https://github.com/ayooshkathuria/pytorch-yolo-v3<br>
* Tensorflow

## Requirements

####Environment

* Python 3.6
* torch 1.0<br>
```pip3 install torch==1.0```
* torchvision 0.3.0
* opencv-python 4.1.0.25
* imutils 0.5.2<br>
```pip3 install torchvision opencv-python imutils```
* tensorflow 1.14.0
* tensorflow-gpu 1.1.0
<br>
```pip3 install tensorflow tensorflow-gpu```

##Pretrained Models

#### 1. Tensorflow Object Detection API

1.1 Go to Tensorflow Object Detection API page
<br>
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

<img src="readme/tensorflow_api.png"/>

1.2 Download pre-trained model and unpack the folder and move to working repository

1.3 In order to use pre-trained model in **Opencv**, convert `.pb` file to `.pbtxt`<br>
For example) if downloaded pretrained model is faster rcnn
```
$ python --input /path/to/.pb --output /path/to/.pbtxt --config /path/to/pipeline.config
```

#### 2. Yolo
1.1 Go to Yolo page.<br>
https://pjreddie.com/darknet/yolo/

1.2 Download pre-trained model(The project was tested only on Yolov3) and unpack the folder and move to working repository

#### 3. Pytorch pre-trained models
