# Target Person Tracker

## Introduction
My initial goal with this project is a tracker with reidentification
cabilities to track the target person. I've developed this tracker to
put into a human tracking robot.

## System
* First the tracker is initialized with target features. If the target
features are given, the system starts tracking else it starts
collecting features from the centered person. In my case the target
should turn 360 degrees so it can be retracked from all angles. Only
features from the upper portion of the detection are collected. After
initialization the target's features are found in the current
frame. Then optical flow is initialized to track found features.
* Every nth frame the tracked points are populated by finding
target's features in the current frame.
* If tracked points are below a certain threshold, retracking is actived.
* Every nth frame new features are added to the target's features.
* Every nth frame target's features and environment features are
matched and matches are removed from the target's features to reduce
false positives.
* If there is no tracked points all detections are considered to find
target person so removing false positives is necessary step to eliminate
wrong tracking.

## Requirements
* OpenCV
* scipy
* sklearn
* numpy

### MobileNet SSD
* [Weights](https://github.com/chuanqi305/MobileNet-SSD)
* mobilenet_iter_73000.caffemodel
* Put mobilenet_iter_73000.caffemodel into mobilenetssd folder.
* You can also use --mobilenet_model argument to pass the model path.

### YOLOv3
* Compile [darknet](https://github.com/AlexeyAB/darknet) with shared library.
* Put the library into lib folder.
* You can also use DARKNET_LIB environment variable to pass the path.
* Weights, Config File and Data files should be downloaded.

### HOG
* imutils

## Usage

``` bash
	python tracker.py --detector mobilenet --mobilenet_model ./mobilenetssd/mobilenet_iter_73000.caffemodel --mobilenet_prototxt ./mobilenetssd/deploy.prototxt --n_tracked 1000 
```

``` bash
	python tracker.py --input video.avi
```

``` bash
	DARKNET_LIB=./lib/darknet.so python tracker.py --detector yolo --yolo_cfg ./yolo/files/yolov3-tiny.cfg --yolo_weights ./yolo/files/yolov3-tiny.weights --yolo_meta ./yolo/files/coco.data
```

## Use it in your own programs
* You can create a class by inheriting the Tracker class. output_function function can be used to interract with the tracked points. Some examples are included.
* example.py shows how to use the library in your own code
* ros_example.py shows how to integrate the library into ROS.
* zeromq_example.py shows how to use it with ZMQ Library.
