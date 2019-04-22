# Target Person Tracker with ORB, Optical Flow and a Detector

## Introduction
My initial goal with this project is a tracker with reidentification
cabilities to track the target person. I've developed this tracker to
put into a human tracking robot.

## System
* First the tracker is initialized with target features. If the target
features are given, the system starts tracking else it starts
collecting features from the center person. In my case the target
should turn 360 degrees so it can be retracked from all angles. Only
features from the upper portion of the detection are collected.
* After initialization the target features are found in the current
frame. Then optical flow is initialized to track found features.
* Every nth frame the tracked points are populated by finding
target features in the current frame to keep tracking continuous.
* If tracked points are below a certain threshold, retracking is actived.
* Every nth frame new features are added to the target features.
* Every nth frame target features and environment features are
compared, if there is a match, matches are removed from the target
features to reduce false positives.
* If there is no tracked points all detections are considered to find
target person so removing false positives is necessary step to eliminate
wrong tracking.

## Requirements
* OpenCV
* scipy
* sklearn
* numpy

### To use with MobileNet SSD Detector
* [MobileNet SSD Repository](https://github.com/chuanqi305/MobileNet-SSD)
* You don't need to compile the MobileNet SSD, OpenCV DNN is used.
* I've used [mobilenet_iter_73000.caffemodel](https://drive.google.com/file/d/0B3gersZ2cHIxVFI1Rjd5aDgwOG8/view) from the [original repository](https://github.com/chuanqi305/MobileNet-SSD)
* Put `mobilenet_iter_73000.caffemodel` into mobilenetssd folder.
* You can also use `--mobilenet_model` argument to pass the model path.

### To use with YOLOv3 Detector
* Compile [darknet](https://github.com/AlexeyAB/darknet) with shared library.
* Put the library into the lib folder.
* You can also use `DARKNET_LIB` environment variable to pass the path.
* [Weights, Config File and Data](https://pjreddie.com/darknet/yolo/) files should be downloaded.
* The detector only uses detections with the person tag

### To use with HOG Detector (not recommended, poor detection)
* imutils

## Usage
* Usage with MobileNet SSD Detector
``` bash
python tracker.py --detector mobilenet --mobilenet_model ./mobilenetssd/mobilenet_iter_73000.caffemodel --mobilenet_prototxt ./mobilenetssd/deploy.prototxt --n_tracked 1000 
```

* Usage with video input
``` bash
python tracker.py --input video.avi
```

* Usage with YOLO Detector
``` bash
DARKNET_LIB=./lib/darknet.so python tracker.py --detector yolo --yolo_cfg ./yolo/files/yolov3-tiny.cfg --yolo_weights ./yolo/files/yolov3-tiny.weights --yolo_meta ./yolo/files/coco.data
```

* To see other options you can use

``` bash
python tracker.py --help
```

You can use `ESC` (if you didn't use with `--no_gui`)
to start tracking earlier without reaching `n_tracked` number of features.

## Use it in your own programs

``` python
from options import args
from tracker import Tracker

class MyTracker(Tracker):
	def __init__(self, args):
		# Initiliaze Tracker class
		Tracker.__init__(self, args)

	    # For this example I am using OpenCV VideoCapture
		self.cap = cv2.VideoCapture(self.args.input)
		# Camera Settings
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
		if (self.args.camera_fps):
			self.cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)

	# You need to write get_frame function which returns (bool, frame)
	def get_frame(self):
		ret, self.frame = self.cap.read()
		return ret, self.frame
		
	# You can use this function to interract with the tracked points.
	# For example publish points with a ROS Node
	def output_function(self):
		print('Tracked Points: %d' % len(self.track_points))
		# Return False if you want to close the system
		return True
		
	# This function executed at the end of each collect_features function
	# Useful when there is no OpenCV window
	def collection_output(self):
		print('Collected Features: %d' % len(self.tkps))
		
	# This function executed while closing.
	# Useful to set certain events in case of threading.
	# Check zeromq_example.py for an example
	def finish(self):
		pass

tracker = MyTracker(args)
tracker.initiliaze_target()
tracker.run()
```

* `example.py` Minimal example
* `ros_example.py` shows how to integrate the library into ROS.
* `zeromq_example.py` shows how to use it with ZMQ Library.

* All examples can be executed, if you want to test the ROS example
  you can use `ros_camera.py` to publish images from the camera.
