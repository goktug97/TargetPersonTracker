import argparse

parser = argparse.ArgumentParser(description='Tracker')

# General
parser.add_argument('--input', default=0,
                    help='Input video, if not given use camera')

parser.add_argument('--window_name', type=str,
                    default='cvwindow',
                    help='OpenCV Window Name')

parser.add_argument('--no_gui', dest='no_gui', action='store_true',
                help='Don\'t Show OpenCV Window')

parser.add_argument('--camera_width', type=int, default=640,
                    help='Camera Width')

parser.add_argument('--camera_height', type=int, default=480,
                    help='Camera Height')

parser.add_argument('--camera_fps', type=int, default=0,
                    help='Camera FPS')

# Detector
parser.add_argument('--detector', type=str, default='mobilenet',
                    choices=('yolo', 'hog', 'mobilenet'),
                    help='Detector name')

# Tracker
parser.add_argument('--width_multiplier', type=float, default=0.6,
                    help=('Multiplier to reduce area of detected person'
                          'to reduce negative features'))

parser.add_argument('--height_multiplier', type=float, default=0.9,
                    help=('Multiplier to reduce area of detected person'
                          'to reduce negative features'))

parser.add_argument('--track_len', type=int, default=9,
                    help='Tracking lenght of a feature')

parser.add_argument('--n_tracked', type=int, default=6000,
                    help='Minumum number of features to start tracking')

parser.add_argument('--ftype', type=str, default='good',
                    choices=('orb', 'good'),
                    help='OpenCV ORB or GoodFeaturesToTrack')

parser.add_argument('--distance', type=int, default=50,
                    help='Filtering distance from the center')

parser.add_argument('--tracking_thresh', type=int, default=10,
                    help=(
                        'If tracked features are less than this value'
                        'retracking will be activated.'))

parser.add_argument('--min_match_threshold', type=int, default=10,
                    help=(
                        'Minimum required match for retracking to'
                        'to reduce false positives.'))

parser.add_argument('--min_tracked', type=int, default=20,
                    help=('Minimum required tracked point to do any'
                          'action like removing and adding'))

parser.add_argument('--max_tracked', type=int, default=100,
                    help=('Maximum tracked points'))

parser.add_argument('--remove_every', type=int, default=50,
                    help=('Remove false positive features once every'
                          'specified frame'))
parser.add_argument('--add_every', type=int, default=10,
                    help=('Remove false positive features once every'
                          'specified frame'))
parser.add_argument('--retrack_every', type=int, default=10,
                    help=('Remove false positive features once every'
                          'specified frame'))
parser.add_argument('--remove_duplicates_every', type=int, default=20,
                    help=('Remove duplicates after adding new features'))

# MobileNet SSD
parser.add_argument('--mobilenet_prototxt', type=str,
                    default='./mobilenetssd/deploy.prototxt',
                    help='MobileNet SSD prototxt')
parser.add_argument('--mobilenet_model', type=str,
                    default='./mobilenetssd/mobilenet_iter_73000.caffemodel',
                    help='MobileNet SSD Caffe Model')
parser.add_argument('--mobilenet_confidence', type=float, default=0.7,
                    help='Mobilnet threshold')

# YOLO
parser.add_argument('--yolo_cfg', type=str,
                    default='./yolo/files/yolov3-tiny.cfg',
                    help='darknet config file')
parser.add_argument('--yolo_weights', type=str,
                    default='./yolo/files/yolov3-tiny.weights',
                    help='darknet weights file')
parser.add_argument('--yolo_meta', type=str,
                    default='./yolo/files/coco.data',
                    help='darknet data file')

parser.add_argument('--thresh', type=float, default=0.5,
                    help='YOLO threshold')

parser.add_argument('--hier_thresh', type=float, default=0.5,
                    help='YOLO hier_threshold')

parser.add_argument('--nms_overlap', type=float, default=0.45,
                    help='YOLO Non-Max Supression Overlap Value')

# ORB
parser.add_argument('--n_features', type=int, default=2000,
                    help='ORB Features detector max detected features')

args = parser.parse_args()
