import argparse

parser = argparse.ArgumentParser(description='Tracker')

# General
parser.add_argument('--input', default=0,
                    help='Input video, 0 for camera')

parser.add_argument('--window_name', type=str,
                    default='cvwindow',
                    help='OpenCV Window Name')

parser.add_argument('--camera_width', type=int, default=640,
                    help='Camera Width')

parser.add_argument('--camera_height', type=int, default=480,
                    help='Camera Height')

parser.add_argument('--camera_fps', type=int, default=0,
                    help='Camera FPS')

# Detector
parser.add_argument('--detector', type=str, default='yolo',
                    choices=('yolo', 'hog'),
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

parser.add_argument('--n_tracked', type=int, default=1600,
                    help='Number of tracked features')

parser.add_argument('--ftype', type=str, default='good',
                    choices=('orb', 'good'),
                    help='OpenCV ORB or GoodFeaturesToTrack')

parser.add_argument('--distance', type=int, default=50,
                    help='Filtering distance from distance')

parser.add_argument('--tracking_thresh', type=int, default=50,
                    help=(
                        'If tracked features are less than this value'
                        'retracking will be activated.'
                    ))

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
