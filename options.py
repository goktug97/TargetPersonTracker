import argparse

parser = argparse.ArgumentParser(description='Tracker')

#Detector
parser.add_argument('--detector', type=str, default='yolo',
                    choices=('yolo', 'hog'),
                    help='Detector name')

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

args = parser.parse_args()
