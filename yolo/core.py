from .c_funcs import *
from .helpers import *
import cv2

class YOLODetector(object):
  def __init__(self, args):
    import os

    # Detection options
    self.thresh = args.thresh
    self.hier_thresh = args.hier_thresh
    self.nms = args.nms_overlap

    self.config_path = args.yolo_cfg
    self.weight_path = args.yolo_weights
    self.meta_path = args.yolo_meta

    # Check files
    if not os.path.exists(self.config_path):
      raise ValueError("Invalid config path `{}`".format(
        os.path.abspath(self.config_path)))
    if not os.path.exists(self.weight_path):
      raise ValueError("Invalid weight path `{}`".format(
        os.path.abspath(self.weight_path)))
    if not os.path.exists(self.meta_path):
      raise ValueError("Invalid data file path `{}`".format(
        os.path.abspath(self.meta_path)))

    # load the network
    self.net_main = load_net_custom(self.config_path.encode("ascii"),
                                    self.weight_path.encode("ascii"), 0, 1)
    self.meta_main = load_meta(self.meta_path.encode("ascii"))

    # Configured height and width for the input
    self.height = lib.network_height(self.net_main)
    self.width = lib.network_width(self.net_main)

    # Parse names 
    with open(self.meta_path) as metaFH:
      meta_contents = metaFH.read()
      import re
      match = re.search("names *= *(.*)$",
                        meta_contents,
                        re.IGNORECASE | re.MULTILINE)
      if match:
        result = match.group(1)
      else:
        result = None
      if os.path.exists(result):
        with open(result) as namesFH:
          names_list = namesFH.read().strip().split("\n")
          self.alt_names = [x.strip() for x in names_list]

  def detect(self, image):
    # prepeare image
    original_h, original_w, _ = image.shape
    image = cv2.resize(image,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR)[:,:,::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im, arr = array_to_image(image)

    # pointer for c function
    num = c_int(0)
    pnum = pointer(num)

    # Detection
    predict_image(self.net_main, im)
    dets = get_network_boxes(
      self.net_main, im.w, im.h,
      self.thresh, self.hier_thresh,
      None, 0, pnum, 0)
    num = pnum[0]

    # non-max supression
    if self.nms:
      do_nms_sort(dets, num, self.meta_main.classes, self.nms)

    # pick only 'person'
    res = []
    for j in range(num):
      for i in range(self.meta_main.classes):
        if dets[j].prob[i] > 0:
          b = dets[j].bbox
          if self.alt_names[i] == 'person':

            # coordinates as percentage
            xmin = (b.x-b.w/2)/self.width
            ymin = (b.y-b.h/2)/self.height
            xmax = (b.x+b.w/2)/self.width
            ymax = (b.y+b.h/2)/self.height

            # scale detections to input image
            xmin = int(round(xmin*original_w))
            ymin = int(round(ymin*original_h))
            xmax = int(round(xmax*original_w))
            ymax = int(round(ymax*original_h))

            res.append([xmin, ymin, xmax, ymax])

    free_detections(dets, num)
    return res

