#!/usr/bin/env python

import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1),
                cv2.FONT_HERSHEY_PLAIN, 1.0,
                (0, 0, 0), thickness = 2,
                lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 1.0,
                (255, 255, 255),
                lineType=cv2.LINE_AA)

def nms(rects, overlapThresh=0.65):
    """Do non-max supression on detections."""
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    picks = non_max_suppression(rects,
                                probs=None,
                                overlapThresh=overlapThresh)
    return picks

def draw_detections(frame, dets):
    for det in dets:
        xmin, ymin, xmax, ymax = det
        cv2.rectangle(frame, (xmin, ymin),
                      (xmax, ymax), (255,255,255), 3)

