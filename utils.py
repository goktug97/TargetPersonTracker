#!/usr/bin/env python

import cv2 as cv
from imutils.object_detection import non_max_suppression
import numpy as np

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

def nms(rects, overlapThresh=0.65):
    """Do non-max supression on detections."""
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    picks = non_max_suppression(rects,
                                probs=None,
                                overlapThresh=overlapThresh)
    return picks

