#!/usr/bin/env python
"""Common functions for other modules."""

import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

colors = np.array([[0.0, 0.0, 127.0],
                   [0.0, 0.0, 204.0],
                   [0.0, 89.0, 255.0],
                   [0.0, 204.0, 255.0],
                   [77.0, 255.0, 153.0],
                   [153.0, 255.0, 77.0],
                   [255.0, 230.0, 0.0],
                   [255.0, 126.0, 0.0],
                   [255.0, 26.0, 0.0],
                   [123.0, 0.0, 0.0]])

def draw_str(dst, target, s):
    """Draw cool string on image."""
    """Taken from opencv samples"""
    x, y = target
    cv2.putText(dst, s, (x+1, y+1),
                cv2.FONT_HERSHEY_PLAIN, 1.0,
                (0, 0, 0), thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 1.0,
                (255, 255, 255),
                lineType=cv2.LINE_AA)


def nms(rects, overlapThresh=0.65):
    """Do non-max supression on detections."""
    picks = non_max_suppression(rects,
                                probs=None,
                                overlapThresh=overlapThresh)
    return picks



def draw_detections(frame, dets):
    """Draws rectangles to given frame."""
    """
    Draw rectangles directly on given input,
    doesn't create a copy.
    dets = xmin, ymin, xmax, ymax
    """

    if len(dets) > 0 and dets[0] is not None:
        for det in dets:
            xmin, ymin, xmax, ymax = det
            cv2.rectangle(frame, (xmin, ymin),
                          (xmax, ymax),
                          (255, 255, 255), 3)
