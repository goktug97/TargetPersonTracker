#!/usr/bin/env python

import numpy as np
import imutils
import cv2
import utils


class HOGDetector(object):
    """Detector class for people detection."""

    def __init__(self):
        """Initiliaze HOG Descriptor for people detection."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        # Resizing for speed and better accuracy
        self.original_shape = frame.shape
        frame = imutils.resize(frame,
                               width=min(400, frame.shape[1]))
        self.resized_shape = frame.shape

        # detection (xmin, ymin, width, height)
        rects, _ = self.hog.detectMultiScale(
            frame, winStride=(4, 4),
            padding=(8, 8), scale=1.05)

        # Non-Max Supression (xmin, ymin, xmax, ymax)
        self.detects = utils.nms(rects=rects, overlapThresh=0.65)

        self.scale_detections()

        return self.detects

    def scale_detections(self):
        """Scale detections for original frame."""
        original_w, original_h, _ = self.original_shape
        resized_w, resized_h, _ = self.resized_shape
        scaled_detects = []
        for detect in self.detects:
            xmin, ymin, xmax, ymax = detect
            w, h = xmax - xmin, ymax-ymin
            w = int(round((w/resized_w)*original_w))
            h = int(round((h/resized_h)*original_h))
            xmin = int(round((xmin/resized_w)*original_w))
            ymin = int(round((ymin/resized_h)*original_h))
            scaled_detects.append([xmin, ymin, xmin+w, ymin+h])

        self.detects = scaled_detects

    def draw_detections(self, frame):
        """Draw scaled detections on given frame."""
        for detect in self.detects:
            xmin, ymin, xmax, ymax = detect
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)



if __name__ == '__main__':
    import argparse

    hog = HOGDetector()

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help='Use given input instead of the camera', default='')
    ap.add_argument('-v', '--video', dest='video', action='store_true',
                    help='To use video input')
    args = vars(ap.parse_args())

    if len(args['input']) > 0 and not args['video']:
        frame = cv2.imread(args['input'])
        detects = hog.detect(frame)
        hog.draw_detections(frame)
        cv2.imshow('cvwindow', frame)
        cv2.waitKey(0)
    else:
        if args['video']:
            cap = cv2.VideoCapture(args['input'])
        else:
            cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            detects = hog.detect(frame)
            hog.draw_detections(frame)
            cv2.imshow('cvwindow', frame)
            if cv2.waitKey(1) == 27:
                break
            ret, frame = cap.read()
