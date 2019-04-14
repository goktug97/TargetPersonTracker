import cv2

from options import args
from detector import Detector

cap = cv2.VideoCapture(0)

detector = Detector(args)

while True:
  ret, frame = cap.read()
  dets = detector.detect(frame)
  for det in dets:
    xmin, ymin, xmax, ymax = det
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,255), 3)

  cv2.imshow('cvwindow', frame)
  if cv2.waitKey(1) == 27:
    break

