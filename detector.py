class Detector(object):
    def __init__(self, args):
        if args.detector == 'yolo':
            from yolo import YOLODetector
            self.detector = YOLODetector(args)
        elif args.detector == 'hog':
            from hog import HOGDetector
            self.detector = HOGDetector()
        elif args.detector == 'mobilenet':
            from mobilenetssd import MobileNetSSD
            self.detector = MobileNetSSD(args)

    def detect(self, frame):
        dets = self.detector.detect(frame)
        return dets


if __name__ == '__main__':
    import cv2

    from options import args
    from utils import draw_detections
    import time

    cap = cv2.VideoCapture(args.input)

    detector = Detector(args)

    ret, frame = cap.read()
    time.sleep(0.5)
    while ret:
        dets = detector.detect(frame)
        draw_detections(frame, dets)
        cv2.imshow('cvwindow', frame)
        if cv2.waitKey(1) == 27:
          break
        ret, frame = cap.read()
