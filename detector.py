class Detector(object):
    def __init__(self, args):
        if args.detector == 'yolo':
            from yolo import YOLODetector
            self.detector = YOLODetector(args)
        elif args.detector == 'hog':
            from hog import HOGDetector
            self.detector = HOGDetector()

    def detect(self, frame):
        dets = self.detector.detect(frame)
        return dets


if __name__ == '__main__':
    import cv2

    from options import args
    from utils import draw_detections

    cap = cv2.VideoCapture(0)

    detector = Detector(args)

    while True:
        ret, frame = cap.read()
        dets = detector.detect(frame)
        draw_detections(frame, dets)
        cv2.imshow('cvwindow', frame)
        if cv2.waitKey(1) == 27:
          break
