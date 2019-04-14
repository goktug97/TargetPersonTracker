
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
