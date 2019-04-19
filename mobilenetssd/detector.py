import cv2

class MobileNetSSD(object):
    def __init__(self, args):
        # Mobilenet SSD with OpenCV DNN 
        self.net = cv2.dnn.readNetFromCaffe(args.mobilenet_prototxt,
                                            args.mobilenet_model)
        self.args = args
        
    def detect(self, frame):
        # Create blob from the image
        blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                    (300, 300),
                                    (127.5, 127.5, 127.5),
                                    False)
        self.net.setInput(blob)

        cols = frame.shape[1]
        rows = frame.shape[0]

        # Detection
        detections = self.net.forward()

        dets = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.args.mobilenet_confidence:
                idx = int(detections[0, 0, i, 1])
                # Check whether the object is person or not
                if idx == 15:
                    xmin = int(detections[0, 0, i, 3] * cols)
                    ymin = int(detections[0, 0, i, 4] * rows)
                    xmax   = int(detections[0, 0, i, 5] * cols)
                    ymax   = int(detections[0, 0, i, 6] * rows)
                    dets.append((xmin, ymin, xmax, ymax))
        return dets
