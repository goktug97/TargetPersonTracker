import tracker
import cv2
import utils
import numpy as np

class Example(tracker.Tracker):
    """Example usage of Tracker class."""

    def __init__(self, args):
        """Initiliaze tracker and image source."""
        tracker.Tracker.__init__(self, args)

        # This is the default
        self._cap = cv2.VideoCapture(self.args.input)
        # Camera Settings
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
        if (self.args.camera_fps):
            self._cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)

    def get_frame(self):
        """Overload get_frame function."""
        ret, self.frame = self._cap.read()
        return ret, self.frame

if __name__ == '__main__':
    from options import args
    tracker = Example(args)
    tracker.initiliaze_target()
    tracker.run()
