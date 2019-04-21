import tracker
import cv2
import utils
import numpy as np

class Example(tracker.Tracker):
    """Example usage of Tracker class."""

    def __init__(self, args):
        """Initiliaze tracker and image source."""
        tracker.Tracker.__init__(self, args)

        self.cap = cv2.VideoCapture(self.args.input)
        # Camera Settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
        if (self.args.camera_fps):
            self.cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)

    def get_frame(self):
        """Overload get_frame function."""
        ret, self.frame = self.cap.read()
        return ret, self.frame

    def output_function(self):
        """Overload output function."""
        vis = self.frame.copy()

        utils.draw_str(vis, (20, 20),
                       'track count: %d' % len(self.track))

        utils.draw_str(vis, (20, 40),
                       'target features: %d' % len(self.tkps))

        '''
        # Print length of features around tracked points
        print('Features around the tracked points: {}'.format(len(self.kps)))
        # Descriptors are self.des
        '''

        '''
        # Print length of tracked features
        print('Total number of learned features: {}'.format(len(self.tkps)))
        # Descriptors are self.tdes
        '''

        # Draw tracked points
        for pts in self.track:
            cv2.polylines(vis, np.array([pts], dtype=np.int32),
                          False, utils.colors[min(len(pts), 9)])

        # Show frame
        cv2.imshow(self.args.window_name, vis)
        if cv2.waitKey(1) == 27:
            return False
        return True

if __name__ == '__main__':
    from options import args
    tracker = Example(args)
    tracker.run()
