import tracker
import cv2
import utils
import numpy as np
import rospy
from sensor_msgs.msg import Image


class Example(tracker.Tracker):
    """Example usage of Tracker class."""

    def __init__(self, args):
        """Initiliaze tracker and image source."""
        tracker.Tracker.__init__(self, args)
        rospy.init_node('reciver', anonymous=True)
        rospy.Subscriber('/camera/image_raw', Image, self.callback)

        self.msg = None

    def callback(self, msg):
        """Call when a ROS Message received."""
        self.msg = msg

    def get_frame(self):
        """Overload get_frame function."""
        if self.msg is None:
            return True, None
        frame = np.frombuffer(self.msg.data, dtype=np.uint8)
        self.frame = frame.reshape(self.msg.height, self.msg.width, 3)
        return True, self.frame

    def initiliaze_target(self):
        """Initiliaze tracking points."""
        collection_not_finished = True
        while not rospy.is_shutdown() and collection_not_finished:
            collection_not_finished = self.collect_features()

    def run(self):
        """Start tracking chosen target."""
        # Find tracked points in current frame to start optical flow
        if not len(self.tkps):
            print('No features to track')
            return False
        while not rospy.is_shutdown() and not len(self.track_points):
            self.init_track_points()

        self.new_points_len = 0
        self.prev_frame = self.frame.copy()
        self.frame_idx = 0
        self.running = True
        while not rospy.is_shutdown() and self.running:
            self.track()

    def output_function(self):
        """Overload output function."""
        vis = self.frame.copy()

        utils.draw_str(vis, (20, 20),
                       'track count: %d' % len(self.track_points))

        utils.draw_str(vis, (20, 40),
                       'target features: %d' % len(self.tkps))

        """
        # Print length of features around tracked points
        print('Features around the tracked points: {}'.format(len(self.kps)))
        # Descriptors are self.des
        """

        """
        # Print length of tracked features
        print('Total number of learned features: {}'.format(len(self.tkps)))
        # Descriptors are self.tdes
        """

        # Draw tracked points
        for pts in self.track_points:
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
    tracker.initiliaze_target()
    tracker.run()
