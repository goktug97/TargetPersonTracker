import tracker
import cv2
import utils
import numpy as np
import zmq
import ctypes
import time
import struct
import threading


def publisher(args, event):
    # Publisher socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    topic = 'image'
    socket.setsockopt(zmq.SNDHWM, 5)
    socket.bind('tcp://*:5555')

    cap = cv2.VideoCapture(args.input)

    # Camera Settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if (args.camera_fps):
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    index = 0

    while not event.is_set():
        index += 1
        ret, frame = cap.read()
        if not ret:
            break

        # timestamp in ms
        timestamp = int(round(time.time() * 1000))

        # Metadata to recover the array
        md = {
            'dtype': str(frame.dtype),
            'shape': frame.shape
        }

        # Send messages
        socket.send_string(topic, zmq.SNDMORE)
        socket.send(ctypes.c_longlong(timestamp), zmq.SNDMORE)
        socket.send(ctypes.c_int(index), zmq.SNDMORE)
        socket.send_json(md, zmq.SNDMORE)
        socket.send(frame, 0, copy=True, track=False)
        # print('Sent frame {}'.format(index))


class Example(tracker.Tracker):
    """Example usage of Tracker class."""

    def __init__(self, args, event):
        """Initiliaze tracker and image source."""
        tracker.Tracker.__init__(self, args)

        # Subscriber socket
        self.event = event
        self.event.clear()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'image')
        # socket.setsockopt(zmq.RCVHWM, 5)
        self.socket.connect('tcp://localhost:5555')

    def get_frame(self):
        """Overload get_frame function."""
        cur_time = int(time.time() * 1000)

        topic = self.socket.recv_string()
        timestamp = struct.unpack('<q', self.socket.recv())[0]
        index = struct.unpack('<i', self.socket.recv())[0]

        # Recv the frame and recontruct it with metadata
        md = self.socket.recv_json(flags=0)
        bframe = self.socket.recv(flags=0, copy=True, track=False)
        frame = np.frombuffer(bframe, dtype=md['dtype'])
        self.frame = frame.reshape(md['shape'])

        if self.args.input == 0:
            # Restart the connection if there is a latency
            if (cur_time - timestamp) > 10:
                self.socket.disconnect('tcp://localhost:5555')
                self.socket.connect('tcp://localhost:5555')
                print('Latency, The connection has been restarted')

        return True, self.frame

    def output_function(self):
        """Overload output function."""
        vis = self.frame.copy()

        utils.draw_str(vis, (20, 20),
                       'track count: %d' % len(self.track_points))

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
        for pts in self.track_points:
            cv2.polylines(vis, np.array([pts], dtype=np.int32),
                          False, utils.colors[min(len(pts), 9)])

        # Show frame
        cv2.imshow(self.args.window_name, vis)
        if cv2.waitKey(1) == 27:
            self.event.set()
            return False
        return True


if __name__ == '__main__':
    from options import args

    event = threading.Event()

    tracker = Example(args, event)

    tracker_thread = threading.Thread(target=tracker.run, args=())

    # Start camera
    camera_thread = threading.Thread(target=publisher, args=(args, event,))
    camera_thread.start()

    tracker.initiliaze_target()

    tracker_thread.start()

    camera_thread.join()
    tracker_thread.join()

