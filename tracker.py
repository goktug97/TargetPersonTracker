#!/usr/bin/env python
"""Tracking target person with ORB features and Lucas Kanade Optical Flow."""

import cv2  # noqa: I201
from detector import Detector  # noqa: I201
import numpy as np  # noqa: I201
from scipy.spatial import cKDTree  # noqa: I201
from sklearn.cluster import MeanShift  # noqa: I201
import utils  # noqa: I201


def extract_features(frame, n_features, ftype, mask):
    """Extract maximum n_features ORB features from given frame."""
    orb = cv2.ORB_create(n_features)
    if ftype == 'orb':
        kps = orb.detect(frame, mask)
    elif ftype == 'good':
        pts = cv2.goodFeaturesToTrack(
            np.mean(frame, axis=2).astype(np.uint8),
            3000, qualityLevel=0.05, minDistance=7,
            mask=mask)
        if pts is not None:
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        else:
            return [], []
    else:
        raise ValueError('Not Implemented')

    kps, des = orb.compute(frame, kps)
    kps = [[int(kp.pt[0]), int(kp.pt[1])] for kp in kps]
    return kps, des


def create_upper_mask(det, frame_shape):
    """Create mask for the upper half of the given detection."""
    xmin, ymin, xmax, ymax = det
    center_y = (ymax - ymin)//2
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    mask[ymin:center_y, xmin:xmax] = 255
    return mask


def find_detection(dets, x, y):
    """Find detection which includes given points."""
    """
    Returns first satisfied detection which includes given
    points.
    """
    for det in dets:
        xmin, ymin, xmax, ymax = det
        if (xmin < x and xmax > x and
                ymin < y and ymax > y):
            return xmin, ymin, xmax, ymax
    return None


def match_features(des1, des2):
    """Match features with given descriptors with brute force."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    idx1, idx2 = [], []

    for match in matches:
        if len(match) < 2:
            continue
        else:
            m, n = match
            # Lowe's ratio test
            if m.distance < 0.75*n.distance:
                if m.distance < 32:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)

    return idx1, idx2


def filter_features(kps, des, det, distance):
    """Pick center features only contained by the given detection."""
    point_tree = cKDTree(kps)

    xmin, ymin, xmax, ymax = det

    height = ymax-ymin
    n_sps = int(height / (distance*2))  # Number of sample points
    x = (xmin+xmax)//2

    # choose points around the center vertical line
    sample_features = []
    sample_descriptors = []

    for sp in range(n_sps):
        y = int(ymin + (sp * (distance*2)) + distance)
        idxs = point_tree.query_ball_point([x, y], distance)
        if len(idxs) > 0:
            sample_features.extend([point_tree.data[idx] for idx in idxs])
            sample_descriptors.extend([des[idx] for idx in idxs])

    sample_features = [[int(sample_feature[0]), int(sample_feature[1])]
                       for sample_feature in sample_features]
    return (np.array(sample_features),
            np.array(sample_descriptors, dtype=np.uint8))


def reduce_area_of_detection(det, width_multiplier, height_multiplier):
    """Reduce area of detection by width and height multiplier."""
    xmin, ymin, xmax, ymax = det
    width, height = xmax - xmin, ymax - ymin
    width = width*width_multiplier
    height = height*height_multiplier
    center_x = (xmin+xmax)//2
    center_y = (ymin+ymax)//2
    xmin, xmax = (int(center_x - width//2),
                  int(center_x + width//2))
    ymin, ymax = (int(center_y - height//2),
                  int(center_y + height//2))

    return xmin, ymin, xmax,  ymax


def find_clusters(tracks):
    """Find clusters in tracked points."""
    tracks = list(map(lambda x: [x[0][0], x[0][1]], tracks))
    ms = MeanShift(bandwidth=30, bin_seeding=True)
    ms.fit(tracks)
    return ms.cluster_centers_


class Tracker(object):
    """Track chosen person using orb features and optical flow."""

    def __init__(self, args):
        """Initiliaze the tracker."""
        self.detector = Detector(args)
        self.args = args

        cv2.namedWindow(args.window_name)

        # Chosen features for tracking
        self.tkps = []
        self.tdes = []

        # Features around the tracked points
        self.kps = []
        self.des = []

        # tracked points
        self.track_points = []

        # Lucas Optical Flow Params
        self.lk_params = {'winSize': (15, 15),
                          'maxLevel': 2,
                          'criteria': (
                              cv2.TERM_CRITERIA_EPS |
                              cv2.TERM_CRITERIA_COUNT,
                              10, 0.03)}

    def collect_features(self):
        """Collect features from chosen target for tracking."""
        ret, self.frame = self.get_frame()
        if self.frame is None:
            return True
        vis = self.frame.copy()
        # Detect people
        dets = self.detector.detect(self.frame)

        if not len(dets):
            return True

        utils.draw_detections(vis, dets)

        # Find center detection
        center_x = self.args.camera_width//2
        center_y = self.args.camera_height//2
        det = find_detection(
            dets, center_x, center_y)
        # Detection might not be centered
        if det is None:
            return True

        # Reduce area of the detection
        det = reduce_area_of_detection(
            det, self.args.width_multiplier,
            self.args.height_multiplier)

        # Create mask with detection
        mask = create_upper_mask(det, self.frame.shape[:2])
        kps, des = extract_features(self.frame,
                                    self.args.n_features,
                                    self.args.ftype,
                                    mask)
        if not len(kps):
            return True

        # Keep center features only
        kps, des = filter_features(kps, des, det, self.args.distance)

        # Draw features
        for i, kp in enumerate(kps):
            cv2.circle(vis, tuple(kps[i]), 2, (255, 165, 0), -1)

        # check matches to reduce duplicates
        # and to collect features evenly
        if len(self.tdes) and len(des):
            idx1, idx2 = match_features(
                np.array(self.tdes, dtype=np.uint8), des)
            des = np.delete(des, idx2, 0)
            kps = np.delete(kps, idx2, 0)

        # Populate the tracked features
        if len(des):
            self.tkps.extend(kps)
            self.tdes.extend(des)

        # break if threshold is satisfied
        utils.draw_str(vis, (20, 20), 'Features: %d' % len(self.tkps))
        if len(self.tkps) > self.args.n_tracked:
            return False

        if not self.collect_features_output_wrapper(vis) or not ret:
            return False

        return True

    def initiliaze_target(self, kps=None, des=None):
        """Initiliaze tracking points."""
        if kps is None or des is None:
            while self.collect_features():
                pass
        else:
            self.tkps, self.tdes = kps, des

    def collect_features_output_wrapper(self, vis):
        """Overload this function for output of the collect_features."""
        cv2.imshow(self.args.window_name, vis)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return False
        return True

    def output_function(self):
        """Overload this function to interact with the tracked points."""
        vis = self.frame.copy()

        utils.draw_str(vis, (20, 20),
                       'track count: %d' % len(self.track_points))

        utils.draw_str(vis, (20, 40),
                       'target features: %d' % len(self.tkps))

        # Draw tracked points
        for pts in self.track_points:
            cv2.polylines(vis, np.array([pts], dtype=np.int32),
                          False, utils.colors[min(len(pts), 9)])
        # Show frame
        cv2.imshow(self.args.window_name, vis)
        if cv2.waitKey(1) == 27:
            return False
        return True

    def get_frame(self):
        """Get frame from initialized source."""
        ret, frame = self.source.read()
        return (ret, frame)

    def init_track_points(self):
        """Find track points in current frame to start tracking."""
        ret, self.frame = self.get_frame()
        dets = self.detector.detect(self.frame)

        center_x = self.args.camera_width//2
        center_y = self.args.camera_height//2
        if len(dets) > 0:
            det = find_detection(
                dets, center_x, center_y)

            if det is None:
                return False
            det = reduce_area_of_detection(det,
                                           self.args.width_multiplier,
                                           self.args.height_multiplier)
            mask = create_upper_mask(det, self.frame.shape)
            kps, des = extract_features(self.frame, self.args.n_features,
                                        self.args.ftype, mask)

            if len(kps):
                idx1, idx2 = match_features(np.array(self.tdes), des)
                self.track_points = [
                    [(kps[idx][0], kps[idx][1])] for idx in idx2]

    def track(self):
        """Track points on current frame."""
        ret, self.frame = self.get_frame()
        if not ret:
            self.running = False
            return False

        # Optical Flow
        if len(self.track_points):
            self.optical_flow_tracking()

        # Add new features
        if (not self.frame_idx % self.args.add_every and
                len(self.track_points) > self.args.min_tracked):
            # Find center of the tracked points
            cluster_centers = find_clusters(self.track_points)

            # extract features
            kps, des = extract_features(
                self.frame, self.args.n_features,
                self.args.ftype, None)
            point_tree = cKDTree(kps)

            # Pick features around the center
            for cluster_center in cluster_centers:
                idxs = point_tree.query_ball_point(
                    cluster_center, 20)
                if len(idxs):
                    print('Added {} new features'.format(len(idxs)))
                    self.kps = []
                    self.des = []
                    for idx in idxs:
                        self.kps.append(kps[idx])
                        self.des.append(des[idx])
                        self.tkps.append(kps[idx])
                        self.tdes.append(des[idx])
                    self.new_points_len += len(idxs)

        # Remove false positive features
        # NOTE: Might remove wrong features in some situations
        if (len(self.track_points) > self.args.min_tracked and
                not self.frame_idx % self.args.remove_every):
            x, y = self.track_points[-1][-1]
            dets = self.detector.detect(self.frame)

            # Find bounding box of current target
            det = find_detection(dets, int(x), int(y))
            if det is not None:
                # Create inverse mask
                xmin, ymin, xmax, ymax = det
                mask = np.ones(self.frame.shape[:2],
                               dtype=np.uint8) * 255
                mask[ymin:ymax, xmin:xmax] = 0

                kps, des = extract_features(self.frame,
                                            self.args.n_features,
                                            self.args.ftype, mask)
                idx1, idx2 = match_features(np.array(self.tdes), des)

                # Remove matches
                if len(idx1):
                    print(
                        'Removed {} False Positives'.format(
                            len(idx1)))
                    for idx in sorted(idx1, reverse=True):
                        del self.tdes[idx]
                        del self.tkps[idx]

        # Remove duplicates
        if not self.frame_idx % self.args.remove_duplicates_every:
            if self.new_points_len:
                # Match old points with recently added points
                idx1, idx2 = match_features(
                    np.array(self.tdes[:-self.new_points_len]),
                    np.array(self.tdes[-self.new_points_len:]))

                # Remove matches
                if len(idx2):
                    print('Removed {} duplicates'.format(len(idx2)))
                    for idx in sorted(idx2, reverse=True):
                        del self.tdes[-self.new_points_len:][idx]
                        del self.tkps[-self.new_points_len:][idx]

                    self.new_points_len = 0

        # Retracking
        if (len(self.track_points) < self.args.tracking_thresh or
                (not self.frame_idx % self.args.retrack_every and
                 len(self.track_points) < 100)):
            dets = self.retrack()

        if not self.output_function():
            self.running = False
            return False

        self.prev_frame = self.frame.copy()
        self.frame_idx += 1

        return True

    def run(self):
        """Start tracking chosen target."""
        # Find tracked points in current frame to start optical flow
        if not len(self.tkps):
            print('No features to track')
            return False
        while not len(self.track_points):
            self.init_track_points()

        self.new_points_len = 0
        self.prev_frame = self.frame.copy()
        self.frame_idx = 0
        self.running = True
        while self.running:
            self.track()

    def optical_flow_tracking(self):
        """Lucas Kanade Optical Flow tracking."""
        p0 = np.float32(
            [tr[-1] for tr in self.track_points]).reshape(-1, 1, 2)
        p1, _, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, self.frame, p0, None, **self.lk_params)
        p0r, _, _ = cv2.calcOpticalFlowPyrLK(
            self.frame, self.prev_frame, p1, None, **self.lk_params)

        # Keep good features
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(
                self.track_points, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > self.args.track_len:
                del tr[0]
            new_tracks.append(tr)
        self.track_points = new_tracks

    def retrack(self):
        """Initiliaze retracking for recognized features."""
        # If tracking is not zero activate retracking
        # on current area, else it will check features
        # for all detections.
        dets = self.detector.detect(self.frame)
        if len(self.track_points):
            x, y = self.track_points[-1][-1]
            dets = [find_detection(dets, int(x), int(y))]

        if len(dets) and dets[0] is not None:
            for det in dets:
                det = reduce_area_of_detection(
                    det,
                    self.args.width_multiplier,
                    self.args.height_multiplier)

                mask = create_upper_mask(det, self.frame.shape)

                kps, des = extract_features(
                    self.frame, self.args.n_features,
                    self.args.ftype, mask=mask)
                if len(kps):
                    idx1, idx2 = match_features(
                        np.array(self.tdes), des)
                    # Check whether it is larger than specified threshold
                    # to reduce false positives.
                    if len(idx1) > self.args.min_match_threshold:
                        kps, des = filter_features(
                            [kps[idx] for idx in idx2],
                            [des[idx] for idx in idx2],
                            det,
                            self.args.distance)

                        for kp in kps:
                            x, y = kp
                            self.track_points.append([(x, y)])
        else:
            # if no detection is found remove noise
            if len(self.track_points) < 10:
                self.track_points = []
        return dets


if __name__ == '__main__':
    from options import args

    class Example(Tracker):
        """Example usage of Tracker class."""

        def __init__(self, args):
            """Initiliaze tracker and image source."""
            Tracker.__init__(self, args)

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
                           'track count: %d' % len(self.track_points))

            utils.draw_str(vis, (20, 40),
                           'target features: %d' % len(self.tkps))

            # Draw tracked points
            for pts in self.track_points:
                cv2.polylines(vis, np.array([pts], dtype=np.int32),
                              False, utils.colors[min(len(pts), 9)])
            # Show frame
            cv2.imshow(self.args.window_name, vis)
            if cv2.waitKey(1) == 27:
                return False
            return True

    tracker = Example(args)
    tracker.initiliaze_target()
    tracker.run()
