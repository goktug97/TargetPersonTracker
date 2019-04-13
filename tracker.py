import cv2
import numpy as np
from scipy.spatial import cKDTree
from common import draw_str
import os
import glob
from sklearn.cluster import DBSCAN

x, y = None, None

def callback(event, cx, cy, flags, param):
    global x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x = cx
        y = cy

def extract_features(frame):
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8),
                                  3000, qualityLevel=0.01, minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(frame, kps)

    return kps, des

def calibration(path):
    global x, y
    tkps = []
    tdes = []
    cv2.namedWindow('cvwindow2')
    cv2.setMouseCallback('cvwindow2', callback)
    images = glob.glob(os.path.join(path, '*' + '.bmp'))
    for image in images:
        frame = cv2.imread(image)
        kps, des = extract_features(frame)
        cv2.drawKeypoints(frame, kps, frame, (0,0,0))
        cv2.imshow('cvwindow2', frame)
        cv2.waitKey(0)
        point_tree = cKDTree(np.array([(kp.pt[0], kp.pt[1]) for kp in kps]))
        index = point_tree.query_ball_point([x, y], 30)
        if len(point_tree.data[index]) > 0:
            tkps.extend(point_tree.data[index])
            tdes.extend(des[index])
    cv2.destroyAllWindows()
    return np.array(tkps), np.array(tdes)


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (
                      cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

track_len = 10

def test(tracks):
    test = []
    for track in tracks:
        x, y = track
        test.append([x,y])
    db = DBSCAN(eps=0.3, min_samples=10).fit(np.array(test))

# cap = cv2.VideoCapture('output.avi')
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
orb = cv2.ORB_create()
cv2.namedWindow('cvwindow')
ret, frame = cap.read()
track = []
prev_frame = frame.copy()
tkps, tdes = calibration('./images')
frame_idx = 0
detect_interval = 10
while ret:
    vis = frame.copy()
    if len(track) > 0:
        img0, img1 = prev_frame, frame
        p0 = np.float32([tr[-1] for tr in track]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
            img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(track, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        track = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in track], False, (0, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(track))
    if frame_idx % detect_interval == 0:
        kps, des = extract_features(frame)
        matches = bf.knnMatch(tdes, des, k=2)

        idx1, idx2 = [], []

        for m,n in matches:
          if m.distance < 0.75*n.distance:
            p1 = tkps[m.queryIdx]
            p2 = [kps[m.trainIdx]]

            # be within orb distance 32
            if m.distance < 30:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

        xtrack = [kps[idx] for idx in idx2] 
        for kp in xtrack:
            track.append([(kp.pt[0], kp.pt[1])]) 
        point_tree = cKDTree(
            np.array([(kp.pt[0], kp.pt[1]) for kp in kps]))
        if len(xtrack) > 0 :
            for group in point_tree.query_ball_point([(kp.pt[0], kp.pt[1]) for kp in xtrack], 30):
             cluster = point_tree.data[group]
             x_list, y_list = cluster[:, 0], cluster[:, 1]
             for x, y in zip(x_list, y_list):
                 track.append([(x,y)])

    prev_frame = frame.copy()
    ret, frame = cap.read()
    frame_idx += 1
    cv2.imshow('cvwindow', vis)
    if cv2.waitKey(30) == 27:
        break

        

