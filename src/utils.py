import cv2
import numpy as np

def load_calibration(path='data/camera_calibration.npz'):
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs']

def undistort_frame(frame, K, D):
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    return cv2.undistort(frame, K, D, None, new_K)
