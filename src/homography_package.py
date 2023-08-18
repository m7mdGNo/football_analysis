import cv2
import numpy as np


def calculate_homography(pts1, pts2, thresh):
    m, _ = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv2.RANSAC, thresh)
    return m


def warp_perspective_with_addweighted(src, dist, H, shape):
    out = cv2.warpPerspective(src, H, (shape[1], shape[0]))
    out = cv2.addWeighted(dist, 0.8, out, 1, 1)
    return out


def get_transformed_point(point, H):
    point = np.array([point], dtype=np.float32)
    trans_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), H)
    transformed_x = trans_point[0][0][0]
    transformed_y = trans_point[0][0][1]
    return (transformed_x, transformed_y)
