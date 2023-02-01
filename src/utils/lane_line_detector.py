import numpy as np
import cv2
from src.detector.YOLO_detector import *


def find_contour(image):
    contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contour


def find_lane_line(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 190], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask


def calculate_distance(point, line):
    return np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) / np.linalg.norm(line[1] - line[0])