import cv2
import numpy as np


points = []

def on_mouse(event, x, y, flags, user_data):
    global points

    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append([x, y])


def get_area_detect(img, points):
    points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)

    return dts