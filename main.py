import cv2

from src.tracker.visualize import plot_tracking
from src.tracker.byte_tracker import BYTETracker
from src.detector.YOLO_detector import *
from src.utils.draw_polygon import *
from src.utils.lane_line_detector import *


class Predictor(object):
    def __init__(self) -> None:
        self.detector = Detector()
        self.tracker = BYTETracker(frame_rate=22)

    def predict(self, img, points):
        self.img_info = {"id": 0}
        self.img_info["raw_img"] = img
        self.img_info["height"], self.img_info["width"] = img.shape[:2]
        self.filter_class = [0, 1, 2, 3, 4]
        self.testsize = (640, 640)
        self.img_detect = get_area_detect(img, points)

        outputs_detect = self.detector.detect(self.img_detect)
        for out in outputs_detect:
            out_list = out.cpu().detach().numpy().tolist()
            box = list(map(int, out_list[:4]))
            img_object = np.zeros(
                [box[3] - box[1], box[2] - box[0], 3], dtype=np.uint8, order='C')
            self.img_detect[box[1]:box[3], box[0]:box[2]] = img_object
        mask = find_lane_line(self.img_detect)
        center = [[int((points[1][0] + points[0][0]) / 2), int((points[1][1] + points[0][1]) / 2)],
                  [int((points[2][0] + points[3][0]) / 2), int((points[2][1] + points[3][1]) / 2)]]
        # cv2.imshow('image', mask)
        # cv2.waitKey(0)
        contours = find_contour(mask)
        for c in contours:
            if len(c) > 20 and len(c) < 100:
                dist = calculate_distance(c[0][0], np.array(center))
                # print(dist)
                if dist < 40:
                    cv2.drawContours(self.img_info["raw_img"], c, -1, (0, 0, 255), 3)
        if outputs_detect is not None:
            outputs_tracking = self.tracker.update(outputs_detect, [
                                                   self.img_info["height"], self.img_info["width"]], self.testsize, self.filter_class)
            tlwhs = []
            ids = []
            scores = []
            for out in outputs_tracking:
                out_tlwh = out.tlwh
                out_id = out.track_id
                out_score = out.score
                vertical = out_tlwh[2] / out_tlwh[3] > 1.6
                if out_tlwh[2] * out_tlwh[3] > 10 and not vertical:
                    tlwhs.append(out_tlwh)
                    ids.append(out_id)
                    scores.append(out_score)

            out_im = plot_tracking(self.img_info["raw_img"], tlwhs, ids)
        else:
            out_im = self.img_info["raw_img"]

        return out_im


if __name__ == "__main__":
    predict = Predictor()

    cap = cv2.VideoCapture(
        '/home/tuyen/Desktop/Data/Tri_Nam/vds/STOPPED/3.avi')
    cv2.namedWindow('frame')
    while True:
        _, frame = cap.read()
        if frame is not None:
            frame = cv2.resize(frame, (1280, 720))
            original_frame = frame.copy()
            cv2.setMouseCallback('frame', on_mouse)
            key = cv2.waitKey(25)
            if key == 27:
                break
            if len(points) < 4:
                cv2.imshow('frame', frame)
            else:
                points = np.asarray(points)
                img = predict.predict(frame, points)
                cv2.polylines(img, [points], isClosed=True,
                              color=(0, 255, 0), thickness=2)
                cv2.imshow('frame', img)
        else:
            break
