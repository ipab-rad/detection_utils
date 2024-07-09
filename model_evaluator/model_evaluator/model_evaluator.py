from model_evaluator.model_evaluator.detection import BBox, Detection
from model_evaluator.yolox_connector import TensorrtYOLOXConnector
import cv2
import numpy as np

def iou(bbox1: BBox, bbox2: BBox):
    # intersection rect
    left = max(bbox1.x1, bbox2.x1)
    top = max(bbox1.y1, bbox2.y1)
    right = min(bbox1.x2, bbox2.x2)
    bottom = min(bbox1.y2, bbox2.y2)

    if right < left or bottom < top:
        return 0.0
    
    intersect_area = (right - left) * (bottom - top)

    return intersect_area / (bbox1.area() + bbox2.area() - intersect_area)

def calculate_metrics(pred_bboxes: list[BBox], gt_bboxes: list[BBox]):
    res = np.zeros(len(pred_bboxes))

    for i, pred in enumerate(pred_bboxes):
        for gt in gt_bboxes:
            iou = iou(pred, gt)

            if iou > 0.5:
                res[i] = 1

                break

    acc_tp = sum(res)
    acc_fp = len(res) - sum(res)

    recall = acc_tp / len(gt_bboxes)
    precision = acc_tp / (acc_tp + acc_fp)


def main():
    connector = TensorrtYOLOXConnector('/sensor/camera/lspf_r/image_raw', '/perception/object_recognition/detection/rois0')

    img = cv2.imread('<IMAGE PATH>')

    detections = connector.runInference(img)

    for detection in detections:
        print(detection)