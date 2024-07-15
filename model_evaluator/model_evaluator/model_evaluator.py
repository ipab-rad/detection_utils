import re
import glob

import cv2
import numpy as np

from model_evaluator.rosbag_reader import RosbagDatasetReader2D
from model_evaluator.waymo_reader import WaymoDatasetReader2D
from model_evaluator.detection import BBox2D, Label2D
from model_evaluator.yolox_connector import TensorrtYOLOXConnector



def calculate_ious_2d(
    pred_bboxes: list[BBox2D], gt_bboxes: list[BBox2D]
) -> np.ndarray:
    ious = np.empty((len(pred_bboxes), len(gt_bboxes)))

    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            ious[i, j] = pred_bbox.iou(gt_bbox)

    return ious


def get_tp_fp(
    ious: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    num_detections = ious.shape[0]
    num_gts = ious.shape[1]

    tp = np.zeros(num_detections)
    fp = np.zeros(num_detections)

    matched_gts = []

    for i in range(num_detections):
        max_iou = 0
        max_iou_idx = -1

        for j in range(num_gts):
            iou = ious[i, j]

            if iou > max_iou:
                max_iou = iou
                max_iou_idx = j

        if max_iou >= threshold:
            if max_iou_idx not in matched_gts:
                tp[i] = 1
                matched_gts.append(max_iou_idx)
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def calculate_ap(tp, fp, num_samples):
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    if num_samples == 0:
        recalls = tp_cumsum
    else:
        recalls = tp_cumsum / num_samples

    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]

    return np.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )


def to_cv_pts(bbox: BBox2D):
    return (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2))


def draw_bboxes(image, gts, detections, included_labels=Label2D.ALL):
    for gt in gts:
        if gt.label not in included_labels:
            continue

        pt1, pt2 = to_cv_pts(gt.bbox)
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)

        cv2.putText(
            image,
            str(gt.label),
            (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    for detection in detections:
        if detection.label not in included_labels:
            continue

        pt1, pt2 = to_cv_pts(detection.bbox)
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(
            image,
            str(detection.label),
            (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


def main():
    pattern = re.compile(r'.*/(?P<date>\d{4}_\d{2}_\d{2})-(?P<time>\d{2}_\d{2}_\d{2})_(?P<name>.+).mcap')
    name_pattern = r'(?P<distance>\d+m)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)_(?P<bag_no>\d)'

    paths = glob.glob('/opt/ros_ws/rosbags/kings_buildings_data/**', recursive=True)

    for path in paths:

        match = pattern.match(path)

        if match:
            date = match.group('date')
            time = match.group('time')
            name = match.group('name')

            print(name)

    exit()

    rosbag_reader = RosbagDatasetReader2D('/opt/ros_ws/rosbags/kings_buildings_data/2024_07_12-10_58_56_5m_1_ped_0/2024_07_12-10_58_39_5m_1_ped_0/2024_07_12-10_58_39_5m_1_ped_0_0.mcap')

    data_ros = rosbag_reader.read_data()


    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_raw',
        '/perception/object_recognition/detection/rois0',
    )

    reader = WaymoDatasetReader2D('/opt/ros_ws/rosbags/waymo/validation')

    data = reader.read_data()

    for i, (image, gts) in enumerate(data_ros):
        detections = connector.run_inference(image)

        if detections is None:
            print('Inference failed')
            continue

        if i % int(len(data_ros) / 10) == 0:
            draw_bboxes(image, gts, detections, Label2D.VRU)
            #cv2.imwrite(f'image{i}.png', image)

        aps = {}

        for label in [Label2D.PEDESTRIAN, Label2D.BICYCLE]:
            label_gts = [gt for gt in gts if gt.label in label]
            label_detections = [
                detection
                for detection in detections
                if detection.label in label
            ]

            label_detections.sort(key=lambda x: x.score, reverse=True)

            ious = calculate_ious_2d(
                [detection.bbox for detection in label_detections],
                [gt.bbox for gt in label_gts],
            )

            tp, fp = get_tp_fp(ious, 0.5)

            aps[label] = calculate_ap(tp, fp, len(label_gts))

        mAP = np.mean(list(aps.values()))

        print(f'{mAP=}')
