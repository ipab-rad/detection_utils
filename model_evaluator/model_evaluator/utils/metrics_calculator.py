from model_evaluator.interfaces.detection2D import BBox2D

from model_evaluator.interfaces.detection3D import BBox3D
import numpy as np

from pytorch3d.ops import box3d_overlap


def calculate_ious_2d(
    pred_bboxes: list[BBox2D], gt_bboxes: list[BBox2D]
) -> np.ndarray:
    ious = np.empty((len(pred_bboxes), len(gt_bboxes)))

    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            ious[i, j] = pred_bbox.iou(gt_bbox)

    return ious


def calculate_ious_3d(
        pred_bboxes: list[BBox3D], gt_bboxes: list[BBox3D]
) -> np.ndarray:
    prediction_corners = [bbox.corners for bbox in pred_bboxes]
    ground_truth_corners = [bbox.corners for bbox in gt_bboxes]
    return box3d_overlap(ground_truth_corners, prediction_corners)[1]


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


def get_unmatched_tp_fp(
    ious: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    num_detections = ious.shape[0]
    num_gts = ious.shape[1]

    tp = np.zeros(num_detections)
    fp = np.zeros(num_detections)

    for i in range(num_detections):
        max_iou = 0

        for j in range(num_gts):
            iou = ious[i, j]

            if iou > max_iou:
                max_iou = iou

        if max_iou >= threshold:
            tp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def calculate_ap(tp, fp, num_samples):
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    if num_samples > 0:
        recalls = tp_cumsum / num_samples
    else:
        recalls = np.zeros_like(tp)

    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]

    return np.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )
