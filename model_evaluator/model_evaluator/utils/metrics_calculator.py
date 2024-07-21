from model_evaluator.interfaces.detection2D import Detection2D, BBox2D
from model_evaluator.interfaces.labels import Label

from model_evaluator.interfaces.detection3D import BBox3D
import numpy as np

import torch
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
    prediction_corners = torch.stack([bbox.corners for bbox in pred_bboxes], dim=0)
    ground_truth_corners = torch.stack([bbox.corners for bbox in gt_bboxes], dim=0)
    return box3d_overlap(ground_truth_corners, prediction_corners)[1]


def get_tp_fp_from_ious(
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


def get_unmatched_tp_fp_from_ious(
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

def calculate_ious_from_dets_gts(
        detections: list[Detection2D], gts: list[Detection2D]
) -> np.ndarray:
    detections = detections.copy()
    detections.sort(key=lambda x: x.score, reverse=True)

    ious = calculate_ious_2d(
        [detection.bbox for detection in detections],
        [gt.bbox for gt in gts],
    )

    return ious


def calculate_ious_per_label_from_dets_gts(
        detections: list[Detection2D], gts: list[Detection2D], labels: set[Label]
) -> dict[Label, np.ndarray]:
    ious_dict = {}

    for label in labels:
        label_gts = [gt for gt in gts if gt.label in label]
        label_detections = [
            detection for detection in detections if detection.label in label
        ]

        ious = calculate_ious_from_dets_gts(label_detections, label_gts)

        ious_dict[label] = ious

    return ious_dict

def calculate_tps_fps_per_label(
        ious_per_label: dict[Label, np.ndarray], threshold: float
) -> dict[Label, tuple[np.ndarray, np.ndarray]]:
    tps_fps_per_label = {}

    for label in ious_per_label:
        tps_fps_per_label[label] = get_unmatched_tp_fp_from_ious(
            ious_per_label[label], threshold
        )

    return tps_fps_per_label


def calculate_fppi(fps: np.ndarray) -> int:
    return sum(fps)


def calculate_fppi_per_label(
        tps_fps_per_label: dict[Label, tuple[np.ndarray, np.ndarray]]
) -> int:
    fps = 0

    for label in tps_fps_per_label:
        _, label_fps = tps_fps_per_label[label]

        fps += label_fps.sum()

    return fps

def calculate_precisions_recalls(tps:np.ndarray, fps:np.ndarray, num_gts:int) -> tuple[np.ndarray, np.ndarray]:
    if num_gts == 0:
        return np.nan

    tps_cumsum = np.cumsum(tps)
    fps_cumsum = np.cumsum(fps)

    precisions = tps_cumsum / (tps_cumsum + fps_cumsum)

    recalls = tps_cumsum / num_gts

    return precisions, recalls

def interpolate_precisions(precisions_raw):
    precisions = precisions_raw.copy()
    for i in range(len(precisions) - 2, -1, -1):
        # set the current precision to the next one along, if it's higher
        precisions[i] = max(precisions[i], precisions[i+1])

    return precisions

def calculate_ap(tps: np.ndarray, fps: np.ndarray, num_gts: int) -> float:
    precisions, recalls = calculate_precisions_recalls(tps,fps,num_gts)

    precisions = interpolate_precisions(precisions)

    precisions = np.concatenate((precisions[0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    # identify recall threshold indices
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    return np.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )


def calculate_mean_ap(
        tps_fps_per_label: dict[Label, tuple[np.ndarray, np.ndarray]],
        num_gts: int,
) -> float:
    aps = []

    for label in tps_fps_per_label:
        tps, fps = tps_fps_per_label[label]

        aps.append(calculate_ap(tps, fps, num_gts))

    return np.mean(aps)


def calculate_mr(tps: np.ndarray, num_gts: int) -> int:
    return num_gts - tps.sum()


def calculate_mr_per_label(
        tps_fps_per_label: dict[Label, tuple[np.ndarray, np.ndarray]],
        num_gts: int,
) -> int:
    tps = 0

    for label in tps_fps_per_label:
        label_tps, _ = tps_fps_per_label[label]

        tps += label_tps.sum()

    return num_gts - tps


def compare_expectations(
        detections: list[Detection2D], expectations: dict[Label, int]
):
    for label in expectations:
        label_detections = [
            detection for detection in detections if detection.label in label
        ]

        if len(label_detections) != expectations[label]:
            print(
                f'Expected {expectations[label]} detections of {label}, got {len(label_detections)}'
            )
            return False

    return True