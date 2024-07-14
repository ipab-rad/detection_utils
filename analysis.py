from pytorch3d.ops import box3d_overlap
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_corners(center, lwh, yaw):
    """ Compute the 8 corners of the oriented bounding box. """
    l, w, h = lwh
    corners = torch.tensor([
        [-l / 2, -w / 2, -h / 2],
        [ l / 2, -w / 2, -h / 2],
        [ l / 2,  w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [ l / 2, -w / 2,  h / 2],
        [ l / 2,  w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2]
    ], dtype=torch.float32)

    rotation_matrix = torch.tensor([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ], dtype=torch.float32)

    rotated_corners = torch.matmul(rotation_matrix, corners.T).T
    translated_corners = rotated_corners + torch.tensor(center)
    return translated_corners


def bounding_box_examples():
    ground_truth_bounding_boxes = [
        {'center': [1,1,0], 'lwh': [1,1,3], 'yaw': 0},
        {'center': [2,1,4], 'lwh': [2,1,1], 'yaw': np.pi/4},
        {'center': [3,1,2], 'lwh': [3,3,1], 'yaw': np.pi/2},
        {'center': [0,1,0], 'lwh': [2,2,2], 'yaw': np.pi},
    ]

    # Define the bounding boxes
    prediction_bounding_boxes = [
        {'center': [1,1,0], 'lwh': [1,1,3.2], 'yaw': 0},  # matches 0 GT bbox
        {'center': [0.05, 0.98, 0], 'lwh': [2,2.5,2.3], 'yaw': np.pi},  # matches 3 GT bbox
        {'center': [2,1,4], 'lwh': [2,1,1], 'yaw': np.pi/2 + 0.05},  # matches 1 GT bbox
    ]

    ground_truth_count = len(ground_truth_bounding_boxes)
    prediction_count = len(prediction_bounding_boxes)

    ground_truth_corners = torch.empty((ground_truth_count, 8, 3), dtype=torch.float32)

    for i in range(ground_truth_count):
        gt = ground_truth_bounding_boxes[i]
        ground_truth_corners[i] = compute_corners(gt['center'], gt['lwh'], gt['yaw'])
  

    prediction_corners = torch.empty((prediction_count, 8, 3), dtype=torch.float32)

    for j in range(prediction_count):
        p = prediction_bounding_boxes[j]
        prediction_corners[j] = compute_corners(p['center'], p['lwh'], p['yaw'])

    return ground_truth_corners, prediction_corners


def compute_iou(ground_truth_corners, prediction_corners):
    iou = box3d_overlap(ground_truth_corners, prediction_corners)[1]
    return iou


def match_bounding_boxes(ious, iou_threshold):
    row_indices, col_indices = linear_sum_assignment(-iou)  # maximizing IoU -> minimize -IoU

    filtered_row_indices = []
    filtered_col_indices = []

    for i in range(len(row_indices)):
        if ious[row_indices[i], col_indices[i]] > iou_threshold:
            filtered_row_indices.append(row_indices[i])
            filtered_col_indices.append(col_indices[i])

    return filtered_row_indices, filtered_col_indices


def compute_positives_and_negatives(matched_count, prediction_count, ground_truth_count):
    true_positives = matched_count  # all the predictions which were successfully matched
    false_positives = prediction_count - true_positives  # all the predictions that were not successfully matched
    false_negatives = ground_truth_count - true_positives  # all the ground truths without a matching prediction

    return true_positives, false_positives, false_negatives


if __name__ == "__main__":
    ground_truth_corners, prediction_corners = bounding_box_examples()
    iou = compute_iou(ground_truth_corners, prediction_corners)

    print(f"IOU: {iou}")

    iou_threshold = 0.5

    matched_prediction_indices, matched_ground_truth_indices = match_bounding_boxes(iou, iou_threshold)

    print(f"Matched predictions: {matched_prediction_indices}")
    print(f"Matched ground truths: {matched_ground_truth_indices}")

    true_positives, false_positives, false_negatives = compute_positives_and_negatives(
        len(matched_prediction_indices), len(prediction_corners), len(ground_truth_corners)
    )

    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")