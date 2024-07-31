from model_evaluator.connectors.lidar_connector import LiDARConnector
from model_evaluator.interfaces.detection3D import Detection3D
from model_evaluator.interfaces.labels import Label, ALL_LABELS, WAYMO_LABELS, labels_match
from model_evaluator.readers.waymo_reader import WaymoDatasetReader3D
from model_evaluator.utils.json_file_reader import write_json
from model_evaluator.utils.kb_rosbag_matcher import match_rosbags_in_path, KBRosbag
from model_evaluator.utils.metrics_calculator import calculate_ious_3d

from scipy.optimize import linear_sum_assignment
import glob
from pathlib import Path


def process_rosbags_3D(connector:LiDARConnector):
    # anything without an IoU threshold does not have any ground truths
    # 0.4 used for pedestrian due to GT annotation strategy
    iou_thresholds = {Label.PEDESTRIAN: 0.4, Label.CAR: 0.5, Label.TRUCK:0.5}

    rosbags = match_rosbags_in_path('/opt/ros_ws/rosbags/kings_buildings_data')

    for r in rosbags:
        process_single_rosbag_3D(connector, r, iou_thresholds)

def process_single_rosbag_3D(connector:LiDARConnector, rosbag:KBRosbag, iou_thresholds: dict[Label, float]):
    print(rosbag.metadata)

    rosbag_reader = rosbag.get_reader_3d()

    if not rosbag_reader.annotations_present:
        print("no annotations")
        return

    rosbag_data = rosbag_reader.read_data()

    start_frame = rosbag_reader.start_frame
    end_frame = rosbag_reader.end_frame

    all_results = {label.name: {"gt_count":0, "results":[]} for label in ALL_LABELS}

    for frame_counter, (point_cloud, gt_dets) in enumerate(rosbag_data):
        if frame_counter < start_frame or frame_counter > end_frame:
            continue

        detections = connector.run_inference(point_cloud)

        detections_in_experiment_area = filter_detections_kb(detections)

        results_per_class = process_frame_detections(
            detections_in_experiment_area,
            gt_dets, iou_thresholds, frame_counter,
            ALL_LABELS
        )

        for result_label in results_per_class:
            overall_dict_entry = all_results[result_label.name]
            result_dict_entry = results_per_class[result_label]

            overall_dict_entry["gt_count"] += result_dict_entry["gt_count"]
            overall_dict_entry["results"] += result_dict_entry["results"]

    results_dir = "/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/results/kb/no_ground"

    write_json(f"{results_dir}/{rosbag.bbox_file_name}.json", all_results)


def process_waymo_3D(connector:LiDARConnector):
    # anything without an IoU threshold does not have any ground truths
    iou_thresholds = {Label.PEDESTRIAN: 0.5, Label.VEHICLE: 0.7, Label.BICYCLE: 0.5, Label.UNKNOWN: 0.5}

    waymo_scenes = [s for s in [Path(p).stem for p in glob.glob("/opt/ros_ws/rosbags/waymo/validation/lidar/*")] if s != "_metadata"]

    for ws in waymo_scenes:
        process_single_waymo_scene_3D(connector, ws, iou_thresholds)


def process_single_waymo_scene_3D(connector:LiDARConnector, waymo_scene:str, iou_thresholds: dict[Label, float]):
    print(f"{waymo_scene}")

    waymo_reader = WaymoDatasetReader3D(
        "/opt/ros_ws/rosbags/waymo/validation",
        waymo_scene
    )

    waymo_data = waymo_reader.read_data()

    all_results = {label.name: {"gt_count": 0, "results": []} for label in WAYMO_LABELS}

    for frame_counter, (point_cloud, gt_dets) in enumerate(waymo_data):
        detections = connector.run_inference(point_cloud)

        results_per_class = process_frame_detections(
            detections, gt_dets, iou_thresholds, frame_counter, WAYMO_LABELS
        )

        for result_label in results_per_class:
            overall_dict_entry = all_results[result_label.name]
            result_dict_entry = results_per_class[result_label]

            overall_dict_entry["gt_count"] += result_dict_entry["gt_count"]
            overall_dict_entry["results"] += result_dict_entry["results"]

    results_dir = "/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/results/waymo/no_ground"

    write_json(f"{results_dir}/{waymo_scene}.json", all_results)


def process_frame_detections(predictions:list[Detection3D], gts: list[Detection3D], iou_thresholds: dict[Label, float], frame:int, label_list: list[Label])\
        -> dict[Label, dict]:
    # TODO implement below filtering for rosbag analysis
    # for pedestrians, log true and false positives
    # for everything else, only log false positives
    detection_results_per_class = {}

    for label in label_list:
        label_preds = [x for x in predictions if labels_match(x.label,label)]
        label_gts = [x for x in gts if labels_match(x.label,label)]

        no_preds = len(label_preds) == 0
        no_gts = len(label_gts) == 0

        if no_preds and no_gts:
            # frame does not interact at all with current label
            continue

        if no_preds:
            # no predictions for class but there are gts
            detection_results_per_class[label] = {
                "gt_count":len(label_gts),
                "results":[]
            }
            continue

        matched_prediction_idcs:list[int]

        if no_gts:
            # predictions but no gts
            # all the predictions are false positives
            # matched_prediction_idcs should be empty
            # this correctly results in all false positives in list comprehension below
            matched_prediction_idcs = []
        else:
            threshold = iou_thresholds[label]

            label_pred_bboxes = [x.bbox for x in label_preds]
            label_gt_bboxes = [x.bbox for x in label_gts]

            # this puts ground truths on the columns and predictions on the rows
            ious = calculate_ious_3d(label_pred_bboxes, label_gt_bboxes)

            _, matched_prediction_idcs = match_bounding_boxes(ious, threshold)

        detection_results = [
            {
                "score" : p.score,
                "true_positive": p_idx in matched_prediction_idcs,
                "frame": frame,
                "bbox": p.bbox.abbrv_str()
            }
            for p_idx, p in enumerate(label_preds)
        ]

        detection_results_per_class[label] = {
            "gt_count":len(label_gts),
            "results":detection_results
        }

    return detection_results_per_class


def filter_detections_kb(detections: list[Detection3D]):
    # filter detections to only the experiment area
    # x between 2 and 50
    # y between -8 and 7
    return [det for det in detections
            if 2 < det.bbox.center_x < 50
            and -8 < det.bbox.center_y < 7
            ]


def match_bounding_boxes(ious, iou_threshold):
    row_indices, col_indices = linear_sum_assignment(-ious)  # maximizing IoU -> minimize -IoU

    filtered_row_indices = []
    filtered_col_indices = []

    for i in range(len(row_indices)):
        if ious[row_indices[i], col_indices[i]] > iou_threshold:
            filtered_row_indices.append(row_indices[i])
            filtered_col_indices.append(col_indices[i])

    return filtered_row_indices, filtered_col_indices

def lidar_run():
    connector = LiDARConnector(
        'lidar_centerpoint',
        '/sensor/lidar/top/points',
        '/perception/object_recognition/detection/centerpoint/objects',
    )

    # process_rosbags_3D(connector)
    process_waymo_3D(connector)