from itertools import islice

import numpy as np
import cv2
import imageio


from model_evaluator.rosbag_reader import RosbagDatasetReader2D, RosbagDatasetReader3D

from model_evaluator.waymo_reader import (
    WaymoDatasetReader2D,
)
from model_evaluator.interfaces.detection2D import Label2D
from model_evaluator.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.lidar_connector import LiDARConnector
from model_evaluator.utils.cv2_bbox_annotator import (
    draw_bboxes,
)
from model_evaluator.utils.metrics_calculator import (
    get_tp_fp,
    get_unmatched_tp_fp,
    calculate_ap,
    calculate_ious_2d,
)

from model_evaluator.utils.kb_rosbag_matcher import match_rosbags_in_path


def process_images(
    data,
    connector,
    gif_path=None,
    iou_thresholds=None,
    tp_fp_function=get_tp_fp,
):
    images = []
    mean_avg_precisions = []

    if iou_thresholds is None:
        iou_thresholds = {Label2D.PEDESTRIAN: 0.5}

    for frame_counter, (image, gts) in enumerate(data):
        detections = connector.run_inference(image)

        if detections is None:
            print('Inference failed')
            continue

        avg_precisions = []

        for label in iou_thresholds:
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

            tps, fps = tp_fp_function(ious, iou_thresholds[label])

            avg_precisions.append(calculate_ap(tps, fps, len(label_gts)))

        mean_avg_precision = np.mean(avg_precisions)
        mean_avg_precisions.append(mean_avg_precision)

        if gif_path is not None and frame_counter % 5 == 0:
            draw_bboxes(image, gts, detections, Label2D.VRU | Label2D.UNKNOWN)
            images.append(image)

    if gif_path is not None:
        imageio.v3.imwrite(
            gif_path,
            [
                cv2.cvtColor(
                    cv2.resize(image, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB
                )
                for image in images
            ],
            fps=2,
            loop=0,
        )

    return mean_avg_precisions

def process_rosbags(connector):
    rosbags = match_rosbags_in_path('/opt/ros_ws/rosbags/kings_buildings_data')
    print(rosbags[0])

    rosbag_reader = RosbagDatasetReader3D(
        rosbags[0].path
    )

    rosbag_data = rosbag_reader.read_data()

    point_cloud, bboxes = next(rosbag_data)

    detections = connector.run_inference(point_cloud)

def process_waymo(connector):
    waymo_reader = WaymoDatasetReader2D(
        '/opt/ros_ws/rosbags/waymo/validation',
        '/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/2d_pvps_validation_frames.txt',
        [1],
    )

    waymo_data = islice(waymo_reader.read_data(), 50)
    waymo_maps = process_images(waymo_data, connector, 'waymo.gif')
    waymo_mAP = np.mean(waymo_maps)

    print(f'{waymo_mAP=}')

def main():
    connector = LiDARConnector(
        'lidar_centerpoint',
        '/sensor/lidar/top/points',
        '/perception/object_recognition/detection/rois0',
    )

    process_rosbags(connector)

    # process_waymo(connector)

