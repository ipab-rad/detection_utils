import re
import glob

import numpy as np

from model_evaluator.rosbag_reader import RosbagDatasetReader2D

from model_evaluator.waymo_reader import WaymoDatasetReader2D, WaymoDatasetReader3D
from model_evaluator.interfaces.detection2D import Label2D
from model_evaluator.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.utils.cv2_bbox_annotator import (
    draw_bboxes,
    write_png_img,
)
from model_evaluator.utils.metrics_calculator import (
    get_tp_fp,
    calculate_ap,
    calculate_ious_2d,
)


def process_frame_results(image, gts, detections, frame_counter):
    if detections is None:
        print('Inference failed')
        return

    draw_bboxes(image, gts, detections, Label2D.ALL)
    write_png_img(f'image{frame_counter}', image)

    aps = {}
    labels = [Label2D.PEDESTRIAN, Label2D.BICYCLE, Label2D.VEHICLE]

    per_frame_average_precision(gts, detections, aps, labels)


def per_frame_average_precision(gts, detections, aps, labels):
    for label in labels:
        label_gts = [gt for gt in gts if gt.label in label]
        label_detections = [
            detection for detection in detections if detection.label in label
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


def main():
    pattern = re.compile(
        r'.*/(?P<date>\d{4}_\d{2}_\d{2})-(?P<time>\d{2}_\d{2}_\d{2})_(?P<name>.+).mcap'
    )
    name_pattern = re.compile(
        r'(?P<distance>\d+m)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)_(?P<bag_no>\d)'
    )

    paths = glob.glob(
        '/opt/ros_ws/rosbags/kings_buildings_data/**', recursive=True
    )

    for path in paths:

        match = pattern.match(path)

        if match:
            date = match.group('date')
            time = match.group('time')
            name = match.group('name')

            name_match = name_pattern.match(name)

            print(f'{date=} {time=} {name=} {name_match=}')

    rosbag_reader = RosbagDatasetReader2D(
        '/opt/ros_ws/rosbags/kings_buildings_data/2024_07_12-10_58_56_5m_1_ped_0/2024_07_12-10_58_39_5m_1_ped_0/2024_07_12-10_58_39_5m_1_ped_0_0.mcap'
    )

    _ = rosbag_reader.read_data()

    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_raw',
        '/perception/object_recognition/detection/rois0',
    )

    reader = WaymoDatasetReader2D('/opt/ros_ws/rosbags/waymo/validation', '/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/2d_pvps_validation_frames.txt', [1])

    data = reader.read_data()

    for frame_counter, (image, gts) in enumerate(data):
        detections = connector.run_inference(image)

        process_frame_results(image, gts, detections, frame_counter)
