from itertools import islice
import re
import glob

import numpy as np
import cv2
import imageio


from model_evaluator.rosbag_reader import RosbagDatasetReader2D

from model_evaluator.waymo_reader import (
    WaymoDatasetReader2D,
)
from model_evaluator.interfaces.detection2D import Label2D
from model_evaluator.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.utils.cv2_bbox_annotator import (
    draw_bboxes,
    write_png_img,
)
from model_evaluator.utils.metrics_calculator import (
    get_tp_fp,
    get_unmatched_tp_fp,
    calculate_ap,
    calculate_ious_2d,
)

def get_rosbags(directory):
    pattern = re.compile(
        r'.*/(?P<date>\d{4}_\d{2}_\d{2})-(?P<time>\d{2}_\d{2}_\d{2})_(?P<name>.+).mcap'
    )
    name_pattern = re.compile(
        r'(?P<distance>\d+m)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)_(?P<bag_no>\d)'
    )

    paths = glob.glob(
        f'{directory}/**', recursive=True
    )

    rosbags = []

    for path in paths:
        match = pattern.match(path)

        if match:
            name = match.group('name')

            name_match = name_pattern.match(name)

            if name_match:
                rosbags.append((path, {
                    Label2D.PEDESTRIAN: int(name_match.group('count'))
                }))

    return rosbags

def process_images(data, connector, gif_path=None, iou_thresholds=None, tp_fp_function=get_tp_fp):
    images = []
    maps = []

    if iou_thresholds is None:
        iou_thresholds = {Label2D.PEDESTRIAN: 0.5}

    for frame_counter, (image, gts) in enumerate(data):
        detections = connector.run_inference(image)

        if detections is None:
            print('Inference failed')
            continue

        aps = []

        for label in iou_thresholds:
            label_gts = [gt for gt in gts if gt.label in label]
            label_detections = [
                detection for detection in detections if detection.label in label
            ]

            label_detections.sort(key=lambda x: x.score, reverse=True)

            ious = calculate_ious_2d(
                [detection.bbox for detection in label_detections],
                [gt.bbox for gt in label_gts],
            )

            tps, fps = tp_fp_function(ious, iou_thresholds[label])

            aps.append(calculate_ap(tps, fps, len(label_gts)))

        map = np.mean(aps)
        maps.append(map)

        if gif_path is not None and frame_counter % 5 == 0:
            draw_bboxes(image, gts, detections, Label2D.VRU | Label2D.UNKNOWN)
            images.append(image)

    if gif_path is not None:
        imageio.v3.imwrite(gif_path, [cv2.cvtColor(cv2.resize(image, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB) for image in images], fps=2, loop=0)

    return maps


def main():
    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_rect_color',
        '/perception/object_recognition/detection/rois0',
    )

    rosbag_reader = RosbagDatasetReader2D(
        '/opt/ros_ws/rosbags/kings_buildings_data/2024_07_12-10_58_56_5m_1_ped_0/2024_07_12-10_58_56_5m_1_ped_0_0.mcap',
        {
            Label2D.PEDESTRIAN: 1
        }
    )

    rosbags = get_rosbags('/opt/ros_ws/rosbags/kings_buildings_data')

    rosbag_reader = RosbagDatasetReader2D(rosbags[0][0], rosbags[0][1])
    print(f'rosbag: {rosbag_reader.path} - expected VRUS: {rosbag_reader.expectations[Label2D.PEDESTRIAN]}')
    rosbag_data = rosbag_reader.read_data()

    rosbag_maps = process_images(rosbag_data, connector, 'rosbag.gif', {Label2D.PEDESTRIAN: 0.0}, get_unmatched_tp_fp)

    rosbag_raw_mAP = np.mean(rosbag_maps)

    cut_off = len(rosbag_maps) // 3
    rosbag_mAP = np.mean(rosbag_maps[cut_off:-cut_off])

    print(f'{rosbag_raw_mAP=}, {rosbag_mAP=}')

    waymo_reader = WaymoDatasetReader2D(
        '/opt/ros_ws/rosbags/waymo/validation',
        '/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/2d_pvps_validation_frames.txt',
        [1],
    )

    waymo_data = islice(waymo_reader.read_data(), 50)
    waymo_maps = process_images(waymo_data, connector, 'waymo.gif')
    waymo_mAP = np.mean(waymo_maps)

    print(f'{waymo_mAP=}')