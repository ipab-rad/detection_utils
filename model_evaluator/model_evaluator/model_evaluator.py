import cv2
import numpy as np

from model_evaluator.waymo_reader import (
    WaymoDatasetReader2D,
    parse_context_names_and_timestamps,
)
from model_evaluator.interfaces.labels import Label
from model_evaluator.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.lidar_connector import LiDARConnector
from model_evaluator.utils.cv2_bbox_annotator import (
    draw_bboxes,
    create_video_writer,
    draw_metrics,
)
from model_evaluator.utils.metrics_calculator import (
    calculate_ap,
    calculate_ious,
    calculate_tps_fps,
    calculate_mr,
)
from model_evaluator.utils.kb_rosbag_matcher import (
    match_rosbags_in_path,
)


def inference_2d(
    data_generator,
    connector,
    thresholds=[0.5],
    video_file: str = None,
    video_size=(960, 640),
    video_fps=10,
    video_annotations=[Label.VRU],
):
    if video_file is not None:
        video_writer = create_video_writer(video_file, video_size, video_fps)

    detections_gts = []

    for frame_counter, (image, gts) in enumerate(data_generator):
        detections = connector.run_inference(image)

        if detections is None:
            # TODO: Handle accordingly
            print(f'Inference failed for frame {frame_counter}')
            continue

        detections_gts.append((detections, gts))

        aps_per_label = np.empty((len(video_annotations), len(thresholds)))
        tps_per_label = np.empty((len(video_annotations), len(thresholds)), dtype=np.uint)
        fps_per_label = np.empty((len(video_annotations), len(thresholds)), dtype=np.uint)
        mrs_per_label = np.empty((len(video_annotations), len(thresholds)), dtype=np.uint)

        num_gts_per_label = np.empty(len(video_annotations), dtype=np.uint)

        for i, label in enumerate(video_annotations):
            label_gts = [gt for gt in gts if gt.label in label]
            label_detections = [
                detection for detection in detections if detection.label in label
            ]
            num_label_gts = len(label_gts)

            num_gts_per_label[i] = num_label_gts

            ious = calculate_ious(label_detections, label_gts)

            for j, threshold in enumerate(thresholds):
                tps, fps = calculate_tps_fps(ious, threshold)

                ap = calculate_ap(tps, fps, num_label_gts)
                mr = calculate_mr(tps, num_label_gts)

                aps_per_label[i, j] = ap
                tps_per_label[i, j] = tps.sum()
                fps_per_label[i, j] = fps.sum()
                mrs_per_label[i, j] = mr
                
                tps_detections = [detection for i, detection in enumerate(label_detections) if tps[i] == 1]
                fps_detections = [detection for i, detection in enumerate(label_detections) if fps[i] == 1]

                draw_bboxes(image, label_gts, tps_detections, fps_detections)

        mean_aps = np.nanmean(aps_per_label, axis=0)

        if video_writer:
            threshold = thresholds[0]
            mean_ap = mean_aps[0]

            draw_metrics(image, threshold, mean_ap, video_annotations, num_gts_per_label, tps_per_label[:, 0], fps_per_label[:, 0], mrs_per_label[:, 0], aps_per_label[:, 0])

            image = cv2.resize(image, video_size)
            video_writer.write(image)

    video_writer.release()

    return detections_gts

def camera_run():
    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_rect_color',
        '/perception/object_recognition/detection/rois0',
    )

    rosbags = match_rosbags_in_path(
        '/opt/ros_ws/rosbags/kings_buildings_data/'
    )

    rosbag = rosbags[0]
    print(rosbag)
    print(
        f'rosbag: {rosbag.path} - expected VRUS: {rosbag.get_expectations_2d()}'
    )

    rosbag_reader = rosbag.get_reader_2d()
    rosbag_data = rosbag_reader.read_data()

    image, _ = next(rosbag_data)

    detections = connector.run_inference(image)

    print(detections)

    waymo_contexts_dict = parse_context_names_and_timestamps(
        '/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/2d_pvps_validation_frames.txt'
    )
    context_name = list(waymo_contexts_dict.keys())[0]

    waymo_reader = WaymoDatasetReader2D(
        '/opt/ros_ws/rosbags/waymo/validation',
        context_name,
        waymo_contexts_dict[context_name],
        [1],
    )

    waymo_data = waymo_reader.read_data()

    waymo_labels = [
        Label.UNKNOWN,
        Label.PEDESTRIAN,
        Label.BICYCLE,
        Label.VEHICLE,
    ]

    inference_2d(
        waymo_data,
        connector,
        [0.5],
        context_name,
        video_size=(1920, 1280),
        video_annotations=waymo_labels,
    )

def process_rosbags_3D(connector):
    rosbags = match_rosbags_in_path('/opt/ros_ws/rosbags/kings_buildings_data')

    rosbags = [x for x in rosbags if x.distance == "10m" and x.vru_type == "ped" and x.take == "0" and x.count == "1"]

    print(rosbags[0])

    rosbag_reader = rosbags[0].get_reader_3d()

    rosbag_data = rosbag_reader.read_data()

    for frame_counter, (point_cloud, _) in enumerate(rosbag_data):
        if frame_counter == 230:
            detections = connector.run_inference(point_cloud)

            print(detections)

def lidar_run():
    connector = LiDARConnector(
        'lidar_centerpoint',
        '/sensor/lidar/top/points',
        '/perception/object_recognition/detection/centerpoint/objects',
    )

    process_rosbags_3D(connector)


def main():
    # camera_run()

    # lidar_run()

    return
