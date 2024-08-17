import json
import os
from typing import Optional
import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from model_evaluator.utils.video_writer import VideoWriter
from model_evaluator.connectors.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.interfaces.detection2D import Detection2D
from model_evaluator.readers.waymo_reader import (
    WaymoDatasetReader2D,
    parse_context_names_and_timestamps,
)
from model_evaluator.interfaces.labels import Label
from model_evaluator.utils.cv2_bbox_annotator import (
    draw_frame_number,
    draw_bboxes,
)
from model_evaluator.utils.metrics import KBMetricsCalculator, MetricsCalculator

from model_evaluator.utils.kb_rosbag_matcher import (
    match_rosbags_in_path,
)


def save(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)

def load(file):
    with open(file, 'r') as f:
        return json.load(f)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)



def inference_2d(
    data_generator,
    connector,
    metrics_calculator,
    video_file: str = None,
    video_size=(960, 640),
    video_fps=10,
    video_annotations: set[Label] = set([Label.PEDESTRIAN, Label.BICYCLE]),
):
    if video_file is not None:
        video_writer = VideoWriter(video_file, video_size, video_fps)

    for frame_counter, (image, ground_truths) in enumerate(data_generator):
        detections = connector.run_inference(image)

        if detections is None:
            raise RuntimeError(f'Inference failed for frame {frame_counter}')

        metrics_calculator.add_frame(detections, ground_truths)

        if video_writer:
            draw_frame_number(image, frame_counter)

            draw_bboxes(image, [ground_truth for ground_truth in ground_truths if ground_truth.label in video_annotations], colour=(0, 255, 0))
            draw_bboxes(image, [detection for detection in detections if detection.label in video_annotations], colour=(0, 0, 255))

            image = cv.resize(image, video_size)
            video_writer.write_frame(image)

    video_writer.release()

def rosbags_evaluation(connector, rosbags, labels, distances, video_directory):
    labels_map = dict([(label, i) for i, label in enumerate(labels)])
    distances_map = dict([(distance, i) for i, distance in enumerate(distances)])

    errors = np.zeros((len(distances), len(labels), 3), dtype=np.float32)
    counts = np.zeros(len(distances), dtype=np.int32)

    for rosbag in rosbags:
        print(rosbag.metadata.name)

        reader = rosbag.get_reader_2d()

        if not reader.has_expectations:
            continue

        video_writer = VideoWriter(os.path.join(video_directory, os.path.basename(rosbag.path)), (2272, 1088), 20)
        metrics = KBMetricsCalculator(labels)

        for frame, (image, expectations) in enumerate(reader.read_data()):
            detections = connector.run_inference(image)

            if detections is None:
                raise RuntimeError(f'Inference failed for frame {frame}')

            metrics.add_frame(detections, expectations)

            draw_frame_number(image, frame)
            draw_bboxes(image, [detection for detection in detections if detection.label in Label.VRU])

            video_writer.write_frame(image)

        video_writer.release()

        error_dict = metrics.error()

        distance_index = distances_map.get(rosbag.metadata.distance)
        if distance_index is not None:
            counts[distance_index] += 1

            for label in error_dict:
                label_index = labels_map[label]

                errors[distance_index, label_index, :] += error_dict[label]

    errors = errors / counts.reshape((-1, 1, 1))

    return errors

def rosbags_run(connector, base_path):
    labels = [Label.PEDESTRIAN, Label.BICYCLE, Label.VRU]

    rosbags = match_rosbags_in_path(
        '/opt/ros_ws/rosbags/kings_buildings_data/kings_building_2024_08_02'
    )

    rosbags.sort(key=lambda x: x.metadata.timestamp)

    distances = list(set([rosbag.metadata.distance for rosbag in rosbags if rosbag.metadata.distance is not None]))
    distances.sort()

    ideal_rosbags = [rosbag for rosbag in rosbags if rosbag.metadata.is_ideal]
    rosbags = [rosbag for rosbag in rosbags if not rosbag.metadata.is_ideal]

    errors = rosbags_evaluation(connector, rosbags, labels, distances, os.path.join(base_path, 'normal'))
    errors_ideal = rosbags_evaluation(connector, ideal_rosbags, labels, distances, os.path.join(base_path, 'ideal'))

    errors_ideal = errors_ideal[:, :, 1]

    save(os.path.join(base_path, 'normal.json'), errors)
    save(os.path.join(base_path, 'ideal.json'), errors_ideal)

def rosbag_plots(base_path):
    errors = load(os.path.join(base_path, 'normal.json'))
    errors_ideal = load(os.path.join(base_path, 'ideal.json'))

    distances_ideal = ['5m', '10m', '20m', '50m', '100m']
    distances = ['5m', '10m', '20m', '50m']

    sns.heatmap(errors[:, 2, :], annot=True, cbar=True, xticklabels=['No VRU', 'Little occlusion', 'Much occlusion'], yticklabels=distances)
    plt.savefig(os.path.join(base_path, 'normal.png'))

    plt.close()

    sns.heatmap(errors_ideal[:, 2].reshape((-1, 1)), annot=True, cbar=True, xticklabels=['No occlusion'], yticklabels=distances_ideal)
    plt.savefig(os.path.join(base_path, 'ideal.png'))


def waymo_run(connector, base_path):
    waymo_contexts_dict = parse_context_names_and_timestamps(
        '/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/2d_pvps_validation_frames.txt'
    )

    waymo_labels = [
        #Label.UNKNOWN,
        Label.PEDESTRIAN,
        Label.BICYCLE,
        #Label.VEHICLE,
    ]

    metrics = {}

    for context_name in waymo_contexts_dict:
        print(context_name)

        waymo_reader = WaymoDatasetReader2D(
            '/opt/ros_ws/rosbags/waymo/validation',
            context_name,
            waymo_contexts_dict[context_name],
            [1],
        )

        metrics_calculator = MetricsCalculator([Label.PEDESTRIAN, Label.BICYCLE])
        
        inference_2d(
            waymo_reader.read_data(),
            connector,
            metrics_calculator,
            video_file=os.path.join(base_path, context_name),
            video_size=(1920, 1280),
            video_fps=5
        )

        metrics[context_name] = metrics_calculator

    save_pickle(os.path.join(base_path, 'metrics.pickle'), metrics)

def waymo_results_lamr(base_path):
    metrics = load_pickle(os.path.join(base_path, 'metrics.pickle'))

    lamrs = []
    lamrs_dict = {
        Label.PEDESTRIAN: [],
        Label.BICYCLE: []
    }


    for context in metrics:
        print(context)

        log_average_miss_rate, log_average_miss_rate_dict = metrics[context].log_average_miss_rate(os.path.join(base_path, f'{context}.png'))

        lamrs.append(log_average_miss_rate)

        print(f'LAMR={log_average_miss_rate:.4f}')

        for label in log_average_miss_rate_dict:
            print(f'{label.name:10}: LAMR={log_average_miss_rate_dict[label]:.4f}')

            lamrs_dict[label].append(log_average_miss_rate_dict[label])

    print()
    print(f'Overall LAMR: {np.nanmean(lamrs):.4f}')
    for label in lamrs_dict:
            print(f'{label.name:10}: LAMR={np.nanmean(lamrs_dict[label]):.4f}')

def waymo_results_map(base_path):
    metrics = load_pickle(os.path.join(base_path, 'metrics.pickle'))

    for context in metrics:
        print(context)

        mean_average_precision, average_precision_dict = metrics[context].mean_average_precision(0.5)

        print(f'mAP={mean_average_precision:.3f}')

        for label in average_precision_dict:
            print(f'{label.name:10}: AP={average_precision_dict[label]:.3f}')

def camera_run():
    # rosbags = match_rosbags_in_path(
    #     '/opt/ros_ws/rosbags/kings_buildings_data/kings_building_2024_08_02'
    # )

    # for rosbag in rosbags:
    #     print(rosbag.metadata.name)

    #     video_file = os.path.join('/opt/ros_ws/src/deps/external/detection_utils/kings_buildings_videos_new/', rosbag.metadata.name)

    #     reader = rosbag.get_reader_2d()

    #     with VideoWriter(video_file, (2272, 1088), 20) as video_writer:
    #         for frame, (image, _) in enumerate(reader.read_data()):
    #             draw_frame_number(image, frame)

    #             video_writer.write_frame(image)


    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_rect_color',
        '/perception/object_recognition/detection/rois0',
    )

    # rosbags_run(connector, '/opt/ros_ws/src/deps/external/detection_utils/results/rosbags_tiny')

    waymo_run(connector, '/opt/ros_ws/src/deps/external/detection_utils/results/waymo_lamr_test')

    # waymo_results_map('/opt/ros_ws/src/deps/external/detection_utils/results/waymo_map')
    # waymo_results_lamr('/opt/ros_ws/src/deps/external/detection_utils/results/waymo_lamr')

    # data = load_pickle('/opt/ros_ws/src/deps/external/detection_utils/results/waymo_lamr/metrics.pickle')

    # # # x = data['5372281728627437618_2005_000_2025_000']
    # x = data['1024360143612057520_3580_000_3600_000']

    # m = MetricsCalculator([Label.PEDESTRIAN])

    # detections, gts, _ = x.frames[0]

    # m.add_frame(detections, gts)

    # detections, gts, _ = x.frames[1]

    # m.add_frame(detections, gts)

    # print(m.log_average_miss_rate('test.png'))

    # score_detections_idxs = MetricsCalculator.filter_score(detections, 0.0)
    # score_detections = [detections[i] for i in score_detections_idxs]

    # ped_detections_idxs = MetricsCalculator.filter_label(score_detections, Label.PEDESTRIAN)
    # ped_gts_idxs = MetricsCalculator.filter_label(gts, Label.PEDESTRIAN)

    # ped_detections = [score_detections[i] for i in ped_detections_idxs]
    # ped_gts = [gts[i] for i in ped_gts_idxs]

    # ious = MetricsCalculator.calculate_ious(ped_detections, ped_gts)

    # matches = MetricsCalculator.correct_matches(ious, 0.5)

    # print(f'{np.sum(matches)} / {len(ped_gts)}  -  {len(ped_detections) - np.sum(matches)}')
    # print([detection for i, detection in enumerate(ped_detections) if matches[i]])