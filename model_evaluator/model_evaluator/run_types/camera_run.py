import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
    create_video_writer,
    draw_metrics,
)
from model_evaluator.utils.metrics import KBMetrics, Metrics, log_average_miss_rate
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

    all_detections = []
    all_gts = []

    metrics = Metrics(0.5)

    for frame_counter, (image, gts) in enumerate(data_generator):
        detections = connector.run_inference(image)

        if detections is None:
            # TODO: Handle accordingly
            print(f'Inference failed for frame {frame_counter}')
            continue

        metrics.add_frame(detections, gts)

        all_detections.append(detections)
        all_gts.append(gts)

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

            draw_frame_number(image, frame_counter)
            draw_metrics(image, threshold, mean_ap, video_annotations, num_gts_per_label, tps_per_label[:, 0], fps_per_label[:, 0], mrs_per_label[:, 0], aps_per_label[:, 0])

            image = cv.resize(image, video_size)
            video_writer.write(image)

    video_writer.release()

    print(f'metrics: {metrics.log_average_miss_rate()} {metrics.mean_average_precision(0.5)}')

    return all_detections, all_gts, mean_aps

def rosbags_run(connector):
    rosbags = match_rosbags_in_path(
        '/opt/ros_ws/rosbags/kings_buildings_data/kings_building_2024_08_02'
    )

    rosbags.sort(key=lambda x: x.metadata.timestamp)

    distances = list(set([rosbag.metadata.distance for rosbag in rosbags if rosbag.metadata.distance is not None]))
    distances.sort()

    distances_map = dict([(distance, i) for i, distance in enumerate(distances)])

    incorrect = np.zeros((len(distances), 3), dtype=np.float32)
    counts = np.zeros(len(distances), dtype=np.int32)

    for rosbag in rosbags:
        print(rosbag.metadata.name)

        reader = rosbag.get_reader_2d()

        if not reader.has_expectations:
            continue


        video_writer = create_video_writer(f'/opt/ros_ws/src/deps/external/detection_utils/kings_buildings_videos/{rosbag.metadata.name}', (2272, 1088), 20)

        metrics = KBMetrics(0.0)

        for frame, (image, expectations) in enumerate(reader.read_data()):
            detections = connector.run_inference(image)

            metrics.add_frame(detections, expectations)

            draw_frame_number(image, frame)

            draw_bboxes(image, [], [], detections)

            video_writer.write(image)

        video_writer.release()

        accuracy = metrics.accuracy()

        distance_index = distances_map.get(rosbag.metadata.distance)

        if distance_index is not None:
            counts[distance_index] += 1

            incorrect[distance_index, :] = accuracy

    errors = incorrect / counts.reshape((-1, 1))

    sns.heatmap(errors, annot=True, cbar=True, xticklabels=['No pedestrian', 'No/little occlusion', 'Much occlusion'], yticklabels=distances)
    plt.savefig('heatmap.png')

def waymo_run(connector):
    waymo_contexts_dict = parse_context_names_and_timestamps(
        '/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/2d_pvps_validation_frames.txt'
    )

    waymo_labels = [
        #Label.UNKNOWN,
        Label.PEDESTRIAN,
        #Label.BICYCLE,
        #Label.VEHICLE,
    ]

    for context_name in waymo_contexts_dict:
        print(context_name)

        waymo_reader = WaymoDatasetReader2D(
            '/opt/ros_ws/rosbags/waymo/validation',
            context_name,
            waymo_contexts_dict[context_name],
            [1],
        )

        all_detections, all_gts, _ = inference_2d(
            waymo_reader.read_data(),
            connector,
            [0.5],
            f'/opt/ros_ws/src/deps/external/detection_utils/waymo_videos/{context_name}',
            video_size=(1920, 1280),
            video_annotations=waymo_labels,
            video_fps=5
        )

        all_detections_gts = [(detections, gts) for detections, gts in zip(all_detections, all_gts)]

        print(log_average_miss_rate(all_detections_gts))

        with open(f'/opt/ros_ws/src/deps/external/detection_utils/waymo_videos/{context_name}.pickle', 'wb') as f:
            pickle.dump(all_detections, f, protocol=pickle.HIGHEST_PROTOCOL)

def camera_run():
    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_rect_color',
        '/perception/object_recognition/detection/rois0',
    )

    rosbags_run(connector)

    # waymo_run(connector)