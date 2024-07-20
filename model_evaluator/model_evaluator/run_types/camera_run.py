import cv2

from model_evaluator.readers.rosbag_reader import DatasetReaderInitialiser
from model_evaluator.readers.waymo_reader import (
    WaymoDatasetReader2D,
    parse_context_names_and_timestamps,
)
from model_evaluator.interfaces.labels import Label
from model_evaluator.connectors.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.utils.cv2_bbox_annotator import (
    draw_bboxes,
)
from model_evaluator.utils.metrics_calculator import (
    calculate_ious_per_label,
    calculate_tps_fps_per_label,
    calculate_mean_ap,
    calculate_fppi,
    calculate_mr,
)

from model_evaluator.utils.kb_rosbag_matcher import (
    match_rosbags_in_path,
)

def inference_2d(
    data_generator,
    connector,
    video_file: str = None,
    video_size=(960, 640),
    video_fps=10,
    video_annotations=[Label.VRU],
):
    if video_file is not None:
        if not video_file.endswith('.avi'):
            video_file += '.avi'

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(
            video_file, fourcc, video_fps, video_size
        )

    detections_gts = []

    for frame_counter, (image, gts) in enumerate(data_generator):
        detections = connector.run_inference(image)

        if detections is None:
            # TODO: Handle accordingly
            print(f'Inference failed for frame {frame_counter}')
            continue

        detections_gts.append((detections, gts))

        ious_per_label = calculate_ious_per_label(
            detections, gts, video_annotations
        )

        threshold = 0.7

        tps_fps_per_label = calculate_tps_fps_per_label(
            ious_per_label, threshold
        )

        num_gts = len(gts)

        mean_ap = calculate_mean_ap(tps_fps_per_label, num_gts)
        fppi = calculate_fppi(tps_fps_per_label)
        mr = calculate_mr(tps_fps_per_label, num_gts)

        if video_writer:
            text_height = 25
            offset = 10
            thickness = 2

            scale = cv2.getFontScaleFromHeight(
                cv2.FONT_HERSHEY_SIMPLEX, text_height, thickness
            )

            cv2.putText(
                image,
                f'mAP@{threshold:.2f}: {mean_ap:.2f}  FPPI: {fppi:.2f}  MR: {mr}',
                (0, text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                thickness,
            )

            for i, label in enumerate(video_annotations):
                label_gts = [gt for gt in gts if gt.label in label]
                num_label_gts = len(label_gts)
                label_tps, label_fps = tps_fps_per_label[label]

                draw_bboxes(image, label_gts, label_tps, label_fps)

                cv2.putText(
                    image,
                    f'{label.name}',
                    (0, (i + 2) * (text_height + offset)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 0),
                    thickness,
                )
                cv2.putText(
                    image,
                    f'{num_label_gts:2}',
                    (250, (i + 2) * (text_height + offset)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 0),
                    thickness,
                )
                cv2.putText(
                    image,
                    f'TP: {label_tps.sum():2}',
                    (350, (i + 2) * (text_height + offset)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 0),
                    thickness,
                )
                cv2.putText(
                    image,
                    f'FP: {label_fps.sum():2}',
                    (500, (i + 2) * (text_height + offset)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 0),
                    thickness,
                )

            image = cv2.resize(image, video_size)
            video_writer.write(image)

    video_writer.release()

    return detections_gts

def get_expectations(count: str) -> dict[Label, int]:
    # TODO: Add support for cycling rosbags
    return {Label.PEDESTRIAN: int(count)}

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
        f'rosbag: {rosbag.path} - expected VRUS: {get_expectations(rosbag.count)}'
    )

    rosbag_reader = DatasetReaderInitialiser().get_reader_2d(rosbag.path)
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

    waymo_detections_gts = inference_2d(
        waymo_data,
        connector,
        context_name,
        video_size=(1920, 1280),
        video_annotations=waymo_labels,
    )

    for detections, gts in waymo_detections_gts:
        ious_per_label = calculate_ious_per_label(detections, gts)

        tps_fps_per_label = calculate_tps_fps_per_label(ious_per_label, 0.7)

        num_gts = len(gts)

        fppi = calculate_fppi(tps_fps_per_label, num_gts)
        mean_ap = calculate_mean_ap(tps_fps_per_label, num_gts)
        mr = calculate_mr(tps_fps_per_label, num_gts)

        print(f'{fppi=:.2f} {mean_ap=:.2f} {mr=}')