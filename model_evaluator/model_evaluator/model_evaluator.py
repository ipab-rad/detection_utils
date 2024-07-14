from model_evaluator.waymo_reader import WaymoDatasetReader2D
from model_evaluator.detection import BBox2D, Label2D
from model_evaluator.yolox_connector import TensorrtYOLOXConnector
from model_evaluator.cv2_bbox_annotator import to_cv_pts, draw_bboxes, write_png_img
from model_evaluator.metrics_calculator import get_tp_fp, calculate_ap,calculate_ious_2d
import numpy as np

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
            detection
            for detection in detections
            if detection.label in label
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
    connector = TensorrtYOLOXConnector(
        '/sensor/camera/fsp_l/image_raw',
        '/perception/object_recognition/detection/rois0',
    )

    reader = WaymoDatasetReader2D('/opt/ros_ws/rosbags/waymo/validation')

    data = reader.read_data()

    for frame_counter, (image, gts) in enumerate(data):
        detections = connector.run_inference(image)

        process_frame_results(image,gts, detections,frame_counter)
