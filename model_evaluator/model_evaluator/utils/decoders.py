from waymo_open_dataset import v2, label_pb2
import cv2
import numpy as np
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D
from model_evaluator.interfaces.detection3D import Detection3D, BBox3D
from model_evaluator.interfaces.labels import Label
from sensor_msgs.msg import PointCloud2


def decode_waymo_image(image_component: v2.CameraImageComponent) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(image_component.image, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def decode_waymo_point_cloud(
    point_cloud_component: v2.LiDARComponent,
) -> PointCloud2:
    return PointCloud2()


def decode_waymo_label_2D(label: int) -> Label:
    match label:
        case label_pb2.Label.TYPE_VEHICLE:
            return Label.VEHICLE
        case label_pb2.Label.TYPE_PEDESTRIAN:
            return Label.PEDESTRIAN
        case label_pb2.Label.TYPE_CYCLIST:
            return Label.BICYCLE
        case _:
            return Label.UNKNOWN


def decode_waymo_camera_detections(
    box_component: v2.CameraBoxComponent,
) -> list[Detection2D]:
    detections = []

    for cx, cy, w, h, label in zip(
        box_component.box.center.x,
        box_component.box.center.y,
        box_component.box.size.x,
        box_component.box.size.y,
        box_component.type,
    ):
        bbox = BBox2D.from_cxcywh(cx, cy, w, h)
        label = decode_waymo_label_2D(label)

        detections.append(Detection2D(bbox, 1.0, label))

    return detections


def decode_waymo_lidar_detections(
    box_component: v2.LiDARComponent,
) -> list[Detection3D]:
    detections = []

    for cx, cy, cz, l, w, h, heading in zip(
        box_component.box.center.x,
        box_component.box.center.y,
        box_component.box.center.z,
        box_component.box.size.x,
        box_component.box.size.y,
        box_component.box.size.z,
        box_component.box.heading,
    ):
        bbox = BBox3D.from_oriented(cx, cy, cz, l, w, h, heading)

        detections.append(Detection3D(bbox, 1.0))

    return detections
