from waymo_open_dataset import v2, label_pb2
import cv2
import numpy as np
from model_evaluator.detection2D import Detection2D, BBox2D, Label2D
from model_evaluator.detection3D import Detection3D, BBox3D

def decode_waymo_image(image_component: v2.CameraImageComponent) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(image_component.image, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

def decode_waymo_point_cloud(point_cloud_component: v2.LiDARComponent) -> np.ndarray:
    # need to figure out format ROS message requires (simple list of x,y,z,r points?)
    raise NotImplementedError

def decode_waymo_label_2D(label: int) -> Label2D:
    match label:
        case label_pb2.Label.TYPE_VEHICLE:
            return Label2D.VEHICLE
        case label_pb2.Label.TYPE_PEDESTRIAN:
            return Label2D.PEDESTRIAN
        case label_pb2.Label.TYPE_CYCLIST:
            return Label2D.BICYCLE
        case _:
            return Label2D.UNKNOWN

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
    box_component: v2.LidarBoxComponent,
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
        bbox = BBox3D.from_oriented(cx, cy, w, h)

        detections.append(Detection3D(bbox, 1.0))

    return detections