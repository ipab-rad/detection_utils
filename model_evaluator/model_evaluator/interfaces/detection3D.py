import torch
import numpy as np
from math import acos

from model_evaluator.interfaces.labels import Label

from autoware_perception_msgs.msg import DetectedObject


class BBox3D:
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    heading: float

    # corners 1-8
    corners: torch.Tensor

    def __init__(self,
                 center_x: float,
                 center_y: float,
                 center_z: float,
                 length: float,
                 width: float,
                 height: float,
                 heading: float,
                 ):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.length = length
        self.width = width
        self.height = height
        self.heading = heading

    @staticmethod
    def from_oriented(
            center_x: float,
            center_y: float,
            center_z: float,
            length: float,
            width: float,
            height: float,
            heading: float,
    ) -> 'BBox3D':
        bbox = BBox3D(center_x, center_y, center_z, length, width, height, heading)
        l, w, h = length, width, height
        yaw = heading
        center = [center_x, center_y, center_z]

        corners = torch.tensor(
            [
                [-l / 2, -w / 2, -h / 2],
                [l / 2, -w / 2, -h / 2],
                [l / 2, w / 2, -h / 2],
                [-l / 2, w / 2, -h / 2],
                [-l / 2, -w / 2, h / 2],
                [l / 2, -w / 2, h / 2],
                [l / 2, w / 2, h / 2],
                [-l / 2, w / 2, h / 2],
            ],
            dtype=torch.float32,
        )

        rotation_matrix = torch.tensor(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )

        rotated_corners = torch.matmul(rotation_matrix, corners.T).T
        translated_corners = rotated_corners + torch.tensor(center)

        bbox.corners = translated_corners
        return bbox

    @staticmethod
    def from_detected_object(det_object: DetectedObject):
        center = det_object.kinematics.pose_with_covariance.pose.position
        dimensions = det_object.shape.dimensions

        quaternion = det_object.kinematics.pose_with_covariance.pose.orientation

        # since it's a yaw only rotation we can use a simple formula to extract the yaw from the quaternion
        yaw = 2.0 * acos(quaternion.w)

        bbox = BBox3D.from_oriented(center.x, center.y, center.z, dimensions.x, dimensions.y, dimensions.z, yaw)
        return bbox

    def __str__(self):
        return f"{self.center_x=} {self.center_y=} {self.center_z=} {self.length=} {self.width=} {self.height=} {self.heading=}"

    def __repr__(self):
        return self.__str__()

    def abbrv_str(self):
        center_str = f"[{round(self.center_x, 2)},{round(self.center_y, 2)},{round(self.center_z, 2)}]"
        dims_str = f"[{round(self.length, 2)},{round(self.width, 2)},{round(self.height, 2)}]"
        heading_str = f"{round(self.heading, 2)}"

        output = f"{center_str},{dims_str},{heading_str}"
        return output


class Detection3D:
    bbox: BBox3D
    score: float
    label: Label

    def __init__(self, bbox: BBox3D, score: float, label: Label):
        self.bbox = bbox
        self.score = score
        self.label = label