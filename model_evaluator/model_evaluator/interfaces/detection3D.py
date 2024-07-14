from enum import IntFlag, auto
import pytorch as torch
import numpy as np

class BBox3D:
    # TODO: Add asserts

    # corners 1-8
    corners: torch.Tensor

    @staticmethod
    def from_oriented(center_x:float, center_y:float, center_z:float,
                      length:float, width:float, height:float, heading:float) -> 'BBox3D':
        bbox = BBox3D()
        l,w,h = length,width,height
        yaw = -heading  # make sure this is correct
        center = [center_x,center_y,center_z]

        corners = torch.tensor([
            [-l / 2, -w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [l / 2, w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, w / 2, h / 2],
            [-l / 2, w / 2, h / 2]
        ], dtype=torch.float32)

        rotation_matrix = torch.tensor([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        rotated_corners = torch.matmul(rotation_matrix, corners.T).T
        translated_corners = rotated_corners + torch.tensor(center)

        bbox.corners = translated_corners
        return bbox


class Detection3D:
    bbox: BBox3D
    score: float

    def __init__(self, bbox: BBox3D, score: float):
        self.bbox = bbox
        self.score = score