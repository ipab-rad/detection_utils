from typing import Generator, Optional

import cv2
import numpy as np
import dask.dataframe as dd
from waymo_open_dataset import v2, label_pb2
from sensor_msgs.msg import PointCloud2

from model_evaluator.interfaces.dataset_reader import (
    DatasetReader2D,
    DatasetReader3D,
)
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D
from model_evaluator.interfaces.detection3D import Detection3D, BBox3D
from model_evaluator.interfaces.labels import Label


def parse_context_names_and_timestamps(
    context_name_timestamp_file: str,
) -> dict[str, list[int]]:
    context_names = {}

    with open(context_name_timestamp_file) as file:
        for line in file.readlines():
            # TODO: add asserts

            context_name, timestamp = line.split(',')

            timestamp = int(timestamp)

            if context_name not in context_names:
                context_names[context_name] = [timestamp]
            else:
                context_names[context_name].append(timestamp)

    return context_names


class WaymoDatasetReaderBase:
    def __init__(self, dataset_dir: str, context_name: str):
        self.dataset_dir = dataset_dir
        self.context_name = context_name

    def read(self, tag):
        return dd.read_parquet(
            f'{self.dataset_dir}/{tag}/{self.context_name}.parquet'
        )


class WaymoDatasetReader2D(WaymoDatasetReaderBase, DatasetReader2D):
    def __init__(
        self,
        dataset_dir: str,
        context_name: str,
        timestamps: Optional[list[int]] = None,
        included_cameras: Optional[list[int]] = None,
    ):
        super().__init__(dataset_dir, context_name)

        self.timestamps = timestamps
        self.included_cameras = included_cameras

    @staticmethod
    def decode_waymo_image(
        image_component: v2.CameraImageComponent,
    ) -> np.ndarray:
        return cv2.imdecode(
            np.frombuffer(image_component.image, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
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

    @staticmethod
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
            label = WaymoDatasetReader2D.decode_waymo_label_2D(label)

            detections.append(Detection2D(bbox, 1.0, label))

        return detections

    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, Optional[list[Detection2D]]], None, None]:
        cam_image_df = self.read('camera_image')
        cam_box_df = self.read('camera_box')

        if self.included_cameras is not None:
            cam_image_df = cam_image_df[
                cam_image_df['key.camera_name'].isin(self.included_cameras)
            ]

        image_w_box_df = v2.merge(cam_image_df, cam_box_df, right_group=True)

        if self.timestamps is not None:
            image_w_box_df = image_w_box_df[
                image_w_box_df['key.frame_timestamp_micros'].isin(
                    self.timestamps
                )
            ]

        for _, r in image_w_box_df.iterrows():
            cam_image = v2.CameraImageComponent.from_dict(r)
            cam_box = v2.CameraBoxComponent.from_dict(r)

            image = self.decode_waymo_image(cam_image)
            detections = self.decode_waymo_camera_detections(cam_box)

            yield image, detections


class WaymoDatasetReader3D(WaymoDatasetReaderBase, DatasetReader3D):
    def __init__(
        self,
        dataset_dir: str,
        context_name: str,
        timestamps: Optional[list[int]] = None,
    ):
        super().__init__(dataset_dir, context_name)

        self.timestamps = timestamps

    @staticmethod
    def decode_waymo_point_cloud(
        point_cloud_component: v2.LiDARComponent,
    ) -> PointCloud2:
        return PointCloud2()

    @staticmethod
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

    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, Optional[list[Detection3D]]], None, None]:
        lidar_df = self.read('lidar')
        lidar_box_df = self.read('lidar_box')

        lidar_w_box_df = v2.merge(lidar_df, lidar_box_df, right_group=True)

        if self.timestamps is not None:
            lidar_w_box_df = lidar_w_box_df[
                lidar_w_box_df['key.frame_timestamp_micros'].isin(
                    self.timestamps
                )
            ]

        for _, r in lidar_w_box_df.iterrows():
            point_cloud = v2.LiDARComponent.from_dict(r)
            lidar_box = v2.LiDARBoxComponent.from_dict(r)

            point_cloud = self.decode_waymo_point_cloud(point_cloud)
            detections = self.decode_waymo_lidar_detections(lidar_box)

            yield point_cloud, detections
