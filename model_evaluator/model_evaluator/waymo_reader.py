import glob
from typing import Generator

import numpy as np
import dask.dataframe as dd
from waymo_open_dataset import v2

from model_evaluator.interfaces.dataset_reader import DatasetReader2D, DatasetReader3D
from model_evaluator.interfaces.detection2D import Detection2D
from model_evaluator.interfaces.detection3D import Detection3D
from model_evaluator.utils.decoders import (
    decode_waymo_image,
    decode_waymo_camera_detections,
    decode_waymo_point_cloud,
    decode_waymo_lidar_detections,
)

class WaymoDatasetReaderBase():
    def __init__(self, dataset_dir: str, context_name_timestamp_file: str):
        self.dataset_dir = dataset_dir

        self.context_name_timestamp_file = context_name_timestamp_file

        self.contexts = self.parse_context_names_and_timestamps()

    def parse_context_names_and_timestamps(self):
        context_names = {}

        with open(self.context_name_timestamp_file) as file:
            for line in file.readlines():
                # TODO: add asserts

                context_name, timestamp = line.split(',')

                timestamp = int(timestamp)

                if context_name not in context_names:
                    context_names[context_name] = [timestamp]
                else:
                    context_names[context_name].append(timestamp)

        return context_names

    def read(self, tag, context_name):
        paths = glob.glob(f'{self.dataset_dir}/{tag}/{context_name}.parquet')
        return dd.read_parquet(paths)


class WaymoDatasetReader2D(WaymoDatasetReaderBase, DatasetReader2D):
    def __init__(self, dataset_dir: str, context_name_timestamp_file: str, included_cameras: list[int]):
        super().__init__(dataset_dir, context_name_timestamp_file)

        self.included_cameras = included_cameras

    def read_data(self) -> Generator[tuple[np.ndarray, list[Detection2D]], None, None]:
        for context_name in self.contexts:
            cam_image_df = self.read('camera_image', context_name)
            cam_box_df = self.read('camera_box', context_name)

            image_w_box_df = v2.merge(
                cam_image_df, cam_box_df, right_group=True
            )

            for timestamp in self.contexts[context_name]:
                frame_df = image_w_box_df.loc[
                    image_w_box_df['key.frame_timestamp_micros'] == timestamp
                ]

                frame_df = frame_df.loc[
                    frame_df['key.camera_name'].isin(self.included_cameras)
                ]

                for _, r in frame_df.iterrows():
                    cam_image = v2.CameraImageComponent.from_dict(r)
                    cam_box = v2.CameraBoxComponent.from_dict(r)

                    image = decode_waymo_image(cam_image)
                    detections = decode_waymo_camera_detections(cam_box)

                    yield image, detections


class WaymoDatasetReader3D(WaymoDatasetReaderBase, DatasetReader3D):
    def read_data(self) -> Generator[tuple[np.ndarray, list[Detection3D]], None, None]:
        for context_name in self.contexts:
            lidar_df = self.read('lidar', context_name)
            lidar_box_df = self.read('lidar_box', context_name)

            lidar_w_box_df = v2.merge(lidar_df, lidar_box_df, right_group=True)

            for timestamp in self.contexts[context_name]:
                frame_df = lidar_w_box_df.loc[
                    lidar_w_box_df['key.frame_timestamp_micros'] == timestamp
                ]

                for _, r in frame_df.iterrows():
                    point_cloud = v2.LiDARComponent.from_dict(r)
                    lidar_box = v2.LiDARBoxComponent.from_dict(r)

                    point_cloud = decode_waymo_point_cloud(point_cloud)
                    detections = decode_waymo_lidar_detections(lidar_box)

                    yield point_cloud, detections