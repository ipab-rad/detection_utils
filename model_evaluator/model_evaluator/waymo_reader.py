import glob

import numpy as np
import dask.dataframe as dd
from waymo_open_dataset import v2, label_pb2

from model_evaluator.interfaces.dataset_reader import DatasetReader
from model_evaluator.detection import Detection2D, Detection3D, BBox2D, Label2D
from model_evaluator.utils.decoders import decode_waymo_image,decode_waymo_camera_detections

class WaymoDatasetReader(DatasetReader):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

        self.contexts = [
            '1024360143612057520_3580_000_3600_000,1553735853462203',
            '1024360143612057520_3580_000_3600_000,1553735853662172',
        ]

    def parse_context_names_and_timestamps(self):
        context_names = {}
        for line in self.contexts:
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

    def read_data_2D(self) -> list[tuple[np.ndarray, list[Detection2D]]]:
        context_names = self.parse_context_names_and_timestamps()

        for context_name in context_names:
            cam_image_df = self.read('camera_image', context_name)
            cam_box_df = self.read('camera_box', context_name)

            image_w_box_df = v2.merge(
                cam_image_df, cam_box_df, right_group=True
            )

            for timestamp in context_names[context_name]:
                frame_df = image_w_box_df.loc[
                    image_w_box_df['key.frame_timestamp_micros'] == timestamp
                ]

                for _, r in frame_df.iterrows():
                    cam_image = v2.CameraImageComponent.from_dict(r)
                    cam_box = v2.CameraBoxComponent.from_dict(r)

                    image = decode_waymo_image(cam_image)
                    detections = decode_waymo_camera_detections(cam_box)

                    yield image, detections

    def read_data_3D(self) -> list[tuple[np.ndarray, list[Detection3D]]]:
        context_names = self.parse_context_names_and_timestamps()

        output = []

        for context_name in context_names:
            lidar_df = self.read('lidar', context_name)
            lidar_box_df = self.read('lidar_box', context_name)

            lidar_w_box_df = v2.merge(
                lidar_df, lidar_box_df, right_group=True
            )

            for timestamp in context_names[context_name]:
                frame_df = lidar_w_box_df.loc[
                    lidar_w_box_df['key.frame_timestamp_micros'] == timestamp
                ]

                for _, r in frame_df.iterrows():
                    point_cloud = v2.LiDARComponent.from_dict(r)
                    lidar_box = v2.LiDARBoxComponent.from_dict(r)

                    point_cloud = self.decode_waymo_point_cloud(point_cloud)
                    detections = self.decode_waymo_lidar_detections(lidar_box)

                    output.append((point_cloud, detections))

        return output