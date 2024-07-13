import glob

import numpy as np
import dask.dataframe as dd
from waymo_open_dataset import v2, label_pb2
import cv2

from model_evaluator.dataset_reader import DatasetReader2D
from model_evaluator.detection import Detection2D, BBox2D, Label2D


contexts = [
    '1024360143612057520_3580_000_3600_000,1553735853462203',
    '1024360143612057520_3580_000_3600_000,1553735853662172',
]

def parse_context_names_and_timestamps():
    lines = contexts

    context_names = {}
    for line in lines:
        context_name, timestamp = line.split(',')

        timestamp = int(timestamp)

        if context_name not in context_names:
            context_names[context_name] = [timestamp]
        else:
            context_names[context_name].append(timestamp)

    return context_names


def parse_waymo_label(label: int) -> Label2D:
    match label:
        case label_pb2.Label.TYPE_VEHICLE:
            return Label2D.VEHICLE
        case label_pb2.Label.TYPE_PEDESTRIAN:
            return Label2D.PEDESTRIAN
        case label_pb2.Label.TYPE_CYCLIST:
            return Label2D.BICYCLE
        case _:
            return Label2D.UNKNOWN


class WaymoDatasetReader2D(DatasetReader2D):
    @staticmethod
    def decode_image(image_component: v2.CameraImageComponent) -> np.ndarray:
        return cv2.imdecode(
            np.frombuffer(image_component.image, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def decode_label(label: int) -> Label2D:
        match label:
            case label_pb2.Label.TYPE_VEHICLE:
                return Label2D.VEHICLE
            case label_pb2.Label.TYPE_PEDESTRIAN:
                return Label2D.PEDESTRIAN
            case label_pb2.Label.TYPE_CYCLIST:
                return Label2D.BICYCLE
            case _:
                return Label2D.UNKNOWN

    @staticmethod
    def decode_detections(
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
            label = WaymoDatasetReader2D.decode_label(label)

            detections.append(Detection2D(bbox, 1.0, label))

        return detections

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def read(self, tag, context_name):
        paths = glob.glob(f'{self.dataset_dir}/{tag}/{context_name}.parquet')
        return dd.read_parquet(paths)

    def read_data(self) -> list[tuple[np.ndarray, list[Detection2D]]]:
        context_names = parse_context_names_and_timestamps()

        output = []

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

                    image = self.decode_image(cam_image)
                    detections = self.decode_detections(cam_box)

                    output.append((image, detections))

        return output
