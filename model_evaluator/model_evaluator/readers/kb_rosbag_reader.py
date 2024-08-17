import json
from typing import Generator, Optional

import numpy as np
from model_evaluator.interfaces.dataset_reader import DatasetReader2D
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D, DifficultyLevel
from model_evaluator.interfaces.labels import Label
from model_evaluator.readers.rosbag_reader import RosbagDatasetReader2D


class KBRosbagReader2D(RosbagDatasetReader2D):
    _IMAGE_TOPIC = '/sensor/camera/fsp_l/image_rect_color'

    has_expectations: bool = False

    def __init__(self, rosbag_path: str, keyframes_file: Optional[str]):
        super().__init__(rosbag_path, self._IMAGE_TOPIC)

        if keyframes_file:
            self._keyframes = self._read_keyframes_file(keyframes_file)
            self.has_expectations = True
        else:
            self._keyframes = None

    @staticmethod
    def _parse_type(type: str):
        match(type):
            case 'CAR':
                return Label.CAR
            case 'TRUCK':
                return Label.TRUCK
            case 'BUS':
                return Label.BUS
            case 'BICYCLE':
                return Label.BICYCLE
            case 'MOTORCYCLE':
                return Label.MOTORCYCLE
            case 'PEDESTRIAN':
                return Label.PEDESTRIAN
            case _:
                return Label.UNKNOWN

    @staticmethod
    def _parse_occlusion_level(occluded: bool):
        if occluded:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EASY

    def _read_keyframes_file(self, file: str):
        with open(file, 'r') as f:
            data = json.load(f)

        keyframes = {}

        for keyframe in data:
            expectations = []

            for vru in data[keyframe]:
                label = self._parse_type(vru['type'])
                difficulty_level = self._parse_occlusion_level(vru['occluded'])

                expectations.append(Detection2D(BBox2D.from_xywh(0, 0, 0, 0), label, difficulty_level=difficulty_level))

            keyframes[int(keyframe)] = expectations

        return keyframes

    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, Optional[list[Detection2D]]], None, None]:
        last_keyframe = 0

        for i, (image, _) in enumerate(super().read_data()):
            expectations = None

            if self._keyframes:
                expectations = self._keyframes.get(i)

                if expectations is not None:
                    last_keyframe = i
                else:
                    expectations = self._keyframes.get(last_keyframe)

            yield image, expectations
