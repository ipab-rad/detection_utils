from abc import ABC, abstractmethod

import numpy as np

from model_evaluator.detection import Detection2D, Detection3D


class DatasetReader(ABC):
    # TODO: use generator
    @abstractmethod
    def read_data_2D(self) -> list[tuple[np.ndarray, list[Detection2D]]]:
        raise NotImplementedError

    def read_data_3D(self) -> list[tuple[np.ndarray, list[Detection3D]]]:
        raise NotImplementedError
