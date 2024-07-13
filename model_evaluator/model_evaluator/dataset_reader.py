from abc import ABC, abstractmethod

import numpy as np

from model_evaluator.detection import Detection2D, Detection3D


class DatasetReader2D(ABC):
    # TODO: use generator
    @abstractmethod
    def read_data(self) -> list[tuple[np.ndarray, list[Detection2D]]]:
        raise NotImplementedError


class DatasetReader3D(ABC):
    # TODO: use generator
    @abstractmethod
    def read_data(self) -> list[tuple[np.ndarray, list[Detection3D]]]:
        raise NotImplementedError
