from abc import ABC, abstractmethod
from typing import Generator

import numpy as np

from model_evaluator.interfaces.detection2D import Detection2D
from model_evaluator.interfaces.detection3D import Detection3D


class DatasetReader2D(ABC):
    @abstractmethod
    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, list[Detection2D]], None, None]:
        raise NotImplementedError


class DatasetReader3D(ABC):
    @abstractmethod
    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, list[Detection3D]], None, None]:
        raise NotImplementedError
