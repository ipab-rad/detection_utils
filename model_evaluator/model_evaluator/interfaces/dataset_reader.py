from abc import ABC, abstractmethod
from typing import Generator, Optional

import numpy as np

from model_evaluator.interfaces.detection2D import Detection2D
from model_evaluator.interfaces.detection3D import Detection3D

from sensor_msgs.msg import PointCloud2


class DatasetReader2D(ABC):
    @abstractmethod
    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, Optional[list[Detection2D]]], None, None]:
        raise NotImplementedError


class DatasetReader3D(ABC):
    @abstractmethod
    def read_data(
        self,
    ) -> Generator[
        tuple[PointCloud2, Optional[list[Detection3D]]], None, None
    ]:
        raise NotImplementedError
