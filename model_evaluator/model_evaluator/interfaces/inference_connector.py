from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from model_evaluator.interfaces.detection2D import Detection2D
from model_evaluator.interfaces.detection3D import Detection3D


class InferenceConnector2D(ABC):
    @abstractmethod
    def run_inference(self, data: np.ndarray) -> Optional[list[Detection2D]]:
        raise NotImplementedError


class InferenceConnector3D(ABC):
    @abstractmethod
    def run_inference(self, data: np.ndarray) -> Optional[list[Detection3D]]:
        raise NotImplementedError
