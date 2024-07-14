from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from model_evaluator.detection import Detection2D, Detection3D


class InferenceConnector2D(ABC):
    @abstractmethod
    def run_inference_2D(self, data: np.ndarray) -> Optional[list[Detection2D]]:
        raise NotImplementedError


class InferenceConnector3D(ABC):
    @abstractmethod
    def run_inference(self, data: np.ndarray) -> Optional[list[Detection3D]]:
        raise NotImplementedError
