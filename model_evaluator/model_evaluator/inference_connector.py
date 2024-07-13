import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from model_evaluator.detection import Detection2D, Detection3D


class InferenceConnector2D(ABC):
    @abstractmethod
    def run_inference(self, data: np.ndarray) -> Optional[list[Detection2D]]:
        raise NotImplementedError
    

class InferenceConnector3D(ABC):
    @abstractmethod
    def run_inference(self, data: np.ndarray) -> Optional[list[Detection3D]]:
        raise NotImplementedError