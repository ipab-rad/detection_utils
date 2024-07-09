from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from model_evaluator.detection import Detection

class InferenceConnector(ABC):
    @abstractmethod
    def runInference(self, image: np.ndarray) -> Optional[list[Detection]]:
        raise NotImplementedError