import numpy as np
from typing import Optional

from model_evaluator.detection import Detection

class InferenceConnector:

    def runInference(self, image: np.ndarray) -> Optional[list[Detection]]:
        pass