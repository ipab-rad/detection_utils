from abc import ABC, abstractmethod
import numpy as np

class ImageDatasetReader(ABC):
    # TODO: use generator
    @abstractmethod
    def readImages(self) -> list[np.ndarray]:
        raise NotImplementedError