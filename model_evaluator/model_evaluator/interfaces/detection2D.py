from enum import Enum, auto
import math
from model_evaluator.interfaces.labels import Label

class BBox2D:
    # TODO: Add asserts

    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> 'BBox2D':
        bbox = BBox2D()

        bbox.x1 = x1
        bbox.y1 = y1
        bbox.x2 = x2
        bbox.y2 = y2

        return bbox

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> 'BBox2D':
        bbox = BBox2D()

        bbox.x1 = x
        bbox.y1 = y
        bbox.x2 = x + w
        bbox.y2 = y + h

        return bbox

    @staticmethod
    def from_cxcywh(cx: float, cy: float, w: float, h: float) -> 'BBox2D':
        bbox = BBox2D()

        bbox.x1 = cx - w / 2
        bbox.y1 = cy - h / 2
        bbox.x2 = cx + w / 2
        bbox.y2 = cy + h / 2

        return bbox
    
    def __str__(self) -> str:
        return f'[{self.x1}, {self.y1}, {self.x2}, {self.y2}]'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def width(self) -> float:
        return (self.x2 - self.x1)
    
    def height(self) -> float:
        return (self.y2 - self.y1)
    
    def center(self) -> tuple[float, float]:
        return self.x1 + self.width() / 2, self.y1 + self.height() / 2

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: 'BBox2D') -> float:
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)

        if right < left or bottom < top:
            return 0.0

        intersect_area = (right - left) * (bottom - top)

        return intersect_area / (self.area() + other.area() - intersect_area)


class DifficultyLevel(Enum):
    """Difficulty level for detecting object. Higher level is harder"""

    UNKNOWN = auto()
    LEVEL_1 = auto()
    LEVEL_2 = auto()


class Detection2D:
    """Detected object in an image"""

    # TODO: rename to Object2D

    bbox: BBox2D
    """Bounding box of object"""

    label: Label
    """Label of object"""

    score: float
    """Confidence score for detected object, NaN for ground truth object"""

    difficulty_level: DifficultyLevel
    """Difficulty level for detecting object"""

    def __init__(self, bbox: BBox2D, label: Label, score: float = math.nan, difficulty_level: DifficultyLevel = DifficultyLevel.UNKNOWN):
        self.bbox = bbox
        self.label = label
        self.score = score
        self.difficulty_level = difficulty_level

    def __str__(self) -> str:
        return f'{self.label} @ {self.bbox}'

    def __repr__(self) -> str:
        return self.__str__()
