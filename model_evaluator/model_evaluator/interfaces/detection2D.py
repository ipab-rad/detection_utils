from enum import IntFlag, auto


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


class Label2D(IntFlag):
    UNKNOWN = auto()
    CAR = auto()
    TRUCK = auto()
    BUS = auto()
    BICYCLE = auto()
    MOTORBIKE = auto()
    PEDESTRIAN = auto()
    ANIMAL = auto()

    VEHICLE = CAR | TRUCK | BUS | MOTORBIKE
    VRU = BICYCLE | PEDESTRIAN | ANIMAL
    ALL = (
        UNKNOWN | CAR | TRUCK | BUS | BICYCLE | MOTORBIKE | PEDESTRIAN | ANIMAL
    )


class Detection2D:
    bbox: BBox2D
    score: float
    label: Label2D

    def __init__(self, bbox: BBox2D, score: float, label: Label2D):
        self.bbox = bbox
        self.score = score
        self.label = label
