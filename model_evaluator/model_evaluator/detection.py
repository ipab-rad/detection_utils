from enum import IntEnum


class BBox:
    # TODO: Add asserts

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> 'BBox':
        bbox = BBox()

        bbox.x1 = x1
        bbox.y1 = y1
        bbox.x2 = x2
        bbox.y2 = y2

        return bbox
    
    @classmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> 'BBox':
        bbox = BBox()

        bbox.x1 = x
        bbox.y1 = y
        bbox.x2 = x + w
        bbox.y2 = y + h

        return bbox
    
    @classmethod
    def from_cxcywh(cx: float, cy: float, w: float, h: float) -> 'BBox':
        bbox = BBox()

        bbox.x1 = cx - w / 2
        bbox.y1 = cy - h / 2
        bbox.x2 = cx + w / 2
        bbox.y2 = cy + h / 2

        return bbox
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    

class Label(IntEnum):
    UNKNOWN = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    BICYCLE = 4
    MOTORBIKE = 5
    PEDESTRIAN = 6
    ANIMAL = 7

class Detection:
    bbox: BBox
    score: float
    label: Label

    def __init__(self, bbox: BBox, score: float, label: Label):
        self.bbox = bbox
        self.score = score
        self.label = label