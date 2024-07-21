from enum import Flag, auto
from autoware_perception_msgs.msg import ObjectClassification

class Label(Flag):
    UNKNOWN = auto()
    CAR = auto()
    TRUCK = auto()
    BUS = auto()
    BICYCLE = auto()
    MOTORCYCLE = auto()
    PEDESTRIAN = auto()

    VEHICLE = CAR | TRUCK | BUS | MOTORCYCLE
    VRU = BICYCLE | PEDESTRIAN
    ALL = UNKNOWN | CAR | TRUCK | BUS | BICYCLE | MOTORCYCLE | PEDESTRIAN

def parse_label(label: int) -> Label:
    match label:
        case ObjectClassification.UNKNOWN:
            return Label.UNKNOWN
        case ObjectClassification.CAR:
            return Label.CAR
        case ObjectClassification.TRUCK:
            return Label.TRUCK
        case ObjectClassification.BUS:
            return Label.BUS
        case ObjectClassification.BICYCLE:
            return Label.BICYCLE
        case ObjectClassification.MOTORCYCLE:
            return Label.MOTORCYCLE
        case ObjectClassification.PEDESTRIAN:
            return Label.PEDESTRIAN
        case _:
            return Label.UNKNOWN
