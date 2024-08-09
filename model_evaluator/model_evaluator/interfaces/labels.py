from enum import Flag, auto
from autoware_perception_msgs.msg import ObjectClassification
from waymo_open_dataset import label_pb2

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

ALL_LABELS = [Label.UNKNOWN, Label.CAR, Label.TRUCK, Label.BUS, Label.BICYCLE, Label.MOTORCYCLE, Label.PEDESTRIAN]
WAYMO_LABELS = [Label.UNKNOWN, Label.VEHICLE, Label.PEDESTRIAN, Label.BICYCLE]

def parse_autoware_label(label: int) -> Label:
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

def parse_waymo_label(label: int) -> Label:
    match label:
        case label_pb2.Label.TYPE_VEHICLE:
            return Label.VEHICLE
        case label_pb2.Label.TYPE_PEDESTRIAN:
            return Label.PEDESTRIAN
        case label_pb2.Label.TYPE_CYCLIST:
            return Label.BICYCLE
        case _:
            return Label.UNKNOWN

def labels_match(label1:Label, label2:Label) -> bool:
    return bool(label1 & label2)