import re
import os
import glob
from datetime import datetime
from typing import Optional

from model_evaluator.readers.rosbag_reader import RosbagDatasetReader3D
from model_evaluator.readers.kb_rosbag_reader import KBRosbagReader2D

class KBRosbagMetaData:
    timestamp: datetime
    name: str
    distance: Optional[int]
    count: int
    vru_type: str
    take: int

    def __init__(
        self,
        timestamp: datetime,
        name: str,
        distance: Optional[int],
        count: int,
        vru_type: str,
        take: int,
    ):
        self.timestamp = timestamp
        self.name = name
        self.distance = distance
        self.count = count
        self.vru_type = vru_type
        self.take = take

    def __str__(self):
        line1 = f"{self.timestamp.isoformat()}"
        line2 = f"{self.distance=} {self.count=} {self.vru_type=}"
        line3 = f"{self.take=}"

        return f"{line1}\n{line2}\n{line3}"

    def __repr__(self):
        return self.__str__()

class KBRosbag:
    LIDAR_TOPIC = '/sensor/lidar/top/points'

    metadata: KBRosbagMetaData

    def __init__(self, path:str):
        self.path = path
        self.metadata = self._parse_metadata(path)

    def get_reader_2d(self) -> KBRosbagReader2D:
        keyframes_file = os.path.join(self.path, 'keyframes.json')
        if not os.path.exists(keyframes_file):
            keyframes_file = None

        return KBRosbagReader2D(self.path, keyframes_file)

    def get_reader_3d(self) -> RosbagDatasetReader3D:
        return RosbagDatasetReader3D(self.path, self.LIDAR_TOPIC)

    @staticmethod
    def _parse_metadata(path: str) -> KBRosbagMetaData:
        pattern = re.compile(
            r'.*/(?P<time>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})_(?P<name>[^/]+)/?'
        )
        name_pattern = re.compile(
            r'(?P<distance>.*)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)'
        )
        distance_pattern = re.compile(r'(\d+)m')

        match = pattern.match(path)

        if not match:
            raise ValueError(f"Could not parse the path of the rosbag ('{path}').")

        timestamp = datetime.strptime(match.group('time'), '%Y_%m_%d-%H_%M_%S')
        name = match.group('name')

        name_match = name_pattern.match(name)

        if not name_match:
            raise ValueError(f"Could not parse the name of the rosbag ('{name}').")

        distance_match = distance_pattern.match(name_match.group('distance'))

        if distance_match:
            distance = int(distance_match.group(1))
        else:
            distance = None

        count = int(name_match.group('count'))
        vru_type = name_match.group('type')
        take = int(name_match.group('take'))

        return KBRosbagMetaData(timestamp, name, distance, count, vru_type, take)


def match_rosbags_in_path(path: str) -> list[KBRosbag]:
    paths = glob.glob(f'{path}/*/')

    rosbags = []
    for path in paths:
        try:
            rosbags.append(KBRosbag(path))
        except ValueError as e:
            print(e)
            continue

    return rosbags
