import re
import glob
from datetime import datetime

from model_evaluator.readers.rosbag_reader import RosbagDatasetReader2D, RosbagDatasetReader3D
from model_evaluator.interfaces.labels import Label

class KBRosbagMetaData:
    timestamp: datetime
    distance: str
    count: int
    vru_type: str
    take: int

    def __init__(
        self,
        timestamp: datetime,
        distance: str,
        count: int,
        vru_type: str,
        take: int,
    ):
        self.timestamp = timestamp
        self.distance = distance
        self.count = count
        self.vru_type = vru_type
        self.take = take

    def __str__(self):
        line1 = f"self.timestamp={self.timestamp.isoformat()}"
        line2 = f"{self.distance=} {self.count=} {self.vru_type=}"
        line3 = f"{self.take=}"

        return f"{line1}\n{line2}\n{line3}"

    def __repr__(self):
        return self.__str__()

class KBRosbag:
    IMAGE_TOPIC = '/sensor/camera/fsp_l/image_rect_color'
    LIDAR_TOPIC = '/sensor/lidar/top/points'

    metadata: KBRosbagMetaData

    def __init__(self, path:str):
        self.path = path
        self.metadata = self.parse_metadata()

    def empty(self):
        return self.metadata is None

    def get_expectations_2D(self) -> dict[Label, int]:
        # TODO: Add support for cycling rosbags
        return {Label.PEDESTRIAN: self.metadata.count}

    def get_reader_2d(self) -> RosbagDatasetReader2D:
        return RosbagDatasetReader2D(self.path, self.IMAGE_TOPIC)

    def get_reader_3d(self) -> RosbagDatasetReader3D:
        return RosbagDatasetReader3D(self.path, self.LIDAR_TOPIC, self.metadata)

    def parse_metadata(self):
        pattern = re.compile(
            r'.*/(?P<time>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})_(?P<name>.+)'
        )
        name_pattern = re.compile(
            r'(?P<distance>\d+m)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)'
        )

        match = pattern.match(self.path)

        if not match:
            return None

        timestamp = datetime.strptime(match.group('time'), '%Y_%m_%d-%H_%M_%S')

        name_match = name_pattern.match(match.group('name'))

        if not name_match:
            return None

        distance = name_match.group('distance')
        count = int(name_match.group('count'))
        vru_type = name_match.group('type')
        take = int(name_match.group('take'))

        return KBRosbagMetaData(timestamp, distance, count, vru_type, take)


def match_rosbags_in_path(path: str) -> list[KBRosbag]:
    paths = glob.glob(f'{path}/*/')

    all_metadata = [KBRosbag(path) for path in paths]

    return [x for x in all_metadata if not x.empty()]
