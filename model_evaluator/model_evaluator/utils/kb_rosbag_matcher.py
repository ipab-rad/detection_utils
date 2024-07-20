import re
import glob
from datetime import datetime


class KBRosbagMetaData:
    IMAGE_TOPIC = '/sensor/camera/fsp_l/image_rect_color'
    LIDAR_TOPIC = '/sensor/lidar/top/points'

    path: str
    timestamp: datetime
    distance: str
    count: str
    vru_type: str
    take: str

    def __init__(
        self,
        path: str,
        timestamp: datetime,
        distance: str,
        count: str,
        vru_type: str,
        take: str,
    ):
        self.path = path
        self.timestamp = timestamp
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


def parse(path: str):
    pattern = re.compile(
        r'.*/(?P<time>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})_(?P<name>.+)'
    )
    name_pattern = re.compile(
        r'(?P<distance>\d+m)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)'
    )

    match = pattern.match(path)

    if not match:
        return None

    timestamp = datetime.strptime(match.group('time'), '%Y_%m_%d-%H_%M_%S')

    name_match = name_pattern.match(match.group('name'))

    if not name_match:
        return None

    distance = name_match.group('distance')
    count = name_match.group('count')
    vru_type = name_match.group('type')
    take = name_match.group('take')

    return KBRosbagMetaData(path, timestamp, distance, count, vru_type, take)


def match_rosbags_in_path(path: str) -> list[KBRosbagMetaData]:
    paths = glob.glob(f'{path}/*/')

    all_metadata = [parse(path) for path in paths]

    return [x for x in all_metadata if x is not None]
