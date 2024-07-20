import re
import glob

from model_evaluator.interfaces.labels import Label


class KBRosbagMetaData:
    path: str
    date: str
    time: str
    distance: str
    count: str
    vru_type: str
    take: str
    bag_no: str

    def __init__(
        self,
        path: str,
        date: str,
        time: str,
        distance: str,
        count: str,
        vru_type: str,
        take: str,
        bag_no: str,
    ):
        self.path = path
        self.date = date
        self.time = time
        self.distance = distance
        self.count = count
        self.vru_type = vru_type
        self.take = take
        self.bag_no = bag_no

    def __str__(self):
        line1 = f"{self.date=} {self.time=}"
        line2 = f"{self.distance=} {self.count=} {self.vru_type}"
        line3 = f"{self.take=} {self.bag_no=}"

        return f"{line1}\n{line2}\n{line3}"

    def __repr__(self):
        return self.__str__()

    def expectations(self):
        return {Label.PEDESTRIAN: int(self.count)}


def parse(path: str):
    pattern = re.compile(
        r'.*/(?P<date>\d{4}_\d{2}_\d{2})-(?P<time>\d{2}_\d{2}_\d{2})_(?P<name>.+).mcap'
    )
    name_pattern = re.compile(
        r'(?P<distance>\d+m)_(?P<count>\d)_(?P<type>(\w|_)+)_(?P<take>\d)_(?P<bag_no>\d)'
    )

    match = pattern.match(path)

    if not match:
        return

    date = match.group('date')
    time = match.group('time')

    name_match = name_pattern.match(match.group('name'))

    if not name_match:
        return

    distance = name_match.group('distance')
    count = name_match.group('count')
    vru_type = name_match.group('type')
    take = name_match.group('take')
    bag_no = name_match.group('bag_no')

    return KBRosbagMetaData(
        path, date, time, distance, count, vru_type, take, bag_no
    )


def match_rosbags_in_path(path: str) -> list[KBRosbagMetaData]:
    paths = glob.glob(f'{path}/**', recursive=True)

    all_metadata = [parse(path) for path in paths]

    return [x for x in all_metadata if x is not None]
