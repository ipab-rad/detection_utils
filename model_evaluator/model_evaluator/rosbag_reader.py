from typing import Generator
from datetime import time

import numpy as np
import rosbag2_py
from sensor_msgs.msg import Image, PointCloud2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

from model_evaluator.interfaces.dataset_reader import (
    DatasetReader2D,
    DatasetReader3D,
)
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D
from model_evaluator.interfaces.detection3D import Detection3D, BBox3D
from model_evaluator.interfaces.labels import Label


class RosbagReader():
    def __init__(self, path: str, topics_with_type: dict[str, type], print_info: bool = False) -> None:
        self.path = path
        self.topics_with_types = topics_with_type

        # TODO: handle non mcap rosbags

        # Read rosbag metadata
        info = rosbag2_py.Info()
        metadata = info.read_metadata(path, 'mcap')

        if print_info:
            print(f'Rosbag \'{path}\'')
            print(f'Size: {metadata.bag_size / 1024.0 / 1024.0:.2f} MB, Duration: {metadata.duration}, Message count: {metadata.message_count}')

        # Check wanted topics exist in rosbag and types match
        for topic_info in metadata.topics_with_message_count:
            if topic_info.topic_metadata.name in self.topics_with_types.keys():
                try:
                    topic_type = get_message(topic_info.topic_metadata.type)

                    if topic_type != self.topics_with_types[topic_info.topic_metadata.name]:
                        raise TypeError(f'Topic \'{topic_info.topic_metadata.name}\' has different type \'{topic_type}\' (\'{topic_info.topic_metadata.type}\') than desired type \'{self.topics_with_types[topic_info.topic_metadata.name]}\'.')

                    if print_info:
                        print(f'Topic \'{topic_info.topic_metadata.name}\' (\'{topic_type}\'): {topic_info.message_count} messages ({topic_info.message_count / metadata.duration.total_seconds():.2f} Hz)')
                except ModuleNotFoundError as e:
                    raise TypeError(f'Type \'{topic_info.topic_metadata.type}\' for Topic \'{topic_info.topic_metadata.name}\' not found: {e.msg}')

        # Create sequential reader for rosbag
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(
            rosbag2_py.StorageOptions(uri=self.path, storage_id='mcap'),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr',
            ),
        )

        # Set filter for wanted topics only
        self.reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics_with_type.keys())))

    def __iter__(self) -> Generator[tuple[str, bytes, time], None, None]:
        # Reset sequential reader to first message
        self.reader.seek(0)

        return self

    def __next__(self) -> tuple[str, bytes, time]:
        if not self.reader.has_next():
            raise StopIteration
        
        return self.reader.read_next()


class RosbagDatasetReader2D(DatasetReader2D):

    def __init__(self, path: str, image_topic: str, expectations: dict[Label, int]):
        self.path = path
        self.image_topic = image_topic
        self.expectations = expectations
        self.gts = {}
        self.cv_bridge = CvBridge()
        self.reader = RosbagReader(path, {image_topic: Image})

    def generate_gts(self, image_size: tuple[int, int]) -> list[Detection2D]:
        gts = []
        for label in self.expectations:
            for _ in range(self.expectations[label]):
                gts.append(
                    Detection2D(
                        BBox2D.from_xywh(0, 0, image_size[0], image_size[1]),
                        1.0,
                        label,
                    )
                )

        return gts
    
    def get_gts(self, image_size: tuple[int, int]):
        gts = self.gts.get(image_size)
        if gts is None:
            gts = self.generate_gts(image_size)
            self.gts[image_size] = gts

        return gts

    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, list[Detection2D]], None, None]:

        for _, msg, _ in self.reader:
            image_msg = deserialize_message(msg, Image)

            yield self.cv_bridge.imgmsg_to_cv2(image_msg), self.get_gts((image_msg.width, image_msg.height))


class RosbagDatasetReader3D(DatasetReader3D):
    def __init__(self, path):
        self.path = path

    def read_data(
        self,
    ) -> Generator[tuple[PointCloud2, list[Detection3D]], None, None]:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.path, storage_id='mcap'),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr',
            ),
        )

        # topic_types = reader.get_all_topics_and_types()

        # TODO: Add asserts

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == '/sensor/lidar/top/points':
                # already in appropriate format
                msg = deserialize_message(data, PointCloud2)

                # TODO look up bounding boxes from file
                bounding_boxes = [BBox3D()]

                yield msg, bounding_boxes
