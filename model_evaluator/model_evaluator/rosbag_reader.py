from typing import Generator

import numpy as np
import rosbag2_py
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge

from model_evaluator.interfaces.dataset_reader import DatasetReader2D, DatasetReader3D
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D, Label2D
from model_evaluator.interfaces.detection2D import Detection3D, BBox3D

from sensor_msgs.msg import PointCloud2

from typing import Generator

class RosbagDatasetReader2D(DatasetReader2D):

    def __init__(self, path, expectations: dict[Label2D, int]):
        self.path = path

        self.expectations = expectations

    def read_data(self) -> Generator[tuple[np.ndarray, list[Detection2D]], None, None]:
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

        cv_bridge = CvBridge()

        while reader.has_next():
            topic, data, _ = reader.read_next()

            if topic == '/sensor/camera/fsp_l/image_rect_color':
                msg = deserialize_message(data, Image)

                image = cv_bridge.imgmsg_to_cv2(msg)

                gts = []

                for label in self.expectations:
                    for _ in range(self.expectations[label]):
                        gts.append(
                            Detection2D(
                                BBox2D.from_xywh(0, 0, msg.width, msg.height),
                                1.0,
                                label,
                            )
                        )

                yield image, gts



class RosbagDatasetReader3D(DatasetReader3D):
    def __init__(self, path):
        self.path = path

    def read_data(self) -> Generator[tuple[PointCloud2, list[Detection3D]], None, None]:
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

            # TODO change to LiDAR topic
            if topic == '/sensor/camera/fsp_l/image_rect_color':
                # already in appropriate format
                msg = deserialize_message(data, PointCloud2)

                # TODO look up bounding boxes from file
                bounding_boxes = None

                yield msg, bounding_boxes