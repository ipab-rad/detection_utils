import numpy as np
import rosbag2_py
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge

from model_evaluator.interfaces.dataset_reader import DatasetReader2D
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D, Label2D


class RosbagDatasetReader2D(DatasetReader2D):

    def __init__(self, path):
        self.path = path

    def read_data(self) -> list[tuple[np.ndarray, list[Detection2D]]]:
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

        msgs = []

        cv_bridge = CvBridge()

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == '/sensor/camera/fsp_l/image_rect_color':
                msg = deserialize_message(data, Image)

                msgs.append(
                    (
                        cv_bridge.imgmsg_to_cv2(msg),
                        [
                            Detection2D(
                                BBox2D.from_xywh(0, 0, msg.width, msg.height),
                                1.0,
                                Label2D.PEDESTRIAN,
                            )
                        ],
                    )
                )

        return msgs

        return None
