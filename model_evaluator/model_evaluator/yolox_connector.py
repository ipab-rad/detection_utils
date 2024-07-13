import threading
import queue
from typing import Optional

import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from tier4_perception_msgs.msg import (
    DetectedObjectsWithFeature,
    DetectedObjectWithFeature,
)

from model_evaluator.inference_connector import InferenceConnector2D
from model_evaluator.detection import Detection2D, BBox2D, Label2D


class TensorrtYOLOXConnectorNode(Node):

    results_queue: queue.SimpleQueue

    def __init__(self, publish_topic: str, subscription_topic: str):
        super().__init__('tensorrt_yolox_connector_node')

        self.results_queue = queue.SimpleQueue()

        self.publisher = self.create_publisher(Image, publish_topic, 10)
        self.subscriber = self.create_subscription(
            DetectedObjectsWithFeature,
            subscription_topic,
            self.subscription_callback,
            10,
        )

    def subscription_callback(self, msg: DetectedObjectsWithFeature):
        self.results_queue.put(msg)


class TensorrtYOLOXConnector(InferenceConnector2D):

    def __init__(self, input_topic: str, output_topic: str):
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        rclpy.init()

        self.node = TensorrtYOLOXConnectorNode(input_topic, output_topic)

        ros_thread = threading.Thread(target=rclpy.spin, args=[self.node])
        ros_thread.start()

    def __del__(self):
        self.node.destroy_node()
        rclpy.shutdown()

    @staticmethod
    def parse_yolox_label(label: int) -> Label2D:
        match (label):
            case 0:
                return Label2D.UNKNOWN
            case 1:
                return Label2D.CAR
            case 2:
                return Label2D.TRUCK
            case 3:
                return Label2D.BUS
            case 4:
                return Label2D.BICYCLE
            case 5:
                return Label2D.MOTORBIKE
            case 6:
                return Label2D.PEDESTRIAN
            case 7:
                return Label2D.ANIMAL

    def run_inference(self, data: np.ndarray) -> Optional[list[Detection2D]]:
        with self.lock:
            msg = self.bridge.cv2_to_imgmsg(data, "bgr8")
            self.node.publisher.publish(msg)

            try:
                result = self.node.results_queue.get(timeout=1)

                objects_with_feature: list[DetectedObjectWithFeature] = (
                    result.feature_objects
                )

                res = []

                for i in range(len(objects_with_feature)):
                    bbox = BBox2D.from_xywh(
                        objects_with_feature[i].feature.roi.x_offset,
                        objects_with_feature[i].feature.roi.y_offset,
                        objects_with_feature[i].feature.roi.width,
                        objects_with_feature[i].feature.roi.height,
                    )
                    score = objects_with_feature[
                        i
                    ].object.existence_probability
                    label = (
                        objects_with_feature[i].object.classification[0].label
                    )

                    detection = Detection2D(
                        bbox, score, self.parse_yolox_label(label)
                    )

                    res.append(detection)

                return res

            except queue.Empty:
                print("No result")

                return None
