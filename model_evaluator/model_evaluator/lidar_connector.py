import threading
import queue
from typing import Optional

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2
from tier4_perception_msgs.msg import (
    DetectedObjects,
    DetectedObject,
)

from model_evaluator.interfaces.inference_connector import InferenceConnector3D
from model_evaluator.interfaces.detection3D import Detection3D, BBox3D


class LiDARConnectorNode(Node):

    results_queue: queue.SimpleQueue

    def __init__(
        self, node_name: str, publish_topic: str, subscription_topic: str
    ):
        super().__init__(node_name)

        self.results_queue = queue.SimpleQueue()

        self.publisher = self.create_publisher(PointCloud2, publish_topic, 10)
        self.subscriber = self.create_subscription(
            DetectedObjects,
            subscription_topic,
            self.subscription_callback,
            10,
        )

    def subscription_callback(self, msg: DetectedObjects):
        self.results_queue.put(msg)


class LiDARConnector(InferenceConnector3D):
    def __init__(self, node_name: str, input_topic: str, output_topic: str):
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        rclpy.init()

        self.node = LiDARConnectorNode(node_name, input_topic, output_topic)

        ros_thread = threading.Thread(target=rclpy.spin, args=[self.node])
        ros_thread.start()

    def __del__(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def detected_object_to_detection3D(self, det_object: DetectedObject):
        bbox = BBox3D.from_footprint(det_object.shape.footprint)
        score = det_object.existence_probability

        return Detection3D(bbox, score)

    def run_inference(self, msg: PointCloud2) -> Optional[list[Detection3D]]:
        with self.lock:
            self.node.publisher.publish(msg)

            try:
                result = self.node.results_queue.get(timeout=1)

                all_objects: list[DetectedObject] = result.objects

                return [
                    self.detected_object_to_detection3D(det_object)
                    for det_object in all_objects
                ]

            except queue.Empty:
                # TODO: Handle correctly, like throwing error
                print("No result (timed out)")

                return None
