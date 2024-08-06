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

from model_evaluator.interfaces.inference_connector import InferenceConnector2D
from model_evaluator.interfaces.detection2D import Detection2D, BBox2D
from model_evaluator.interfaces.labels import parse_label


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

        ros_thread = threading.Thread(target=rclpy.spin, args=[self], daemon=True)
        ros_thread.start()

    def __del__(self):
        self.destroy_node()

    def subscription_callback(self, msg: DetectedObjectsWithFeature):
        self.results_queue.put(msg)


class TensorrtYOLOXConnector(InferenceConnector2D):

    def __init__(self, input_topic: str, output_topic: str):
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        rclpy.init()

        self.node = TensorrtYOLOXConnectorNode(input_topic, output_topic)

    def __del__(self):
        rclpy.try_shutdown()

    def detected_object_with_feature_to_detection2D(
        self, object_wf: DetectedObjectWithFeature
    ):
        bbox = BBox2D.from_xywh(
            object_wf.feature.roi.x_offset,
            object_wf.feature.roi.y_offset,
            object_wf.feature.roi.width,
            object_wf.feature.roi.height,
        )
        score = object_wf.object.existence_probability
        label = object_wf.object.classification[0].label

        return Detection2D(bbox, score, parse_label(label))

    def run_inference(self, data: np.ndarray) -> Optional[list[Detection2D]]:
        with self.lock:
            msg = self.bridge.cv2_to_imgmsg(data, "bgr8")
            self.node.publisher.publish(msg)

            try:
                result = self.node.results_queue.get(timeout=1)

                all_objects_with_feature: list[DetectedObjectWithFeature] = (
                    result.feature_objects
                )

                all_detections = []

                for object_wf in all_objects_with_feature:
                    all_detections.append(
                        self.detected_object_with_feature_to_detection2D(
                            object_wf
                        )
                    )

                return all_detections

            except queue.Empty:
                # TODO: Handle correctly, like throwing error
                print("No result (timed out)")

                return None
