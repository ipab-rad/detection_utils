import rclpy
from rclpy.node import Node
import numpy as np
from typing import Optional
from cv_bridge import CvBridge
import threading
import queue

from sensor_msgs.msg import Image
from tier4_perception_msgs.msg import DetectedObjectsWithFeature, DetectedObjectWithFeature

from model_evaluator.inference_connector import InferenceConnector
from model_evaluator.detection import Detection, BBox, Label

class TensorrtYOLOXConnectorNode(Node):

    def __init__(self, publish_topic: str, subscription_topic: str, results_queue: queue.SimpleQueue):
        super().__init__('tensorrt_yolox_connector_node')

        self.results_queue = results_queue

        self.publisher = self.create_publisher(Image, publish_topic, 10)
        self.subscriber = self.create_subscription(DetectedObjectsWithFeature, subscription_topic, self.subscription_callback, 10)


    def subscription_callback(self, msg: DetectedObjectsWithFeature):
        self.results_queue.put(msg)


class TensorrtYOLOXConnector(InferenceConnector):

    def run_node(self):
        rclpy.spin(self.node)

        self.node.destroy_node()
        rclpy.shutdown()

    def __init__(self, input_topic: str, output_topic: str):
        self.lock = threading.Lock()
        self.results_queue = queue.SimpleQueue()

        self.bridge = CvBridge()

        rclpy.init()

        self.node = TensorrtYOLOXConnectorNode(input_topic, output_topic, self.results_queue)

        self.ros_thread = threading.Thread(target=self.run_node)
        self.ros_thread.start()
        
    def runInference(self, image: np.ndarray) -> Optional[list[Detection]]:
        with self.lock:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.node.publisher.publish(msg)
            
            try:
                result = self.results_queue.get(timeout=5)

                objects_with_feature: list[DetectedObjectWithFeature] = result.feature_objects

                res = []

                for i in range(len(objects_with_feature)):
                    bbox = BBox.from_xywh(objects_with_feature[i].feature.roi.x_offset,
                                          objects_with_feature[i].feature.roi.y_offset,
                                          objects_with_feature[i].feature.roi.width,
                                          objects_with_feature[i].feature.roi.height)
                    score = objects_with_feature[i].object.existence_probability
                    label = objects_with_feature[i].object.classification[0].label

                    detection = Detection(bbox, score, Label(label))

                    res.append(detection)

                return res

            except queue.Empty:
                print("No result")

                return None
