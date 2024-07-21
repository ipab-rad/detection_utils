import json
from typing import Generator, Optional
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
from model_evaluator.interfaces.detection2D import Detection2D
from model_evaluator.interfaces.detection3D import Detection3D, BBox3D
from model_evaluator.interfaces.labels import Label


class RosbagReader:
    def __init__(
        self,
        path: str,
        topics_with_type: dict[str, type],
        print_info: bool = False,
    ) -> None:
        self.path = path
        self.topics_with_types = topics_with_type

        # TODO: handle non mcap rosbags

        self.read_rosbag_metadata(print_info)

        # Create sequential reader for rosbag
        self.reader = rosbag2_py.SequentialReader()

        self.read_bag_into_reader()


    def read_rosbag_metadata(self, print_info:bool):
        # Read rosbag metadata
        info = rosbag2_py.Info()
        metadata = info.read_metadata(self.path, 'mcap')

        if print_info:
            print(f'Rosbag \'{self.path}\'')
            print(
                f'Size: {metadata.bag_size / 1024.0 / 1024.0:.2f} MB, Duration: {metadata.duration}, Message count: {metadata.message_count}'
            )

        # Check wanted topics exist in rosbag and types match
        matching_topics = [topic_info for topic_info in metadata.topics_with_message_count
                           if topic_info.topic_metadata.name in self.topics_with_types.keys()]

        for topic_info in matching_topics:
            self.check_ros_topic_type(topic_info, metadata, print_info)


    def check_ros_topic_type(self,topic_info, metadata, print_info):
        try:
            topic_type = get_message(topic_info.topic_metadata.type)
            expected_type = self.topics_with_types[topic_info.topic_metadata.name]

            if topic_type != expected_type:
                error_msg = f"Topic '{topic_info.topic_metadata.name}' has different type '{topic_type}' ('{topic_info.topic_metadata.type}')"
                error_msg += f" than desired type '{self.topics_with_types[topic_info.topic_metadata.name]}'."
                raise TypeError(error_msg)

            if print_info:
                info_msg = f"Topic '{topic_info.topic_metadata.name}' ('{topic_type}'): {topic_info.message_count} messages"
                info_msg += f" ({topic_info.message_count / metadata.duration.total_seconds():.2f} Hz)"
                print(info_msg)

        except ModuleNotFoundError as e:
            raise TypeError(
                f"Type '{topic_info.topic_metadata.type}' for Topic '{topic_info.topic_metadata.name}' not found: {e.msg}"
            )

    def read_bag_into_reader(self):
        self.reader.open(
            rosbag2_py.StorageOptions(uri=self.path, storage_id='mcap'),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr',
            ),
        )

        # Set filter for wanted topics only
        self.reader.set_filter(
            rosbag2_py.StorageFilter(topics=list(self.topics_with_types.keys()))
        )

    def __iter__(self) -> Generator[tuple[str, bytes, time], None, None]:
        # Reset sequential reader to first message
        self.reader.seek(0)

        return self

    def __next__(self) -> tuple[str, bytes, time]:
        if not self.reader.has_next():
            raise StopIteration

        return self.reader.read_next()


class RosbagDatasetReader2D(DatasetReader2D):

    def __init__(self, path: str, image_topic: str):
        self.path = path
        self.image_topic = image_topic

        self.cv_bridge = CvBridge()
        self.reader = RosbagReader(path, {image_topic: Image})

    def read_data(
        self,
    ) -> Generator[tuple[np.ndarray, Optional[list[Detection2D]]], None, None]:

        for _, msg, _ in self.reader:
            image_msg = deserialize_message(msg, Image)

            # Return no bounding boxes as there is no ground truth in rosbag
            yield self.cv_bridge.imgmsg_to_cv2(image_msg), None


class RosbagDatasetReader3D(DatasetReader3D):
    start_frame: int
    end_frame: int
    gt_detections_by_frame: dict[int,list[Detection3D]]

    gt_ped_bbox_dims = [0.5, 0.7, 1.9]  # fixed size for ease
    gt_ped_bbox_heading = 0  # axis aligned for ease

    def __init__(self, path: str, pointcloud_topic: str, bbox_file_name:str):
        self.path = path
        self.pointcloud_topic = pointcloud_topic

        self.reader = RosbagReader(path, {pointcloud_topic: PointCloud2})
        self.gt_detections_by_frame, self.start_frame, self.end_frame = self.read_bboxes_from_files(bbox_file_name)

    def read_bboxes_from_files(self, bbox_file_name:str) -> tuple[dict[int,list[Detection3D]], int, int]:
        bboxes_parent_dir = "/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/bbox_generator"

        ped_bboxes_file_dir = f"{bboxes_parent_dir}/scene_boxes"
        ped_bboxes_file_name = bbox_file_name

        with open(f"{ped_bboxes_file_dir}/{ped_bboxes_file_name}.json") as f:
            ped_bboxes_centers_json = json.load(f)

        start_frame:int = ped_bboxes_centers_json[0]["frame"] - 10  # 1 sec before first bbox
        end_frame:int = ped_bboxes_centers_json[-1]["frame"] + 10  # 1 sec after last bbox

        ped_bbox_tuples : list[tuple[list, int]] = [(ped_json["center"], ped_json["frame"]) for ped_json in ped_bboxes_centers_json]

        l, w, h = self.gt_ped_bbox_dims[0], self.gt_ped_bbox_dims[1], self.gt_ped_bbox_dims[2]
        yaw = self.gt_ped_bbox_heading

        ped_bboxes = [(
            BBox3D.from_oriented(pjt[0][0], pjt[0][1], pjt[0][2], l, w, h, yaw)
        , pjt[1]) for pjt in ped_bbox_tuples]
        ped_dets = [(Detection3D(bbox, 1, Label.PEDESTRIAN), frame) for bbox,frame in ped_bboxes]
        ped_dets_dict = {}

        for pd in ped_dets:
            frame = pd[1]
            det3d = pd[0]
            if frame in ped_dets_dict:
                ped_dets_dict[frame].append(det3d)
            else:
                ped_dets_dict[frame] = [det3d]

        static_vehicle_bboxes_file_dir = bboxes_parent_dir
        static_vehicle_bboxes_file_name = "static_vehicle_boxes"

        with open(f"{static_vehicle_bboxes_file_dir}/{static_vehicle_bboxes_file_name}.json") as f:
            static_vehicle_bboxes_json = json.load(f)

        static_vehicle_bboxes_with_classes = [(BBox3D.from_oriented(
            svbj["center"][0], svbj["center"][1], svbj["center"][2],
            svbj["dimensions"][0], svbj["dimensions"][1], svbj["dimensions"][2],
            svbj["heading"]
        ), svbj["class"]) for svbj in static_vehicle_bboxes_json]

        car_bboxes = [bbox_tuple[0] for bbox_tuple in static_vehicle_bboxes_with_classes if bbox_tuple[1] == 1]
        truck_bboxes = [bbox_tuple[0] for bbox_tuple in static_vehicle_bboxes_with_classes if bbox_tuple[1] == 2]

        car_dets = [Detection3D(bbox, 1, Label.CAR) for bbox in car_bboxes]
        truck_dets = [Detection3D(bbox, 1, Label.TRUCK) for bbox in truck_bboxes]

        all_dets = {}

        for frame_counter in range(start_frame, end_frame+1):
            all_dets[frame_counter] = car_dets + truck_dets  # included in every frame
            if frame_counter in ped_dets_dict:
                all_dets[frame_counter] += ped_dets_dict[frame_counter]

        return all_dets, start_frame, end_frame

    def read_data(
        self,
    ) -> Generator[
        tuple[PointCloud2, Optional[list[Detection3D]]], None, None
    ]:
        for frame_counter, (_, msg, _) in enumerate(self.reader):
            # already in appropriate format
            pointcloud_msg = deserialize_message(msg, PointCloud2)

            bounding_boxes = []

            if frame_counter in self.gt_detections_by_frame:
                bounding_boxes = self.gt_detections_by_frame[frame_counter]

            yield pointcloud_msg, bounding_boxes