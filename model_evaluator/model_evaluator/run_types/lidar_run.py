from model_evaluator.connectors.lidar_connector import LiDARConnector
from model_evaluator.interfaces.detection3D import Detection3D
from model_evaluator.utils.kb_rosbag_matcher import match_rosbags_in_path


def process_rosbags_3D(connector:LiDARConnector):
    rosbags = match_rosbags_in_path('/opt/ros_ws/rosbags/kings_buildings_data')

    rosbags = [x for x in rosbags
               if x.metadata.distance == "10m" and x.metadata.vru_type == "ped"  and
               x.metadata.take == "0" and x.metadata.count == "1"]

    print(rosbags[0].metadata)

    rosbag_reader = rosbags[0].get_reader_3d()

    rosbag_data = rosbag_reader.read_data()

    for frame_counter, (point_cloud, _) in enumerate(rosbag_data):
        if frame_counter == 230:
            detections = connector.run_inference(point_cloud)

            print([x.bbox for x in detections])


def filter_detections_kb(detections: list[Detection3D]):
    # filter detections to only the experiment area
    # x greater than 2
    # y between -8 and 7
    return [det for det in detections
            if 2 < det.bbox.center_x
            and -8 < det.bbox.center_y < 7
            ]


def lidar_run():
    connector = LiDARConnector(
        'lidar_centerpoint',
        '/sensor/lidar/top/points',
        '/perception/object_recognition/detection/centerpoint/objects',
    )

    process_rosbags_3D(connector)