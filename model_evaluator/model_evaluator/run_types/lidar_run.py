
from model_evaluator.connectors.lidar_connector import LiDARConnector
from model_evaluator.utils.kb_rosbag_matcher import match_rosbags_in_path


def process_rosbags_3D(connector:LiDARConnector):
    rosbags = match_rosbags_in_path('/opt/ros_ws/rosbags/kings_buildings_data')

    rosbags = [x for x in rosbags if x.distance == "10m" and x.vru_type == "ped" and x.take == "0" and x.count == "1"]

    print(rosbags[0])

    rosbag_reader = rosbags[0].get_reader_3d()

    rosbag_data = rosbag_reader.read_data()

    for frame_counter, (point_cloud, _) in enumerate(rosbag_data):
        if frame_counter == 230:
            detections = connector.run_inference(point_cloud)

            print(detections)

def lidar_run():
    connector = LiDARConnector(
        'lidar_centerpoint',
        '/sensor/lidar/top/points',
        '/perception/object_recognition/detection/centerpoint/objects',
    )

    process_rosbags_3D(connector)