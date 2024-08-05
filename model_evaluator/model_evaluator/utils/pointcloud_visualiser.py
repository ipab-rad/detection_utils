import argparse
import os
import sys
import math
import struct
import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2, PointField

# Datatypes for PointField
_DATATYPES = {
    PointField.INT8: ('b', 1),
    PointField.UINT8: ('B', 1),
    PointField.INT16: ('h', 2),
    PointField.UINT16: ('H', 2),
    PointField.INT32: ('i', 4),
    PointField.UINT32: ('I', 4),
    PointField.FLOAT32: ('f', 4),
    PointField.FLOAT64: ('d', 8)
}


class PointCloudVisualizer:
    def __init__(self):
        pass

    def _get_struct_fmt(self, is_bigendian, fields, field_names=None):
        """Generate the struct format string for unpacking PointCloud2 data."""
        fmt = '>' if is_bigendian else '<'
        offset = 0
        for field in (f for f in sorted(fields, key=lambda f: f.offset) if
                      field_names is None or f.name in field_names):
            if offset < field.offset:
                fmt += 'x' * (field.offset - offset)
                offset = field.offset
            if field.datatype not in _DATATYPES:
                print(f'Skipping unknown PointField datatype [{field.datatype}]', file=sys.stderr)
            else:
                datatype_fmt, datatype_length = _DATATYPES[field.datatype]
                fmt += field.count * datatype_fmt
                offset += field.count * datatype_length
        return fmt

    def read_points(self, cloud, field_names=None, skip_nans=True):
        """
        Read points from a sensor_msgs.PointCloud2 message.
        """
        assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
        fmt = self._get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
        width, height, point_step, row_step, data, isnan = \
            cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
        unpack_from = struct.Struct(fmt).unpack_from

        if skip_nans:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    # to keep the colouring consistent between frames
                    p = unpack_from(data, offset)[:3]
                    p_within_bounds = 100 > p[0] > -40
                    if not any(isnan(pv) for pv in p) and p_within_bounds:
                        yield p
                    offset += point_step
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)[:3]
                    offset += point_step

    def create_basis(self, origin=[0, 0, 0]):
        """Create a 3D basis at the specified origin."""
        origin = np.array(origin)
        points = [
            origin,                    # Origin
            origin + [1, 0, 0],        # X-axis end point
            origin + [0, 1, 0],        # Y-axis end point
            origin + [0, 0, 1]         # Z-axis end point
        ]
        lines = [
            [0, 1],  # X-axis
            [0, 2],  # Y-axis
            [0, 3]   # Z-axis
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [
            [1, 0, 0],  # X-axis color
            [0, 1, 0],  # Y-axis color
            [0, 0, 1]   # Z-axis color
        ]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def create_bounding_box(self, center=[0.0, 0.0, 0.0], rotation=np.eye(3),
                            dimensions=[0.5, 0.5, 1.0], color=[0.0, 1.0, 0.0]):

        """Create an oriented bounding box at the specified location."""
        bbox = o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=dimensions)
        bbox.color = color
        return bbox

    def get_bold_bbox(self, center=[0.0, 0.0, 0.0], rotation=np.eye(3),
                      dimensions=[0.5, 0.5, 1.0], color=[0.0, 1.0, 0.0]):
        offset = np.array([0.01, 0.01, 0.01])

        center_np = np.array(center)

        main = self.create_bounding_box(
            center_np,
            rotation,
            dimensions,
            color
        )

        plus_one = self.create_bounding_box(
            center_np + offset,
            rotation,
            dimensions,
            color
        )

        plus_two = self.create_bounding_box(
            center_np + 2 * offset,
            rotation,
            dimensions,
            color
        )

        minus_one = self.create_bounding_box(
            center_np - offset,
            rotation,
            dimensions,
            color
        )

        minus_two = self.create_bounding_box(
            center_np - 2 * offset,
            rotation,
            dimensions,
            color
        )

        return [main, plus_one, plus_two, minus_one, minus_two]

    def visualize_pointcloud(self, pc_np_array, lidar_frame, point_size=3):
        """Visualize a point cloud with Open3D."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Set background to black
        #opt.point_color_option = o3d.visualization.PointColorOption.XCoordinate
        opt.point_size = point_size  # Set initial point size

        filter = lambda x: 41.75 <= x[0] <= 44.25 and -1.8 <= x[1] <= 0.9 and -1.1 <= x[2] <= 2

        # # pedestrian clusters for 40m_2_ped_same_way_0 frame 103
        # ped = [92985, 92986, 94523, 94524, 94525, 96074, 96075, 96076, 96077, 97639, 97640, 97641,
        #        97642, 99214, 99215, 99216, 99217, 100834, 100835, 100836, 102431, 102432, 102433,
        #        103982, 103983, 103985, 103986, 105516, 105517, 105519, 105520, 107023, 107024, 107025,
        #        107028, 108667, 108670, 108672]

        # ped_2 = [89901, 89902, 89903, 91452, 91453, 91454, 91455, 92987, 92988, 94526, 100837,
        #          102434, 102435, 103984, 103987, 105518, 105521, 105522, 107026, 107027, 107029,
        #          107030, 107031, 108668, 108669, 108673]

        indices_1 = [x for x in range(0, pc_np_array.shape[0])
                if filter(pc_np_array[x])]
        indices_2 = [x for x in range(0, pc_np_array.shape[0]) if not filter(pc_np_array[x])]
        indices_3 = []

        # indices_1 = ped
        # indices_2 = [x for x in range(0,pc_np_array.shape[0]) if x not in ped+ped_2]
        # indices_3 = ped_2

        # pc_np_array_1 = pc_np_array  # no splitting
        pc_np_array_1 = pc_np_array[indices_1]
        pc_np_array_2 = pc_np_array[indices_2]
        pc_np_array_3 = pc_np_array[indices_3]

        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc_np_array_1)
        o3d_pc.paint_uniform_color([1,0,0])

        o3d_pc2 = o3d.geometry.PointCloud()
        o3d_pc2.points = o3d.utility.Vector3dVector(pc_np_array_2)
        o3d_pc2.paint_uniform_color([0.4,0.4,0.4])

        o3d_pc3 = o3d.geometry.PointCloud()
        o3d_pc3.points = o3d.utility.Vector3dVector(pc_np_array_3)
        o3d_pc3.paint_uniform_color([0,0.5,1])

        # Create basis at origin
        origin_basis = self.create_basis([0.0, 0.0, 0.0])

        # Define the location where you want to place the box and create it[

        center = [13.3, 4.4,-0.65]
        rot_matrix = np.eye(3)          # Identity, i.e no rotation
        dimensions = [0.5, 0.7, 1.9]    # x (depth), y (width), z (height)

        color = [0, 1, 1]  # Pink
        bbox1_bold = self.get_bold_bbox(center, rot_matrix, dimensions, color)

        center2 = [43.02,-0.47,-0.04]
        dims2 = [0.91,0.78,1.64]
        yaw2 = 1.23
        color2 = [1, 0.5, 0]  # Orange

        bbox2_bold = self.get_bold_bbox(
            center2,
            np.array([
                [np.cos(yaw2), -np.sin(yaw2), 0],
                [np.sin(yaw2), np.cos(yaw2), 0],
                [0, 0, 1]
            ]),
            dims2,
            color2
        )

        # car_bboxes = self.car_bboxes()

        all_bboxes = bbox2_bold

        # Add geometries to visualizer
        vis.add_geometry(o3d_pc)
        vis.add_geometry(o3d_pc2)
        vis.add_geometry(o3d_pc3)
        vis.add_geometry(origin_basis)

        for bb in all_bboxes:
            vis.add_geometry(bb)

        # Set the camera view parameters
        ctr = vis.get_view_control()
        ctr.set_lookat([4.3, -0.6, -0.84])  # Look at the origin
        ctr.set_front([-0.88, 0, 0.48])  # Camera front direction
        ctr.set_up([0.73, 0.0, 0.68])  # Camera up direction
        ctr.set_zoom(0.03)  # Zoom level

        vis.run()
        vis.capture_screen_image(f"imgs/{lidar_frame}.png")
        vis.destroy_window()

    def read_rosbag(self, input_bag: str, selected_lidar_frame=0):
        """Read messages from a ROS bag and visualize a specific point cloud frame."""
        reader = rosbag2_py.SequentialReader()
        print(f'Reading mcap file ...')
        reader.open(
            rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )

        topic_types = reader.get_all_topics_and_types()

        def typename(topic_name):
            for topic_type in topic_types:
                if topic_type.name == topic_name:
                    return topic_type.type
            raise ValueError(f"topic {topic_name} not in bag")

        lidar_frame = 0
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic == '/sensor/lidar/top/points':
                if lidar_frame == selected_lidar_frame:
                    print(f'Visualising lidar frame: {selected_lidar_frame}')
                    msg_type = get_message(typename(topic))
                    msg = deserialize_message(data, msg_type)
                    pcd_as_numpy_array = np.array(list(self.read_points(msg)))
                    self.visualize_pointcloud(pcd_as_numpy_array, selected_lidar_frame)
                    break
                lidar_frame += 1
        del reader

    def car_bboxes(self):
        car_bbox_colour = [1, 1, 1]  # white

        yaw1 = 1.554
        car_bbox_1 = self.create_bounding_box(
            [12.037, 5.274, -0.521],
            np.array([
                [np.cos(yaw1), -np.sin(yaw1), 0],
                [np.sin(yaw1), np.cos(yaw1), 0],
                [0, 0, 1]
            ]),
            [4.601, 1.995, 1.569],
            car_bbox_colour
        )

        yaw2 = 1.611
        car_bbox_2 = self.create_bounding_box(
            [18.974, 4.948, -0.704],
            np.array([
                [np.cos(yaw2), -np.sin(yaw2), 0],
                [np.sin(yaw2), np.cos(yaw2), 0],
                [0, 0, 1]
            ]),
            [4.089, 1.845, 1.449],
            car_bbox_colour
        )

        yaw3 = 1.598
        car_bbox_3 = self.create_bounding_box(
            [9.462, 5.227, -0.674],
            np.array([
                [np.cos(yaw3), -np.sin(yaw3), 0],
                [np.sin(yaw3), np.cos(yaw3), 0],
                [0, 0, 1]
            ]),
            [4.456, 1.903, 1.512],
            car_bbox_colour
        )

        yaw4 = 1.459
        car_bbox_4 = self.create_bounding_box(
            [28.691, 5.238, -0.363],
            np.array([
                [np.cos(yaw4), -np.sin(yaw4), 0],
                [np.sin(yaw4), np.cos(yaw4), 0],
                [0, 0, 1]
            ]),
            [4.729, 2.032, 1.821],
            car_bbox_colour
        )

        yaw5 = 1.653
        car_bbox_5 = self.create_bounding_box(
            [6.964, 4.921, -0.709],
            np.array([
                [np.cos(yaw5), -np.sin(yaw5), 0],
                [np.sin(yaw5), np.cos(yaw5), 0],
                [0, 0, 1]
            ]),
            [4.610, 1.931, 1.499],
            car_bbox_colour
        )

        yaw6 = 1.515
        car_bbox_6 = self.create_bounding_box(
            [23.838, 5.311, -0.505],
            np.array([
                [np.cos(yaw6), -np.sin(yaw6), 0],
                [np.sin(yaw6), np.cos(yaw6), 0],
                [0, 0, 1]
            ]),
            [3.774, 1.839, 1.497],
            car_bbox_colour
        )

        yaw7 = 1.515
        car_bbox_7 = self.create_bounding_box(
            [14.964, 5.887, -0.231],
            np.array([
                [np.cos(yaw7), -np.sin(yaw7), 0],
                [np.sin(yaw7), np.cos(yaw7), 0],
                [0, 0, 1]
            ]),
            [4.425, 1.924, 1.579],
            car_bbox_colour
        )

        # actually classed as a truck
        yaw8 = 1.614
        car_bbox_8 = self.create_bounding_box(
            [35.923, 4.266, -0.354],
            np.array([
                [np.cos(yaw8), -np.sin(yaw8), 0],
                [np.sin(yaw8), np.cos(yaw8), 0],
                [0, 0, 1]
            ]),
            [5.485, 2.401, 2.101],
            car_bbox_colour
        )

        return [car_bbox_1, car_bbox_2, car_bbox_3, car_bbox_4, car_bbox_5, car_bbox_6, car_bbox_7, car_bbox_8]


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize a point cloud from a ROS bag file.")
    parser.add_argument("mcap_file", type=str, help="Path to the MCAP file.")
    parser.add_argument("selected_lidar_frame", type=int, help="Index of the lidar frame to visualize.")

    # Parse arguments
    args = parser.parse_args()

    # Extract arguments
    mcap_file = args.mcap_file
    selected_lidar_frame = args.selected_lidar_frame

    # Verify if the file exists
    if not os.path.isfile(mcap_file):
        print(f"Error: The file '{mcap_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Create visualizer instance and read messages from the ROS bag
    visualizer = PointCloudVisualizer()
    visualizer.read_rosbag(mcap_file, selected_lidar_frame)
