#!/usr/bin/env python

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
        for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
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
                    p = unpack_from(data, offset)[:3]
                    if not any(isnan(pv) for pv in p):
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

    def visualize_pointcloud(self, pc_np_array, point_size=1):
        """Visualize a point cloud with Open3D."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Set background to black
        opt.point_color_option = o3d.visualization.PointColorOption.XCoordinate
        opt.point_size = point_size  # Set initial point size

        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc_np_array)

        # Create basis at origin
        origin_basis = self.create_basis([0.0, 0.0, 0.0])

        # Define the location where you want to place the box and create it
        center = [11.5, 3.0, -1.0]
        rot_matrix = np.eye(3)          # Identitty, i.e no rotation
        dimensions = [0.5, 0.5, 1.8]    # W, D, H
        color = [1, 0.5, 1]             # Pink
        bbox = self.create_bounding_box(center,rot_matrix, dimensions, color)

        # Add geometries to visualizer
        vis.add_geometry(o3d_pc)
        vis.add_geometry(bbox)
        vis.add_geometry(origin_basis)

        # Set the camera view parameters
        ctr = vis.get_view_control()
        ctr.set_lookat([4.3, -0.6, -0.84])  # Look at the origin
        ctr.set_front([-0.83, 0.02, 0.55])  # Camera front direction
        ctr.set_up([0.55, 0.0, 0.84])       # Camera up direction
        ctr.set_zoom(0.02)                  # Zoom level

        vis.run()
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
                    self.visualize_pointcloud(pcd_as_numpy_array)
                    break
                lidar_frame += 1
        del reader


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
