import time
import warnings

import open3d as o3d
import numpy as np
from waymo_open_dataset.v2.perception.utils import lidar_utils
from waymo_open_dataset import v2
import dask.dataframe as dd
import tensorflow as tf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)


class WaymoPointcloudHandler(Node):
    def __init__(self, dataset_dir, context_name, mode):
        super().__init__('waymo_pointcloud_handler')

        # Parameters
        self.dataset_dir = dataset_dir
        self.context_name = context_name

        # Dictionaries to hold data
        self.lidar_calibration = {}
        self.lidar_scans = {}
        self.top_lidar_pointclouds_o3d = {}
        self.top_lidar_pointclouds_ros2 = {}
        self.lidar_bboxes_o3d = {}
        self.total_lidar_frames = 0

        # Constants for lidar types
        self.TOP_LIDAR = 1
        self.FRONT_LIDAR = 2
        self.LEFT_LIDAR = 3
        self.RIGHT_LIDAR = 4
        self.REAR_LIDAR = 5

        # Object type colours
        self.type_color = {
            0: [1, 0, 0],  # Red for unknown
            1: [0, 1, 1],  # Blue for vehicle
            2: [1, 0.5, 1],  # Pink for pedestrian
            3: [1, 1, 0],  # Yellow for sign
            4: [0, 1, 0],  # Green for cyclist
        }

        self.lidar_pointcloud2_idx = 0

        if mode == 'ros':

            self.load_data(load_for_ros=True)

            # Create ROS2 Publisher
            self.pointcloud_publisher = self.create_publisher(
                PointCloud2, '/sensor/lidar/top/points', 10
            )

            self.get_logger().info('Publishing pointclouds in a loop')
            # Timer for publish pointcloud at 10 hz)
            self.timer = self.create_timer(0.1, self.publish_pointcloud)

        else:
            self.load_data(load_for_ros=False)

    def create_basis(self, origin=[0, 0, 0]):
        """Create a 3D basis at the specified origin."""
        origin = np.array(origin)
        points = [
            origin,  # Origin
            origin + [1, 0, 0],  # X-axis end point
            origin + [0, 1, 0],  # Y-axis end point
            origin + [0, 0, 1],  # Z-axis end point
        ]
        lines = [[0, 1], [0, 2], [0, 3]]  # X-axis  # Y-axis  # Z-axis
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [
            [1, 0, 0],  # X-axis colour
            [0, 1, 0],  # Y-axis colour
            [0, 0, 1],  # Z-axis colour
        ]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def heading_to_rotation_matrix(self, heading):
        """
        Convert a yaw heading value (in radians) to a rotation matrix around the yaw axis (z-axis).

        :param heading: Yaw angle in radians
        :return: 3x3 rotation matrix as a NumPy array
        """
        cos_yaw = np.cos(heading)
        sin_yaw = np.sin(heading)

        yaw_rotation_matrix = np.array(
            [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]]
        )

        return yaw_rotation_matrix

    def create_bounding_box(
        self,
        center=[0.0, 0.0, 0.0],
        rotation=np.eye(3),
        dimensions=[0.5, 0.5, 1.0],
        color=[0.0, 1.0, 0.0],
    ):
        """Create an oriented bounding box at the specified location."""
        bbox = o3d.geometry.OrientedBoundingBox(
            center=center, R=rotation, extent=dimensions
        )
        bbox.color = color
        return bbox

    def visualise_pointcloud_w_bboxes(
        self, pointcloud_o3d, objects_o3d, point_size=1.8
    ):
        """Visualise a point cloud with Open3D."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Set visualisation options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Set background to black
        opt.point_color_option = o3d.visualization.PointColorOption.XCoordinate
        opt.point_size = point_size  # Set initial point size

        # Create basis at origin
        origin_basis = self.create_basis([0.0, 0.0, 0.0])

        # Add geometries to visualiser
        vis.add_geometry(pointcloud_o3d)
        vis.add_geometry(origin_basis)
        for obj in objects_o3d:
            vis.add_geometry(obj)

        # Set the camera view parameters
        ctr = vis.get_view_control()
        ctr.set_lookat([4.3, -0.6, -0.84])  # Look at the origin
        ctr.set_front([-0.89, 0.05, 0.45])  # Camera front direction
        ctr.set_up([0.45, 0.0, 0.89])  # Camera up direction
        ctr.set_zoom(0.06)  # Zoom level

        vis.run()
        vis.destroy_window()

    def update_visualisation(self, vis, point_cloud, objects):
        """
        Update the visualiser with the new set of objects.

        :param vis: Open3D visualiser
        :param point_cloud: Open3D point cloud object
        :param objects: List of Open3D geometry objects to update
        """
        vis.clear_geometries()

        # Add new pointcloud
        vis.add_geometry(point_cloud)

        # Create and add base at origin
        origin_basis = self.create_basis([0.0, 0.0, 0.0])
        vis.add_geometry(origin_basis)

        # Add objects bounding boxes
        for obj in objects:
            vis.add_geometry(obj)

        # Set the camera view parameters after adding geometries
        ctr = vis.get_view_control()
        ctr.set_lookat([4.3, -0.6, -0.84])  # Look at the origin
        ctr.set_front([-0.89, 0.05, 0.45])  # Camera front direction
        ctr.set_up([0.45, 0.0, 0.89])  # Camera up direction
        ctr.set_zoom(0.06)  # Zoom level

        vis.poll_events()
        vis.update_renderer()

    def read(self, tag: str) -> dd.DataFrame:
        """Create a Dask DataFrame for the component specified by its tag."""
        paths = tf.io.gfile.glob(
            f'{self.dataset_dir}/{tag}/{self.context_name}.parquet'
        )
        return dd.read_parquet(paths)

    def convert_waymo_lidar_to_pointcloud(
        self, lidar_component, lidar_calibration_component
    ):
        """Extract point clouds from LiDAR components."""
        first_return_points = lidar_utils.convert_range_image_to_point_cloud(
            lidar_component.range_image_return1, lidar_calibration_component
        ).numpy()
        second_return_points = lidar_utils.convert_range_image_to_point_cloud(
            lidar_component.range_image_return2, lidar_calibration_component
        ).numpy()

        return np.concatenate(
            (first_return_points, second_return_points), axis=0
        )

    def convert_to_od3_pointcloud(self, lidar_points):
        """Convert points array into Open3D point cloud."""
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(lidar_points)

        return o3d_pointcloud

    def convert_to_ros_pointcloud2(
        self, lidar_points, ros_frame_id="lidar_ouster_top"
    ):
        """
        Convert a numpy array of shape [n, 3] to a ROS2 PointCloud2 message.

        :param lidar_points: numpy array of shape [n, 3] where each row is a point (x, y, z)
        :param ros_frame_id: Frame ID to use in the PointCloud2 header
        :return: PointCloud2 message
        """
        # Create a header
        header = Header()
        header.frame_id = ros_frame_id
        header.stamp = self.get_clock().now().to_msg()

        # Create a PointCloud2 message
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, lidar_points)

        return point_cloud_msg

    def load_data(self, load_for_ros=True):
        """Load and prepare lidar and calibration data."""
        self.get_logger().info('Loading lidar data...')
        lidar_df = self.read('lidar')
        lidar_calibration_df = self.read('lidar_calibration')
        lidar_box_df = self.read('lidar_box')

        rows_n = lidar_df.shape[0].compute()
        self.total_lidar_frames = rows_n / 5
        self.get_logger().info(
            f'Total top lidar frames: {self.total_lidar_frames}'
        )

        # Get laser calibration
        for _, row in lidar_calibration_df.iterrows():
            lidar_calibration_component = (
                v2.LiDARCalibrationComponent.from_dict(row)
            )
            self.lidar_calibration[
                lidar_calibration_component.key.laser_name
            ] = lidar_calibration_component

        # Merge lidar and lidar box component to iterate simultaneously into their data table
        lidar_w_box_df = v2.merge(lidar_df, lidar_box_df, right_group=True)
        # Keep track of scene sequence idx
        lidar_frame_seq_idx = 0
        self.get_logger().info('Extracting lidar points ...')

        for i, (_, row) in enumerate(lidar_w_box_df.iterrows()):
            lidar_component = v2.LiDARComponent.from_dict(row)

            # Only get frames from Top Lidar
            if lidar_component.key.laser_name == self.TOP_LIDAR:
                # Convert lidar frame into cartesian points array
                lidar_points = self.convert_waymo_lidar_to_pointcloud(
                    lidar_component, self.lidar_calibration[self.TOP_LIDAR]
                )

                if load_for_ros:
                    self.top_lidar_pointclouds_ros2[lidar_frame_seq_idx] = (
                        self.convert_to_ros_pointcloud2(lidar_points)
                    )
                else:
                    self.top_lidar_pointclouds_o3d[lidar_frame_seq_idx] = (
                        self.convert_to_od3_pointcloud(lidar_points)
                    )

                lidar_box_component = v2.LiDARBoxComponent.from_dict(row)

                frame_bboxes = []
                for j, (
                    object_id,
                    x,
                    y,
                    z,
                    dx,
                    dy,
                    dz,
                    heading,
                    type_id,
                ) in enumerate(
                    zip(
                        lidar_box_component.key.laser_object_id,
                        lidar_box_component.box.center.x,
                        lidar_box_component.box.center.y,
                        lidar_box_component.box.center.z,
                        lidar_box_component.box.size.x,
                        lidar_box_component.box.size.y,
                        lidar_box_component.box.size.z,
                        lidar_box_component.box.heading,
                        lidar_box_component.type,
                    )
                ):
                    frame_bboxes.append(
                        self.create_bounding_box(
                            [x, y, z],
                            self.heading_to_rotation_matrix(heading),
                            [dx, dy, dz],
                            self.type_color[type_id],
                        )
                    )

                self.lidar_bboxes_o3d[lidar_frame_seq_idx] = frame_bboxes
                lidar_frame_seq_idx += 1

    def publish_pointcloud(self):
        """Publish point clouds periodically."""
        self.top_lidar_pointclouds_ros2[
            self.lidar_pointcloud2_idx
        ].header.stamp = (self.get_clock().now().to_msg())
        self.pointcloud_publisher.publish(
            self.top_lidar_pointclouds_ros2[self.lidar_pointcloud2_idx]
        )

        if self.lidar_pointcloud2_idx < self.total_lidar_frames - 1:
            self.lidar_pointcloud2_idx += 1
        else:
            self.lidar_pointcloud2_idx = 0

    def visualise_all_lidar_pointclouds(self):
        """Visualise all the context point clouds."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Set visualisation options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Set background to black
        opt.point_color_option = o3d.visualization.PointColorOption.Color
        opt.point_size = 1.8  # Set initial point size

        idx = 0
        try:
            while idx < self.total_lidar_frames:
                # Update the visualiser with the next pointcloud and objects
                self.update_visualisation(
                    vis,
                    self.top_lidar_pointclouds_o3d[idx],
                    self.lidar_bboxes_o3d[idx],
                )

                idx += 1
                # Sleep for 100 milliseconds
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        vis.destroy_window()

    def visualise_lidar_frame(self, selected_lidar_seq_id):
        """Visualise a selected point cloud."""
        # Visualise point cloud with ground truth bounding boxes
        center = [-28.99, 5.31, 0.97]
        dims = [5.32, 2.24, 2.37]
        heading = 0.86

        extra_box = self.create_bounding_box(
            center,
            self.heading_to_rotation_matrix(heading),
            dims,
            [0, 1, 0],
        )

        self.visualise_pointcloud_w_bboxes(
            self.top_lidar_pointclouds_o3d[selected_lidar_seq_id],
            self.lidar_bboxes_o3d[selected_lidar_seq_id] + [extra_box],
        )


def main(args=None):
    """Initialise ROS2 node."""
    rclpy.init(args=args)

    # ** Change dataset_dir according to host waymo dataset location **

    dataset_dir = '/opt/ros_ws/rosbags/waymo/validation'
    context_name = '1024360143612057520_3580_000_3600_000'

    # dataset_dir = '/mnt/mydrive/waymo_od/validation'
    # context_name = '10359308928573410754_720_000_740_000'

    # dataset_dir = '/mnt/mydrive/waymo_od/testing' # Testing throw error :/
    # context_name = '17136775999940024630_4860_000_4880_000'

    # Mode 1: ROS
    # handler = WaymoPointcloudHandler(dataset_dir, context_name, mode = 'ros')
    # rclpy.spin(handler)

    # Mode 2: Visualliser
    handler = WaymoPointcloudHandler(
        dataset_dir, context_name, mode='visualiser'
    )
    # handler.visualise_all_lidar_pointclouds()
    handler.visualise_lidar_frame(22)

    # Shutdown ROS2
    handler.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
