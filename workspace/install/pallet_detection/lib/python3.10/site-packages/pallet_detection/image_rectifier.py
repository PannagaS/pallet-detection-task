import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageRectifierNode(Node):
    def __init__(self):
        super().__init__('image_rectifier')
        
        # Parameters for topics
        # self.declare_parameter('camera_info_topic', '/camera_info')
        # self.declare_parameter('raw_image_topic', '/zed1/camera')
        # self.declare_parameter('rectified_image_topic', '/input_images')

        # # Get the topics
        # self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        # self.raw_image_topic = self.get_parameter('raw_image_topic').get_parameter_value().string_value
        # self.rectified_image_topic = self.get_parameter('rectified_image_topic').get_parameter_value().string_value
        self.camera_info_topic = "/robot1/zed2i/left/camera_info"
        self.raw_image_topic = "/robot1/zed2i/left/image_color" #assuming this is NOT rectified
        # Initialize variables
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None
        self.image_size = None

        # CvBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.raw_image_topic, self.image_callback, 10)

        # Publisher
        self.publisher_ = self.create_publisher(Image, self.rectified_image_topic, 10)

        self.get_logger().info(f"Subscribed to {self.camera_info_topic} and {self.raw_image_topic}")
        self.get_logger().info(f"Publishing rectified images to {self.rectified_image_topic}")

    def camera_info_callback(self, msg):
        """Callback for camera info to get intrinsic parameters."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.image_size = (msg.width, msg.height)

            # Prepare undistortion maps
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.camera_matrix,
                self.image_size,
                cv2.CV_16SC2
            )
            self.get_logger().info("Camera info received. Rectification maps prepared.")

    def image_callback(self, msg):
        """Callback for raw images to rectify and republish."""
        if self.map1 is None or self.map2 is None:
            self.get_logger().warn("Camera info not received yet. Cannot rectify images.")
            return

        try:
            # Convert ROS Image to OpenCV format
            raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Rectify the image
            rectified_image = cv2.remap(raw_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

            # Convert rectified OpenCV image back to ROS Image
            rectified_image_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding="bgr8")

            # Publish the rectified image
            self.publisher_.publish(rectified_image_msg)
            self.get_logger().info("Published rectified image.")

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageRectifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
