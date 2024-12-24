import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
# from IPython import embed
class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        self.declare_parameter('test_images', '/home/ws/Pallet-Detection-1/test/images')

        test_images = self.get_parameter('test_images').get_parameter_value().string_value
        self.publisher_ = self.create_publisher(Image, '/input_images', 50)
        self.timer = self.create_timer(2.0, self.publish_image)   
        self.image_files = iter(sorted(os.listdir(test_images)))
        self.bridge = CvBridge()
        self.image_folder = test_images

    def publish_image(self):
        try:
            image_file = next(self.image_files)
            
            img = cv2.imread(os.path.join(self.image_folder, image_file))
            # embed()
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published image: {image_file}")
        except StopIteration:
            self.get_logger().info("No more images to publish.")

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()  
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
