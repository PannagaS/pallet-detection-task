import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import os


class ModelReadyWaiter(Node):
    def __init__(self):
        super().__init__('model_ready_waiter')
        self.flag_file = "/tmp/model_ready_flag"  # File to indicate readiness
        self.subscription = self.create_subscription(
            Bool,
            'model_ready',
            self.model_ready_callback,
            10
        )
        self.get_logger().info("Waiting for model_ready signal...")

    def model_ready_callback(self, msg):
        if msg.data:
            self.get_logger().info("Model is ready!")
            with open(self.flag_file, 'w') as f:
                f.write("ready")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ModelReadyWaiter()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
