from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch_ros.actions import Node
import yaml

# Function to load parameters from the YAML file
def load_params_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

# Function to create and conditionally launch nodes
def generate_launch_description():
    yaml_path = "/home/ws/src/pallet_detection/pallet_detection/param_config.yaml"  
    params = load_params_from_yaml(yaml_path)

    # Get the value of 'topic_to_subscribe' from the YAML file
    topic_to_subscribe = params.get("image_subscriber", {}).get("ros__parameters", {}).get("topic_to_subscribe", None)

    # Print out messages for debugging
    print(f"Loaded parameters from YAML: {params}")
    print(f"topic_to_subscribe: {topic_to_subscribe}")

    # Base nodes
    image_subscriber_node = Node(
        package='pallet_detection',
        executable='image_subscriber',
        name='image_subscriber',
        parameters=[yaml_path]
    )

    wait_for_model_ready_node = Node(
        package="pallet_detection",
        executable="model_ready_waiter",
        name="model_ready_waiter",
        output="screen",
    )


    nodes = [
        LogInfo(msg=f"Launching image_subscriber with parameters from {yaml_path}"),
        image_subscriber_node,
        LogInfo(msg="Waiting for model to be ready..."),
        wait_for_model_ready_node
    ]

    # Add a delay before launching the image_publisher node
    if topic_to_subscribe == "/input_images":
        delayed_image_publisher_node = TimerAction(
            period=10.0,  # Delay in seconds
            actions=[
                Node(
                    package='pallet_detection',
                    executable='image_publisher',
                    name='image_publisher',
                    parameters=[yaml_path]
                )
            ]
        )

        nodes.append(LogInfo(msg="Delaying image_publisher launch by 10 seconds..."))
        nodes.append(delayed_image_publisher_node)

    # Play ROS bag if topic_to_subscribe matches a specific topic
    if topic_to_subscribe == "/robot1/zed2i/left/image_rect_color":
        delayed_rosbag_play_command = TimerAction(
            period=10.0,  # Delay in seconds
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "bag", "play", "/home/ws/internship_assignment_sample_bag/internship_assignment_sample_bag_0.db3"],
                    output="screen"
                )
            ]
        )

        nodes.append(LogInfo(msg="Delaying ROS bag play by 10 seconds..."))
        nodes.append(delayed_rosbag_play_command)
    elif topic_to_subscribe == "/input_images":
        nodes.append(LogInfo(msg="No ROS bag is played as topic_to_subscribe is '/input_images'"))

    return LaunchDescription(nodes)
