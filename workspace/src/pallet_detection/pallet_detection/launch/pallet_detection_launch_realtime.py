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

    # wait_for_model_ready_node = Node(
    #     package="pallet_detection",
    #     executable="model_ready_waiter",
    #     name="model_ready_waiter",
    #     output="screen",
    # )


    print("Staarting image_subscriber node ...")


    

    return LaunchDescription(nodes)
