#!/bin/bash
source /opt/ros/humble/setup.bash
# Create best.engine model from best.pt
echo "=========================================================================="
echo "                  Peer Robotics Pallet Detection Task"
echo "=========================================================================="
echo "Building a TensorRT model from best.pt"
echo "This could take a few minutes ..."

./get_tensor_rt_model.sh

echo "Configuring DDS ..."
 
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/ws/fastdds.xml
echo $FASTRTPS_DEFAULT_PROFILES_FILE
# Define the YAML file path
YAML_FILE="/home/ws/src/pallet_detection/pallet_detection/param_config.yaml"


# Prompt the user to select the topic to subscribe to
echo "To run the detection and segmentation model on test_images, please choose option (1) "
echo "To run the detection and segmentation model for the given bag file, please choose option (2)"
echo "To run the detection and segmentation model in real-time, please choose option (3)"
read -p "Enter your choice (1, 2, or 3): " choice



# Set the topic and launch file based on the user's choice
if [ "$choice" -eq 1 ]; then
    TOPIC="/input_images"
    LAUNCH_FILE="pallet_detection_launch.py"
elif [ "$choice" -eq 2 ]; then
    TOPIC="/robot1/zed2i/left/image_rect_color"
    LAUNCH_FILE="pallet_detection_launch.py"
elif [ "$choice" -eq 3 ]; then
    TOPIC="/robot1/zed2i/left/image_rect_color"
    LAUNCH_FILE="pallet_detection_launch_realtime.py"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "Would you like to save predictions as images? (1/0)"
read -p "Enter your choice (1/0): " save_choice

if [ "$save_choice" -eq 1 ]; then 
    SAVE_PREDICTIONS="True"
elif [ "$save_choice" -eq 0 ]; then 
    SAVE_PREDICTIONS="False"
else 
    echo "Invalid choice. Exiting."
    exit 1
fi


# Prompt the user to set inference parameters
echo "Would you like to specify inference arguments? (1 for yes, 0 for no)"
read -p "Enter your choice (1/0): " param_choice

if [ "$param_choice" -eq 1 ]; then
    read -p "Enter confidence (0 to 1): " CONF
    CONF=${CONF:-0.75}  # Use default value if user leaves input empty
    read -p "Enter IOU (0 to 1): " IOU
    IOU=${IOU:-0.7}  # Use default value if user leaves input empty
    read -p "Set FP16 precision (set_half) [yes/no] (default no): " IS_HALF
    if [[ "$IS_HALF" == "yes" || "$IS_HALF" == "y" ]]; then
        IS_HALF="True"
    else
        IS_HALF="False"
    fi
elif [ "$param_choice" -eq 0 ]; then
    echo "Parameters set to default values (IOU: 0.7, CONF: 0.35, set_half: False)"
    IOU="0.7"
    CONF="0.35"
    IS_HALF="False"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Update the YAML file using `yq`
if command -v yq &> /dev/null; then
    yq eval ".image_subscriber.ros__parameters.topic_to_subscribe = \"$TOPIC\"" -i "$YAML_FILE"
    yq eval ".image_subscriber.ros__parameters.confidence = $CONF" -i "$YAML_FILE"
    yq eval ".image_subscriber.ros__parameters.iou = $IOU" -i "$YAML_FILE"
    yq eval ".image_subscriber.ros__parameters.half = \"$IS_HALF\"" -i "$YAML_FILE"
    yq eval ".image_subscriber.ros__parameters.save_predictions = \"$SAVE_PREDICTIONS\"" -i "$YAML_FILE"
    echo "Updated parameters in $YAML_FILE:"
    echo " - topic_to_subscribe: $TOPIC"
    echo " - confidence: $CONF"
    echo " - iou: $IOU"
    echo " - half: $IS_HALF"
    echo " - save_predictions: $SAVE_PREDICTIONS"
else
    echo "Error: 'yq' is not installed. Please install it to use this script."
    exit 1
fi

# Source ROS 2 setup
cd /home/ws/
source /opt/ros/humble/setup.bash

# For visualization 
# rviz2 --display-config /home/ws/rviz_view_config_file.rviz &

source install/setup.bash


# Change directory to launch file scripts
cd /home/ws/src/pallet_detection/pallet_detection/launch/
echo "Launching ROS2 application with $LAUNCH_FILE ..."
ros2 launch "$LAUNCH_FILE"

