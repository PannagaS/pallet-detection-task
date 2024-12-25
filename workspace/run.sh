#!/bin/bash

# Get the directory of the script (assumes the script is in the workspace directory)
WORKSPACE_DIR=$(dirname "$(realpath "$0")")
echo $WORKSPACE_DIR
# Prompt the user for whether they want to run with or without RViz
echo "Do you want to run the Docker container with RViz?"
read -p "Enter 'yes' or 'no': " user_input

# Determine the appropriate start script based on user input
if [[ "$user_input" == "yes" || "$user_input" == "y" ]]; then
    START_SCRIPT="/home/ws/start_with_rviz.sh"
elif [[ "$user_input" == "no" || "$user_input" == "n" ]]; then
    START_SCRIPT="/home/ws/start.sh"
else
    echo "Invalid input. Exiting."
    exit 1
fi

# Run the Docker container with the appropriate command
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v "$WORKSPACE_DIR":/home/ws \
    peer_robotics:jetson_v5 "$START_SCRIPT"
