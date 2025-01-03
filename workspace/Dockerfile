#
# Dockerfile for building ROS 2 Humble on a Jetson-compatible Ultralytics image
#

# For jetson
FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Non-interactive environment variable
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

# Download the ROS GPG key into /usr/share/keyrings
RUN mkdir -p /usr/share/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the ROS 2 repository (using the signed key)
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
  | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble Desktop  
RUN apt-get update && \
    apt-get install -y ros-humble-desktop

# Install colcon for building ROS packages
RUN apt-get install -y python3-colcon-common-extensions

# Install rosdep and initialize
RUN apt-get install -y python3-rosdep && \
    rosdep init && \
    rosdep update

# Downgrade setuptools (setuptools version 70.0.0 gave me no errors during colcon build)
RUN pip install setuptools==70.0.0
RUN pip install numpy==1.23.5


# # (Optional) Source ROS automatically
# SHELL ["/bin/bash", "-c"]
# RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# # Default entrypoint to Bash
# ENTRYPOINT ["/bin/bash"]

WORKDIR /home/ws

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_arm64 -O /usr/local/bin/yq
RUN chmod +x /usr/local/bin/yq



COPY get_tensor_rt_model.sh /home/ws/get_tensor_rt_model.sh
RUN chmod +x /home/ws/get_tensor_rt_model.sh 

COPY start.sh /home/ws/start.sh
RUN chmod +x /home/ws/start.sh 

COPY start_with_rviz.sh /home/ws/start_with_rviz.sh
RUN chmod +x /home/ws/start_with_rviz.sh 

 
# Default command to source ROS and run the shell script
# CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && ./get_tensor_rt_model.sh"]\
ENTRYPOINT ["/bin/bash", "-c"] 
