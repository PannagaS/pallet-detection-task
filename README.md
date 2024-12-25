# Peer Robotics Pallet Detection Task
## Task description 
### Task 1
* Given a dataset of pallets, annotate **pallets** and **ground** using any annotation tools. 
* Split the dataset into training, validation, and testing.
* Include data augmentation techniques to realize real-world scenarios. 

### Task 2
* Implement an object detection model (e.g., YOLOv11) to identify pallets.
* Develop a semantic segmentation model to segment pallets and ground. 
* Train and fine-tune the models on the dataset created in task 1. 
* Assess performance of model using mAP for detection, and IoU for segmentation. 

### Task 3
* Develop a ROS2 (humble) package with appropriate nodes. The nodes should do the following: 
    * Subscribe to image and depth topics from a simulated or real camera
    * Perform object detection and segmentation 

### Additional Tasks 
* Optimize models for edge deployments - convert to TensorRT or ONNX format. 
* Apply optimization techniques like quantization and pruning to enhance performance
* Dockerize the complete pipeline so it can natively run on different devices as long as NVIDIA drivers are present. 

## Dataset
[Click here](https://drive.google.com/drive/folders/1xSqKa55QrNGufLRQZAbp0KFGYr9ecqgT) to download the original dataset.\
[Click here](https://drive.google.com/drive/folders/1QyhZSldxGswyWTF8BNhKs0QMDdTOQqbL?usp=sharing) to download full annotated dataset.

The original dataset was resized to 640x640, and augmented dataset consists of images that were generated 3 per training example with **brightness** set between -15% to 15%, and **exposure** between -10% to 10%.

Total number of images (without data augmentation): 519\
Total number of images (including data augmentation): 1089  
| Split   |  # Images    | % of total dataset    |
| :-----: | :---: | :---: |
| Train| 1089   | 87  |
|Validation| 104    |  8  |
|Test      |  52    |  4  |

full dataset includes 1245 images
<!-- ![train-val-test split](assets\train-val-test-split.png) -->
## Process flow
The process flow inside the docker is as shown in the figure below. 
![assets](https://github.com/PannagaS/pallet-detection-task/blob/main/assets/process_flow.png) 

- The entry point for the project is at run.sh shell script.
- Once the docker image is up and running, the user is prompted with options for converting best.pt to best.engine (TensorRT model).
- Further, depending on whether the user wants to run the inference on test images or from the ros bag ([click here](https://drive.google.com/file/d/1BvhP653G3PqfUq96L18gDBIi-5oOYqcr/view) for bag file) or on live camera feed, other model parameters such as confidence, iou, half precision flag (FP16) can be set accordingly. 

Note that you will be prompted for however you want would like to run the model (by specifying model parameters and/or running inference on specific topics). I believe these prompts are self-explanatory. 

## How to run this project ?
Note that this project was developed using Jetson Orin Nano Super. Everything is configured to match arm64 architecture. 
This project is wrapped in a docker for easy replication on other devices with NVIDIA drivers. 
Please pull the docker image from [here](https://hub.docker.com/repository/docker/pans06/peer_robotics/general). 
```
docker pull pans06/peer_robotics:jetson_v5
```
Clone this repo, and navigate to `workspace`.
Download the bag file and place it under `workspace` directory. 

Before running the docker container, make your folder structure should look something like the following: 

```
workspace
├── build
├── Dockerfile
├── get_tensor_rt_model.sh
├── install
├── internship_assignment_sample_bag
│   ├── internship_assignment_sample_bag_0.db3
│   └── metadata.yaml
├── log
├── main.py
├── model_config.yaml
├── models
│   ├── best.cache
│   ├── best.engine
│   ├── best.onnx
│   └── best.pt
├── Pallet-Detection-1
│   ├── data.yaml
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── test/
│   ├── train/
│   └── valid/
├── scripts
│   └── generate_engine_model.py
├── src
│   └── pallet_detection/
├── start.sh
└── yolo11s-seg.pt
```
Note that you can also choose to place the dataset folder `Pallet-Detection-1` inside workspace, but since we are not training, you can skip this step. 

Then perform the following steps
- run `chmod +x run.sh`
- run `./run.sh` 


The above shell script runs a user friendly docker container that I believe is self-explanatory. If you want to visualize the detections in `rviz` please enter *yes*, otherwise enter *no*. 

---
Note that the docker image comes with ROS2 humble and Ultralytics installed for performing the above mentioned tasks. 

The container will automatically run the `start.sh` or `start_with_rviz.sh` script depending on visualization requirements, and it will further prompt the user to input optional arguments to perform each of these individual tasks. 

If you choose to save the predictions locally, the shell script will create a directory (`predictions`) & subdirectories (`class_0` and `class_1`) inside `workspace` and saves the output images that are published to the 5 topics (`/all_detections`, `/ground_detections`, `/pallet_detections`, `/ground_segmentmask`, `/pallet_segmentmask`)

---
In case you want an interactive shell inside the docker container run the following command: 
```
docker run -it --rm --runtime nvidia --net=host -v  <path-to-workspace>:/home/ws peer_robotics:jetson_v5 /bin/bash
```
Once inside, run `chmod +x start.sh` and run `./start.sh` or `chmod +x start_with_rviz.sh` and run `./start_with_rviz.sh`. 

## Results I obtained can be viewed here : [link for the detection results I got](https://drive.google.com/drive/folders/1fs4lLZgcdoZoiF7aGXPC6BKuSoB8UfwN?usp=sharing)

### Visuals
![assets\my_results-gif.gif](https://github.com/PannagaS/pallet-detection-task/blob/main/assets/my_results-gif.gif)

### RViz visuals
![assets\rviz-screen-recording-gif.gif](https://github.com/PannagaS/pallet-detection-task/blob/main/assets/rviz-screen-recording-gif.gif)
### Object detection and segmentation model performance metrics
The following charts show the model's performance after training for 100 epochs on training dataset & was validated against the validation dataset. 

#### Results
![assets\results.png](https://github.com/PannagaS/pallet-detection-task/blob/main/assets/results.png)
 
#### Normalized Confusion Matrix
 ![assets\confusion_matrix_normalized.png#center](https://github.com/PannagaS/pallet-detection-task/blob/main/assets/confusion_matrix_normalized.png)
 

## Additional Notes
 
As a side, it takes ~6-7 seconds for the `image_processor` node to load the **best.engine** model. To make sure the incoming images are not processed before the model is loaded, I am delaying the execution of image_publisher node or playing the given ros bag. For this reason the `image_processor` node is also publishing a flag (True/False) to `model_ready` topic to check if best.engine has been loaded yet. Depending on the flag, I then utilize `model_ready_waiter` node to subscribe to the `model_ready` topic and see if the flag reads True. If it reads True, only then I am executing the remainder of the process in launch file. 

**However**, the above process is **redundant and unnecessary** when you are feeding live data. The start.sh script takes care of running the appropriate launch file depending on the arguments you pass. 
 

## Contributor
Pannaga Sudarshan (pannaga@umich.edu)
