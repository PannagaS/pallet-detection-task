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
    * Perform object detection and segmentaation 

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
![process-flow](assets\process_flow.png)

The entry point for the project is at start.sh shell script. The user is prompted with options for converting best.pt to best.engine (TensorRT model). Further, depending on whether the user wants to run the inference on test images or from the ros bag ([click here](https://drive.google.com/file/d/1BvhP653G3PqfUq96L18gDBIi-5oOYqcr/view) for bag file) or on live camera feed, other model parameters such as confidence, iou, half precision flag (FP16) can be set accordingly. 


## Navigating the project
This project is wrapped in a docker for easy replication on other devices with NVIDIA drivers. 
Please pull the docker image from [here](https://hub.docker.com/repository/docker/pans06/peer_robotics/general). 
```
docker pull pans06/peer_robotics:jetson
```

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

---
Create a new interactive container from the image you just pulled by running the following command: 
```
docker run -it --rm --runtime=nvidia -v <path to workspace>:/home/ws --ipc=host peer_robotics:jetson
```
Note that this will open an interactive shell with ROS2 humble and Ultralytics installed for performing the above mentioned tasks. 

The container will automatically run the `start.sh` script that will further prompt the user to input optional arguments to perform each of these individual tasks. 

If you choose to save the predictions locally, the shell script will create a directory (`predictions`) & subdirectories (`class_0` and `class_1`) inside `workspace` and saves the output images that are published to the 5 topics (`/all_detections`, `/ground_detections`, `/pallet_detections`, `/ground_segmentmask`, `/pallet_segmentmask`)

#### Results I obtained can be viewed here : [link for the detection results I got](https://drive.google.com/drive/folders/1fs4lLZgcdoZoiF7aGXPC6BKuSoB8UfwN?usp=sharing)
### Object detection and segmentation model performance metrics
The following charts show the model's performance after training for 100 epochs on training dataset & was validated against the validation dataset. 

#### Results
![training-results-graphs](assets\results.png)

#### Normalized Confusion Matrix
![normalized-confusion-matrix](assets\confusion_matrix_normalized.png#center) 
 

