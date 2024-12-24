
from ultralytics import YOLO
import torch
# import numpy as np
from IPython import embed
# Control flow for training, testing, and exporting
train = False
test = False
export_model = False
test_onnx_model = True

# Training
if train:
    model = YOLO("yolo11s-seg.pt")
    path_to_dataset = "/home/ps/Pallet-Detection-1/data.yaml"
    results = model.train(data=path_to_dataset, epochs=100)

# Testing PyTorch Model
if test:
    model = YOLO("/home/ps/best.pt")
    images_path = "/home/ps/Pallet-Detection-1/test/images"
    results = model.predict(images_path, conf=0.8, save=True)

# Export to engine
if export_model:
    embed()
    model = YOLO("/home/ps/best.pt")
    model.export(format="engine", task='detect')
    print("engine model exported")

# Testing ONNX Model
if test_onnx_model:
    # Load ONNX model
    onnx_model = YOLO("/home/ps/best.engine", task='segment')
    
    
    # Run inference
    images_path = "/home/ps/Pallet-Detection-1/test/images"
    # embed()
    results = onnx_model.predict(images_path, imgsz = (640,640), conf=0.8, save=True)
    #for result in results:
    #    print("Raw result:", result)
    #    print("Class IDs:", result.boxes.cls)
    #    print("Class probabilities:", result.boxes.conf)
    
    print("ONNX inference completed.")
