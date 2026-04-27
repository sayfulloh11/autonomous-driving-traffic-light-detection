# Autonomous Driving Traffic Light Detection

A professional computer vision project for traffic light detection in autonomous driving systems.

## Key Features

- Small object detection
- COCO dataset processing
- COCO → YOLO conversion
- YOLO training (Ultralytics)
- RF-DETR training
- Swin Transformer experiments
- Tracking pipeline
- Auto-labeling workflow

## Pipeline

Raw annotations → COCO → YOLO format → Training → Inference → Tracking → Auto-labeling

## Tech Stack

Python, PyTorch, OpenCV, YOLO, RF-DETR, Swin Transformer, Docker

## Structure

configs/ → training configs  
src/ → training + inference  
scripts/ → data processing  
rasnet/ → FasterRCNN experiments  
docs/ → explanation  
results/ → outputs  

## Note

Private datasets, logs, and weights are removed.

## Author

Sayfulloh Khurbaev  
Computer Vision Engineer
