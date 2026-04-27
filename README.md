# Autonomous Driving Traffic Light Detection

Computer vision project focused on traffic light detection for autonomous driving datasets, with emphasis on small-object detection, dataset conversion, model training, and auto-labeling workflows.

## Overview

This project demonstrates an end-to-end traffic light detection pipeline for autonomous driving perception tasks.

The main goals are:

- Detect small traffic lights in road-scene images
- Prepare datasets in COCO format
- Train and evaluate deep learning detection models
- Build an auto-labeling pipeline for large-scale video frame datasets
- Compare models such as YOLO, Swin Transformer, RF-DETR, and DINO-based detectors

## Key Features

- Traffic light detection for autonomous driving images
- Small-object detection focused preprocessing
- COCO dataset conversion and validation
- Training and inference pipeline
- Auto-labeling workflow design
- Model evaluation using mAP, AP50, AP75, and F1 score

## Tech Stack

- Python
- PyTorch
- OpenCV
- MMDetection
- YOLO
- RF-DETR
- COCO format
- Docker
- Linux

## Project Structure

```text
autonomous-driving-traffic-light-detection/
├── configs/
├── data_samples/
├── docs/
├── scripts/
├── src/
├── results/
├── README.md
├── requirements.txt
└── LICENSE
