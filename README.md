# Multi-Model Object Detection Suite  
**YOLOv8 + DETR + TensorFlow SSD | Real-Time & Batch Inference | Mac M1 Optimized**

---

## Overview

This project demonstrates a comprehensive object detection pipeline using three powerful models:

- **YOLOv8** from Ultralytics (for real-time webcam and video inference)
- **DETR (Detection Transformer)** from Hugging Face (for transformer-based object detection)
- **SSD MobileNet V2** from TensorFlow Hub (for efficient and scalable deployment)

The goal is to process both **webcam streams** and **video files** using pre-trained models across PyTorch and TensorFlow ecosystems. Optimized for Apple Silicon (Mac M1/M2) using Metal Performance Shaders (MPS), this suite serves as a reference implementation for scalable, cross-platform computer vision workflows.


---

## Features

- Real-time object detection from webcam using YOLOv8
- Batch video processing with DETR (Transformers) and SSD (TF Hub)
- Cross-framework support: PyTorch, TensorFlow, Hugging Face
- Frame-by-frame visualizations with bounding boxes and class labels
- Export results as annotated video and JSON/CSV data
- Optimized for **Mac M1** with **MPS backend**
- Modular Python design for extension and experimentation

---

## Model Architectures Used
- YOLOv8n / YOLOv8m	Ultralytics	
- DETR-ResNet50	Hugging Face Transformer Model	
- SSD MobileNet V2

## Mac M1 Optimization
All scripts check for MPS (Metal Performance Shaders) support and default to device='mps' if available. This ensures smooth and accelerated inference on MacBooks with Apple Silicon chips.

## Outputs
Each script generates:
- Annotated video frames
- Combined output video
- (Optional) JSON/CSV file with detected labels, boxes, and scores
