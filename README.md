# Real Time Semantic Segmentation and Vision Based Heading Controller
This repository features a robust, real-time autonomous navigation system designed for campus environments. Leveraging the BiSeNet V1 architecture, the system achieves high-precision semantic segmentation to identify navigable paths, pedestrians, and specific road features like speed breakers.

The model was trained to a peak pixel-wise accuracy of 93.82% on a localized campus dataset, ensuring reliable navigation under specific domain constraints.

## Project Overview

The core objective of this project is to bridge the gap between computer vision and robotic control. By processing a live camera feed, the system generates real-time control commands (V for linear velocity and W for angular velocity) based on a segmented mask of the environment.
Key Features

- **Precision Segmentation:** Identifies six distinct classes:
  - Background
  - Human
  - Obstacle
  - Road
  - Sidewalk
  - Speed Breaker.
- **Speed Breaker Awareness:** A specialized logic branch that detects speed breakers and automatically reduces linear velocity by 50% (from 0.22 to 0.11) for smoother traversal.
- **Safety Corridor (ROI):** Implements a dynamic "Region of Interest" in the bottom-middle of the frame to trigger Emergency Stops if pedestrians or obstacles block the immediate path.
- **Camera Calibration:** Includes a full calibration pipeline using checkerboard patterns to undistort raw wide-angle camera feeds before inference.

## Modular Architecture
The repository is structured for maintainability and scalability:
Robot-Navigation-BiSeNet/
```text
├── weights/
│   └── bisenet_v2_final_best.pth   <-- Store your trained model here
├── data/
│   └── camera_calibration.npz      <-- Store calibration results here
├── src/
│   ├── __init__.py
│   ├── utils.py                   <-- Calibration and math helper functions
│   ├── dataset.py                 <-- LabeledRoadDataset class
│   └── navigation.py              <-- get_control_commands logic
├── train.py                        <-- Main script to start training
├── demo.py                         <-- Presentation Day Script (Video Gen)
├── requirements.txt                <-- List of libraries (torch, cv2, etc.)
└── README.md                       <-- Project documentation
```
## Performance Metrics
- Accuracy: 93.82% (Pixel-wise accuracy on unseen campus test frames) and 85 mIoU.
- Architecture: BiSeNet V1 with a ResNet-18 backbone for a balance between inference speed and feature extraction depth.
- Frameworks: PyTorch, OpenCV, NumPy.

## Installation & Usage
**1. Clone the Repository**
```
git clone https://github.com/your-username/Robot-Navigation-BiSeNet.git
cd Robot-Navigation-BiSeNet
```
**2. Install Dependencies**
```
pip install -r requirements.txt
```
**3. Run the Demonstration**
To process a video and generate the navigation overlay:
Bash
```
python demo.py
```

## Demonstration Logic
The output video provides a side-by-side view:

**1. Left (Input):** Undistorted RGB feed with a green navigation line pointing toward the calculated target road center.

**2. Right (Mask):** The semantic output where:
  - Red: Road (Navigable Path)
  - Green: Speed Breaker (Triggers Slowing)
  - Yellow/Purple: Human/Obstacle (Triggers Stop)
  - Blue: Sidewalk (Boundary)
