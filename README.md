# Real-Time Custom Object Detection: Purdey's Energy Drink

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

A complete end-to-end Computer Vision pipeline built in **under 6 hours**. 
This project demonstrates how to fine-tune a state-of-the-art YOLOv8 Nano model to detect a highly reflective, custom object (Purdey's Apple & Grape can) using a micro-dataset collected via webcam.

## Goals
* **Local Inference:** Run entirely on a local CPU/GPU without cloud APIs.
* **Rapid Prototyping:** Go from raw data collection to deployed model in one afternoon.
* **Robustness:** Handle metallic reflections and varying lighting conditions.

## Structure
```text
purdeys-detector/
├── data/
│   ├── raw/                 # Original captured frames & labels
│   └── dataset/             # YOLO formatted (train/val split)
├── models/
│   ├── yolov8n.pt           # Base pretrained model
│   └── purdeys_v1.pt        # Fine-tuned custom weights
├── src/
│   ├── collect_data.py      # Webcam capture script
│   ├── split_data.py        # Dataset partitioning script
│   └── inference.py         # Real-time detection & recording
└── runs/                    # Training metrics and logs
```

## Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/purdeys-detector.git](https://github.com/YOUR_USERNAME/purdeys-detector.git)
    cd purdeys-detector
    ```

2.  **Install Dependencies:**
    ```bash
    pip install ultralytics opencv-python
    ```

## Usage Pipeline

### 1. Data Collection
Capture training images using your webcam.
* Press `s` to save a valid object frame.
* Press `b` to save a background frame (negative sample).
```bash
python src/collect_data.py
```

### 2. Labeling
We use [MakeSense.ai](https://www.makesense.ai/) for rapid, browser-based annotation (no installation required).
1.  Upload images from `data/raw`.
2.  Draw bounding boxes (Class: `purdeys_can`).
3.  Export Annotations in **YOLO format**.
4.  Unzip the text files into `data/raw`.

### 3. Training
Fine-tune the YOLOv8n model. The configuration is handled in `data/dataset/data.yaml`.
```bash
yolo task=detect mode=train model=yolov8n.pt data=data/dataset/data.yaml epochs=50 imgsz=640
```

### 4. Inference & Recording
Run the detector on your live webcam feed. This script automatically records the session to `.mp4`.
```bash
python src/inference.py
```

## Results
The model achieves consistent detection at ~30 FPS on a standard laptop CPU.

## Lessons Learned
* **Data Quality > Quantity:** 50 high-quality images with diverse angles outperformed 200 repetitive ones.
* **Negative Sampling:** Explicitly training on "background" images reduced false positives by 90%.
* **Reflective Surfaces:** Diffuse lighting is critical for labeling metallic objects.

---
*Built with Ultralytics YOLOv8 and OpenCV.*
