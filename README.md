# Can Detection

## Overview

This project uses a custom-trained YOLOv8 model to detect cans in images.

## Project Structure

-   `data/`: The image dataset used for training and validation.
-   `models/can.pt`: The trained YOLOv8 model for can detection.
-   `runs/`: Output from model training, including weights and performance metrics.
-   `src/`: Python source code for data preparation and model inference.
    -   `collect_data.py`: Script for collecting image data.
    -   `splitdata.py`: Script for splitting the dataset.
    -   `test.py`: Script for running detection on new images.

## Getting Started

### Prerequisites

-   Python 3.8+
-   A virtual environment is provided in the `venv` directory.

### Installation

1.  **Activate the virtual environment:**

    ```bash
    source venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install ultralytics opencv-python
    ```

### Usage

To run can detection on an image, use the `test.py` script. *Note: The implementation of `test.py` is assumed.*

```bash
python src/test.py --weights models/can.pt --source path/to/your/image.jpg
```

## Training

The model was trained using the YOLOv8 framework. The training configuration and results, such as loss curves and validation metrics, are stored in the `runs/detect/` directory. The final trained model is available at `models/can.pt`.
