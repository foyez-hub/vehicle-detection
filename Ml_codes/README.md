# Vehicle Detection and Tracking (ML Codes)

This directory contains the machine learning components of the Vehicle Detection project, including training scripts, inference tools, and a graphical interface for real-time monitoring.

## 📂 Directory Structure

- `model_train/`: Contains Jupyter notebooks for training the YOLO model.
- `models/`: Storage for trained weights (`.pt` and `.onnx` formats).
- `src/`: Core logic for tracking, GUI, and video processing.
- `video/`: Sample videos for testing and inference.
- `image/`: Sample images for testing.
- `run_gui.py`: Main GUI application for real-time detection.
- `annotate_video.py`: Script for batch video processing and annotation.
- `annotate_image.py`: Script for image and directory annotation.

---

## 🚀 Core Features

### 1. Real-time Monitoring GUI
A user-friendly interface to visualize vehicle detection and counting.
- Supports video file inputs.
- Real-time vehicle counting and tracking.
- Toggle-able detection and pause/resume controls.

### 2. Batch Video Annotation
Process long video files to generate annotated versions with bounding boxes and tracking IDs.

### 3. Image Processing
Annotate single images or entire directories, with options to save results in YOLO format for further training.

---

## 🧠 Model Training

You can train or fine-tune the vehicle detection model using the provided notebook or on Kaggle.

### Local Training
Use the notebook located at:
`Ml_codes/model_train/train/train-vehicle-detector.ipynb`

### Kaggle Training
The training code is also available on Kaggle for GPU-accelerated training:
👉 [Train Vehicle Detector on Kaggle](https://www.kaggle.com/code/emammame/train-vehicle-detector)

---

## 🛠️ Setup and Installation

1.  **Environment**: Python 3.8+ is required.
2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `opencv-python`, `ultralytics`, and `filterpy` are the primary dependencies.*

---

## 📖 Usage Instructions

### Run GUI Application
```bash
python run_gui.py --video "video/Supporting video for Dataset-1.mp4" --model "models/best.pt"
```

### Annotate Video
```bash
python annotate_video.py --video "video/Supporting video for Dataset-1.mp4" --output "video/output_annotated.mp4"
```

### Annotate Images
```bash
# Process a directory
python annotate_image.py --image "image/" --output "results/annotated_images" --conf 0.5 --save-txt
```

**Common Arguments:**
- `--model`: Path to `.pt` or `.onnx` model (default: `models/best.pt`).
- `--video`: Path to input video.
- `--conf`: Confidence threshold for detections.

---

## 🔧 Troubleshooting

- **Missing `cv2`**: Run `pip install opencv-python`.
- **Path Errors**: Ensure you are running scripts from the `Ml_codes` directory or provide absolute paths.
- **ONNX Performance**: Ensure you have `onnxruntime` or `onnxruntime-gpu` installed for faster ONNX inference.

