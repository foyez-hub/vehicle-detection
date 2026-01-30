# Vehicle Detection and Tracking

This project implements vehicle detection and tracking using YOLOv8 and OpenCV. It includes a GUI for real-time monitoring and a script for batch video annotation.

## Prerequisites

- Python 3.8 or higher
- GPU recommended for faster inference

## Installation

1.  Clone the repository (if applicable) and navigate to the project directory.

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. GUI Application (`run_gui.py`)

Run the graphical interface to monitor traffic in real-time.

```bash
python run_gui.py --video "path/to/video.mp4" --model "path/to/model.pt"
```

**Arguments:**
- `--video`: Path to the input video file. Default: `video/Inference -1.mp4`
- `--model`: Path to the trained YOLO model. Default: `models/best.pt`

**Controls:**
- **Start**: Button or 'S' key
- **Stop**: Button or 'P' key
- **Quit**: 'Q' key

**Example:**
```bash
# Run with default settings
python run_gui.py

# Run with custom video and model
python run_gui.py --video "video/Traffic Sample.mp4" --model "models/yolov8n.pt"
```

### 2. Video Annotation (`annotate_video.py`)

Process a video file to detect vehicles and save the output with bounding boxes and labels.

```bash
python annotate_video.py --video "path/to/video.mp4" --model "path/to/model.pt" --output "path/to/output.mp4"
```

**Arguments:**
- `--video`: Path to the input video file. Default: `video/Inference -1.mp4`
- `--model`: Path to the trained YOLO model. Default: `models/best.pt`
- `--output`: Path to save the annotated video. Default: `video/output_annotated.mp4`

**Example:**
```bash
python annotate_video.py --video "video/test.mp4" --output "results/annotated.mp4"
```

## Troubleshooting

- **ModuleNotFoundError: No module named 'cv2'**: Run `pip install opencv-python`.
- **FileNotFoundError**: Ensure paths to video and model files are correct. Use quotes for paths with spaces.
