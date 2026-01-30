import sys
import subprocess
import torch
from ultralytics import YOLO

class VehicleTracker:
    """
    Handles YOLO model initialization, inference, and device management.
    """

    def __init__(self, model_path):
        """
        Initialize the VehicleTracker.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model_path = model_path
        self.device = "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model and determine the computation device."""
        try:
            # Check for GPU
            if torch.cuda.is_available():
                self.device = "0"  # Use first GPU
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("No GPU detected. Using CPU.")

            print(f"Loading model from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def track(self, frame):
        """
        Run tracking on a single frame.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            list: The tracking results from YOLO.
        """
        return self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False,
            device=self.device
        )

    def get_names(self):
        """Return the class names from the model."""
        return self.model.names

    def get_gpu_usage(self):
        """Get GPU usage percentage as a string."""
        if self.device == "cpu":
            return "N/A"

        try:
            # Simple nvidia-smi check for Linux
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            return f"{int(result.strip())}%"
        except Exception:
            return "Err"
