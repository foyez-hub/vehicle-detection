# Source Code (src)

This directory contains the core logic and helper modules for the vehicle detection system.

## 📁 Modules

- `gui_app.py`: Logic for the Traffic Monitor GUI (built with OpenCV/Tkinter).
- `tracker.py`: Implementation of the `VehicleTracker` class, handling YOLO inference and SORT/Kalman Filter tracking.
- `process_video.py`: Utilities for batch processing and annotating video streams.

## 🛠️ Internal Use

These modules are imported by the top-level scripts:
- `run_gui.py` -> `src.gui_app`
- `annotate_video.py` -> `src.process_video`
- `annotate_image.py` -> `src.tracker`
