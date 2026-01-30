import argparse
import os
import sys

# Ensure src module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gui_app import TrafficMonitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Traffic Monitor GUI.")
    parser.add_argument("--video", type=str, default="video/Supporting video for Dataset-1.mp4", help="Path to input video file.")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model file.")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    model_path = os.path.abspath(args.model)

    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found.")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    app = TrafficMonitor(model_path, video_path)
    app.run()