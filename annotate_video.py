import argparse
import os
import sys

# Ensure src module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.process_video import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate video using trained YOLO model.")
    parser.add_argument("--video", type=str, default="video/Inference -1.mp4", help="Path to input video file.")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model file.")
    parser.add_argument("--output", type=str, default="video/output_annotated.mp4", help="Path to save annotated video.")

    args = parser.parse_args()

    # Expand paths
    video_path = os.path.abspath(args.video)
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")

    process_video(model_path, video_path, output_path)
