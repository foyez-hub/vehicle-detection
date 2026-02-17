import argparse
import os
import sys

# Ensure src module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tracker import VehicleTracker

def annotate_image(model_path, input_path, output_path, conf=0.25, save_txt=False):
    """
    Annotate image(s) using trained YOLO model via predict() mode.
    Handles batch processing and saving automatically.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    tracker = VehicleTracker(model_path)
    
    print(f"Running inference on {input_path}...")
    print(f"Results will be saved to {output_path}")

    # Map output_path to project/name structure for Ultralytics
    # Ultralytics saves to project/name.
    # We want output_path to be the final folder.
    # So we set project to current dir (or parent of output) and name to output folder name.
    
    project_dir = os.path.dirname(os.path.abspath(output_path))
    name_dir = os.path.basename(output_path)
    
    # Run Prediction
    # save=True ensures annotated images are saved
    # save_txt=True (if flag set) ensures .txt labels are saved
    tracker.predict(
        source=input_path,
        conf=conf,
        save=True,
        save_txt=save_txt,
        project=project_dir,
        name=name_dir,
        exist_ok=True # Increment name if exists? User probably wants overwrite/merge into same folder
    )

    print(f"Done! Check results in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate images using trained YOLO/ONNX model.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file or directory.")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model file (pt or onnx).")
    parser.add_argument("--output", type=str, default="runs/detect/predict", help="Path to save annotated output.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--save-txt", action="store_true", help="Save text labels alongside images.")

    args = parser.parse_args()
    
    annotate_image(
        args.model, 
        args.image, 
        args.output, 
        conf=args.conf, 
        save_txt=args.save_txt
    )
