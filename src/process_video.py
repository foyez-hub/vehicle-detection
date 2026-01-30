import cv2
import os
import sys
from tqdm import tqdm
from src.tracker import VehicleTracker

def process_video(model_path, video_path, output_path):
    """
    Process video, run tracking, and save annotated output.
    """
    # 1. Initialize Tracker
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        return

    tracker = VehicleTracker(model_path)
    names = tracker.get_names()

    # 2. Open Video Source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 3. Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {video_path}...")
    print(f"Output will be saved to {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    # 4. Processing Loop
    try:
        for _ in tqdm(range(total_frames), desc="Processing Frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            # Run Tracking
            results = tracker.track(frame)

            # Draw Annotations
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                
                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = names[int(cls)]
                    
                    # Draw Rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw Label
                    label = f"{class_name} ID:{int(tid)}"
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Write Frame
            out.write(frame)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        cap.release()
        out.release()
        print("\nResources released.")
        if os.path.exists(output_path):
             print(f"Done! Video saved to {output_path}")

