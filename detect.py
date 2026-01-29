import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# Load model
model = YOLO("best.pt")

# Video source
VIDEO_PATH = "Video\\Supporting video for Dataset-2.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# FPS calculation
prev_time = 0

# Vehicle count per class
class_counts = defaultdict(set)  # class_name -> set(track_ids)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking inference
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]

            # Store unique vehicle IDs
            class_counts[class_name].add(int(track_id))

            label = f"{class_name} ID:{int(track_id)} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        frame, f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 0, 0), 2
    )

    cv2.imshow("YOLO Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Print vehicle counts
print("Vehicle Counts:")
for cls, ids in class_counts.items():
    print(f"{cls}: {len(ids)}")
