import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("best.pt")
cap = cv2.VideoCapture("Video\Inference -2.mp4")

running = False
prev_time = 0
counts = defaultdict(set)

def draw_dashboard(frame, fps):
    y = 70
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for cls, ids in counts.items():
        cv2.putText(
            frame, f"{cls}: {len(ids)}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        y += 25


while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # Start / Stop controls
    if key == ord('s'):
        running = True
    elif key == ord('p'):
        running = False
    elif key == ord('q'):
        break

    if running:
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, cls, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]

                counts[class_name].add(int(tid))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} ID:{int(tid)} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    draw_dashboard(frame, fps)

    cv2.imshow("Traffic Monitoring System", frame)

cap.release()
cv2.destroyAllWindows()
