import cv2
import time
import sys
import os
import psutil
from collections import defaultdict
from src.tracker import VehicleTracker

class TrafficMonitor:
    """
    A class to monitor traffic using a VehicleTracker for detection, with a GUI.
    """

    def __init__(self, model_path, video_path):
        """
        Initialize the TrafficMonitor.

        Args:
            model_path (str): Path to the YOLO model file.
            video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        self.running = False
        self.prev_time = 0
        self.counts = defaultdict(set)
        self.cap = None
        self.window_name = "Traffic Monitoring System"
        
        # Initialize Tracker
        self.tracker = VehicleTracker(model_path)
        
        # Button configurations (x, y, w, h)
        self.btn_start = (20, 100, 100, 40)
        self.btn_stop = (140, 100, 100, 40)

        self._load_video()
        self._setup_window()

    def _load_video(self):
        """Load the video capture."""
        try:
            print(f"Opening video from {self.video_path}...")
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file {self.video_path}")
        except Exception as e:
            print(f"Error loading video: {e}")
            sys.exit(1)

    def _setup_window(self):
        """Setup the GUI window and mouse callback."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Start Button
            bx, by, bw, bh = self.btn_start
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.running = True

            # Check Stop Button
            bx, by, bw, bh = self.btn_stop
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.running = False

    def draw_dashboard(self, frame, fps):
        """
        Draw the dashboard statistics and buttons on the frame.
        """
        # --- System Stats ---
        cpu_usage = psutil.cpu_percent()
        gpu_usage = self.tracker.get_gpu_usage()
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"CPU: {cpu_usage}%", (140, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"GPU: {gpu_usage}", (280, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- Buttons ---
        start_color = (0, 255, 0) if self.running else (100, 100, 100)
        sbx, sby, sbw, sbh = self.btn_start
        cv2.rectangle(frame, (sbx, sby), (sbx + sbw, sby + sbh), start_color, -1)
        cv2.putText(frame, "START", (sbx + 15, sby + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        stop_color = (0, 0, 255) if not self.running else (100, 100, 100)
        stbx, stby, stbw, stbh = self.btn_stop
        cv2.rectangle(frame, (stbx, stby), (stbx + stbw, stby + stbh), stop_color, -1)
        cv2.putText(frame, "STOP", (stbx + 20, stby + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Object Counts ---
        y_pos = 180
        cv2.putText(frame, "Object Counts:", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for cls, ids in self.counts.items():
            cv2.putText(
                frame, f"{cls}: {len(ids)}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_pos += 25

    def process_frame(self, frame):
        """
        Process a single frame for vehicle tracking using the tracker module.
        """
        # Delegate tracking to the tracker module
        results = self.tracker.track(frame)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            names = self.tracker.get_names()

            for box, tid, cls, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = names[int(cls)]

                self.counts[class_name].add(int(tid))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} ID:{int(tid)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

    def run(self):
        """Run the main loop of the traffic monitor."""
        print("Starting traffic monitoring... Controls available via GUI buttons or keys (S/P/Q).")
        
        while True:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video file or cannot read the frame.")
                    self.running = False
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.current_frame = frame.copy()
            else:
                if not hasattr(self, 'current_frame'):
                     ret, frame = self.cap.read()
                     if ret:
                         self.current_frame = frame
                     else:
                         self.current_frame = None
                
                if self.current_frame is not None:
                     frame = self.current_frame.copy()
                     cv2.putText(frame, "PAUSED", (500, 360), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    break

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                self.running = True
            elif key == ord('p'):
                self.running = False
            elif key == ord('q'):
                break

            if self.running and self.current_frame is not None:
                 self.process_frame(frame)

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
            self.prev_time = curr_time

            self.draw_dashboard(frame, fps)

            cv2.imshow(self.window_name, frame)

        self.cleanup()

    def cleanup(self):
        """Release resources and close windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Exiting.")

