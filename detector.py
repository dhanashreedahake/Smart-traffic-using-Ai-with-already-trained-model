import cv2
import threading
import time
import uuid
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# COCO class IDs for vehicles: 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class TrafficDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.current_data = {
            "vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "density": "LOW",
            "greenTime": 10,
            "signalStatus": "RED",
            "fps": 0,
            "timestamp": datetime.now().isoformat(),
            "status": "IDLE"
        }
        self.history = []       # stores per-analysis summaries
        self.is_processing = False
        self._thread = None
        self._lock = threading.Lock()
        self.current_frame = None # store the current annotated frame for streaming

    # ------------------------------------------------------------------ #
    def get_latest_data(self):
        with self._lock:
            return dict(self.current_data)

    def get_history(self):
        return list(self.history)
    
    def get_frame(self):
        with self._lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _density_level(count: int) -> str:
        if count < 10:
            return "LOW"
        elif count < 20:
            return "MEDIUM"
        elif count < 40:
            return "HIGH"
        else:
            return "CRITICAL"

    @staticmethod
    def _green_time(count: int) -> float:
        return max(10, min(60, 10 + count * 2))

    # ------------------------------------------------------------------ #
    def _process_loop(self, video_path: str, analysis_id: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.is_processing = False
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_no = 0
        t0 = time.time()
        
        # Aggregators for final result
        total_counts = []
        car_counts = []
        bike_counts = []
        bus_counts = []
        truck_counts = []

        while cap.isOpened() and self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            
            # Speed up: perform detection every 2nd or 3rd frame if needed, 
            # but here we do it every frame for accuracy as requested.
            # Strictly filter for vehicle classes only (1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck)
            # This ensures 'person' (class 0) is never detected or plotted.
            results = self.model(frame, verbose=False, conf=0.25, classes=[1, 2, 3, 5, 7])
            counts = {cid: 0 for cid in VEHICLE_CLASSES}

            annotated_frame = results[0].plot()

            for r in results:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    if cid in VEHICLE_CLASSES:
                        counts[cid] += 1

            total = sum(counts.values())
            
            # bike_count combines both bicycles (1) and motorcycles (3)
            bike_combined = counts[1] + counts[3]

            # Record for final average
            total_counts.append(total)
            car_counts.append(counts[2])
            bike_counts.append(bike_combined)
            bus_counts.append(counts[5])
            truck_counts.append(counts[7])

            elapsed = time.time() - t0
            fps = len(total_counts) / elapsed if elapsed > 0 else 0

            density = self._density_level(total)
            green = self._green_time(total)
            signal = "GREEN" if density in ("LOW", "MEDIUM") else "AMBER"

            with self._lock:
                self.current_frame = annotated_frame
                self.current_data = {
                    "vehicles": total,
                    "cars": counts[2],
                    "motorcycles": bike_combined,
                    "buses": counts[5],
                    "trucks": counts[7],
                    "density": density,
                    "greenTime": int(green),
                    "signalStatus": signal,
                    "fps": round(fps, 1),
                    "timestamp": datetime.now().isoformat(),
                    "analysisId": analysis_id,
                    "status": "PROCESSING"
                }
            
            # Real-time pacing (adjust based on system load)
            time.sleep(0.005)

        cap.release()

        # Calculation of final accurate results (averages during the video)
        if total_counts:
            avg_total = round(np.mean(total_counts), 1)
            avg_cars = round(np.mean(car_counts), 1)
            avg_bikes = round(np.mean(bike_counts), 1)
            avg_buses = round(np.mean(bus_counts), 1)
            avg_trucks = round(np.mean(truck_counts), 1)
            
            final_density = self._density_level(int(avg_total))
            final_green = self._green_time(int(avg_total))
            
            with self._lock:
                self.current_data = {
                    "vehicles": avg_total,
                    "cars": avg_cars,
                    "motorcycles": avg_bikes,
                    "buses": avg_buses,
                    "trucks": avg_trucks,
                    "density": final_density,
                    "greenTime": int(final_green),
                    "signalStatus": "FINISHED",
                    "fps": 0,
                    "timestamp": datetime.now().isoformat(),
                    "analysisId": analysis_id,
                    "status": "COMPLETED"
                }

            # Save to history
            self.history.append({
                "id": analysis_id,
                "filename": video_path.split("/")[-1].split("\\")[-1],
                "avgVehicles": avg_total,
                "density": final_density,
                "greenTime": int(final_green),
                "framesAnalysed": len(total_counts),
                "processedAt": datetime.now().isoformat(),
            })

        self.is_processing = False

    # ------------------------------------------------------------------ #
    def process_video(self, video_path: str):
        if self._thread and self._thread.is_alive():
            self.is_processing = False
            self._thread.join(timeout=3)

        analysis_id = str(uuid.uuid4())[:8]
        self.is_processing = True
        self.current_frame = None
        self._thread = threading.Thread(
            target=self._process_loop,
            args=(video_path, analysis_id),
            daemon=True,
        )
        self._thread.start()
        return analysis_id

    def stop(self):
        self.is_processing = False
