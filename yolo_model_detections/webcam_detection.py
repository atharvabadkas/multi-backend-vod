import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO

class DETRClass:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        print("Using device: ", self.device)
        
        # Load the RT-DETR model from Ultralytics
        try:
            self.model = YOLO('yolov8n.pt')  # Using YOLOv8n as default model
            # Alternatively, you can use RT-DETR with:
            # self.model = YOLO('rtdetr-l.pt')  # or 'rtdetr-x.pt'
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Downloading model...")
            self.model = YOLO('yolov8n')  # This will automatically download the model
        
        self.CLASS_NAMES_DICT = self.model.names
        print("Classes: ", self.CLASS_NAMES_DICT)

    def plot_bboxs(self, results, frame):
        # Extract detections
        boxes = results[0].boxes.cpu().numpy()
        class_ids = boxes.cls.astype(np.int32)
        confidences = boxes.conf
        xyxys = boxes.xyxy

        # Draw boxes and labels
        for i in range(len(confidences)):
            x1, y1, x2, y2 = xyxys[i].astype(int)
            class_id = class_ids[i]
            conf = confidences[i]
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{self.CLASS_NAMES_DICT[class_id]} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1-20), (x1+label_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Failed to open video capture"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame)
            frame = self.plot_bboxs(results, frame)

            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("DETR", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# Initialize and run
transformer_detector = DETRClass(0)
transformer_detector()