import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from datetime import datetime

# Configure paths - Simple structure
VIDEO_NAME = "inventory.mp4"  # Your video file name
CURRENT_DIR = os.getcwd()
INPUT_VIDEO_PATH = os.path.join(CURRENT_DIR, VIDEO_NAME)

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    print("Loading model...")
    model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    print("Model loaded successfully!")
    return model

def process_frame(frame, model):
    # Preprocess
    input_tensor = tf.convert_to_tensor(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = tf.expand_dims(input_tensor, 0)
    
    # Detect
    predictions = model(input_tensor)
    boxes = predictions['detection_boxes'][0].numpy()
    scores = predictions['detection_scores'][0].numpy()
    classes = predictions['detection_classes'][0].numpy().astype(int)
    
    # Filter detections
    valid_indices = scores >= 0.5
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]
    
    # Define class names (modify as needed)
    class_names = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck'
        # Add more classes as needed
    }
    
    # Draw results
    height, width, _ = frame.shape
    for box, score, class_id in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box
        xmin, xmax = int(xmin * width), int(xmax * width)
        ymin, ymax = int(ymin * height), int(ymax * height)
        
        # Draw box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_names.get(class_id, f'Class {class_id}')}: {score:.2f}"
        cv2.putText(frame, label, (xmin, ymin-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_video():
    # First check if video file exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Video file not found at {INPUT_VIDEO_PATH}")
        return
    
    print(f"Processing video: {VIDEO_NAME}")
    
    # Load model
    model = load_model()
    
    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f'detected_{timestamp}.mp4')
    
    # Try different codecs if one doesn't work
    try:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    except:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    frame_count = 0
    print("Starting video processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = process_frame(frame, model)
        out.write(processed_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Output saved to: {output_path}")

# Run the detection
if __name__ == "__main__":
    process_video()