# WEBCAM_DETECTION.PY

This code implements a real-time object detection system using the YOLO (You Only Look Once) model, specifically the YOLOv8n variant, with the help of the Ultralytics library. The code is structured within a class called `DETRClass`, which handles video capture, model loading, object detection, and visualization of results.

## Detailed Breakdown:

### Imports
- `torch`: The PyTorch library, used for tensor computations and model handling.
- `numpy`: A library for numerical operations, particularly for handling arrays.
- `cv2`: OpenCV library for image and video processing.
- `time`: A module to measure time intervals, useful for calculating frames per second (FPS).
- `YOLO`: The YOLO class from the Ultralytics library, which provides pre-trained models for object detection.

### Class Definition: `DETRClass`
This class encapsulates the functionality for object detection.

#### `__init__` Method
- **Parameters**: `capture_index` specifies the index of the video capture device (e.g., webcam).
- **Device Selection**:
  - `self.device` is set to use the Metal Performance Shaders (MPS) backend if available (for Mac M1), otherwise it defaults to CPU.
  - Prints the device being used.
- **Model Loading**:
  - Attempts to load the YOLOv8n model from a file named `yolov8n.pt`.
  - If loading fails, catches the exception, prints an error message, and downloads the model automatically using `YOLO('yolov8n')`.
- **Class Names**:
  - `self.CLASS_NAMES_DICT` stores the class names that the model can detect, which are printed to the console.

#### `plot_bboxs` Method
- **Parameters**: Takes `results` (the output from the model) and `frame` (the current video frame).
- **Extract Detections**:
  - Retrieves bounding boxes, class IDs, confidences, and coordinates from the model's results.
- **Draw Bounding Boxes**:
  - For each detected object, extracts the coordinates of the bounding box and draws a rectangle using OpenCV.
  - Creates a label with the class name and confidence score, displayed above the bounding box.
- **Return**: The modified frame with bounding boxes and labels drawn.

#### `__call__` Method
This method allows the class instance to be called like a function.
- **Video Capture**:
  - Initializes video capture from the specified index and sets the frame dimensions to 1280x720 pixels.
- **Main Loop**:
  - Continuously captures frames from the video.
  - **Frame Capture**: Reads a frame from the video. If reading fails, breaks the loop.
  - **Object Detection**: Uses the model to predict objects in the current frame.
  - **Draw Bounding Boxes**: Calls `plot_bboxs` to draw the detected objects on the frame.
  - **FPS Calculation**: Measures the time taken to process the frame and calculates frames per second (FPS).
  - **Display Frame**: Shows the processed frame in a window titled "DETR" with the FPS displayed.
  - **Exit Condition**: The loop breaks if the 'q' key is pressed.
- **Cleanup**: Releases the video capture and closes OpenCV windows.

### Initialization and Execution
- An instance of `DETRClass` is created with `capture_index` set to 0, which typically refers to the default webcam.
- The instance is called to start the object detection process.

### Summary
This code sets up a real-time object detection system that:
- Captures video from a webcam.
- Processes each frame to detect objects using a YOLO model.
- Displays results with bounding boxes and labels in a window.
- Uses the MPS backend for optimized performance on compatible hardware.
- Allows easy termination by pressing the 'q' key.

---

# VIDEO_DETECTION.PY

This code implements a simple object detection system using the YOLO (You Only Look Once) model from the Ultralytics library. It processes a video file named `cosmetics.mp4` and displays the detected objects with bounding boxes and class labels in real time.

## Detailed Breakdown:

### Imports
- `cv2`: The OpenCV library, used for image and video processing.
- `YOLO`: The YOLO class from the Ultralytics library, which provides pre-trained models for object detection.
- `numpy`: A library for numerical operations, particularly for handling arrays.

### Video Capture
- **Initialization**: `cap = cv2.VideoCapture("cosmetics.mp4")` initializes video capture from the specified video file.

### Model Loading
- **Model**: `model = YOLO("yolov8m.pt")` loads the YOLOv8 model weights from `yolov8m.pt`.

### Main Loop
1. **Frame Reading**:
   - `ret, frame = cap.read()`: Reads a frame from the video.
   - If not successful (`if not ret:`), breaks the loop.
2. **Object Detection**:
   - `results = model(frame, device="mps")`: Runs the YOLO model for object detection.
   - Extracts bounding boxes and class IDs:
     - `bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")`
     - `classes = np.array(result.boxes.cls.cpu(), dtype="int")`
3. **Drawing Bounding Boxes**:
   - Draws rectangles and class labels using OpenCV functions.
4. **Displaying the Frame**:
   - `cv2.imshow("Img", frame)` displays the processed frame.
5. **Exit Condition**:
   - Breaks the loop if the 'Esc' key is pressed.

### Cleanup
- Releases the video capture (`cap.release()`) and closes OpenCV windows (`cv2.destroyAllWindows()`).

---

# OBJECT_DETECTION_TF.PY

This code implements an object detection system using a pre-trained SSD MobileNet V2 model from TensorFlow Hub. It processes a video file named `inventory.mp4`, detects objects in each frame, and saves the output video.

## Detailed Breakdown:

### Imports
- `os`: Provides functions for interacting with the operating system.
- `cv2`: The OpenCV library, used for image and video processing.
- `tensorflow`: TensorFlow library for machine learning.
- `tensorflow_hub`: A library for loading pre-trained models from TensorFlow Hub.
- `numpy`: A library for numerical operations.
- `datetime`: A module for working with timestamps.

### Configuration
- **Video Name**: `VIDEO_NAME` specifies the input video file name.
- **Paths**: Constructs paths for input and output directories.

### Functions
1. **`load_model()`**:
   - Loads the SSD MobileNet V2 model.
2. **`process_frame(frame, model)`**:
   - Preprocesses the video frame.
   - Detects objects and draws bounding boxes.
3. **`process_video()`**:
   - Orchestrates the video processing pipeline.

---

# OBJECT_DETECTION_HF.PY

This code implements an object detection system using the DETR (DEtection TRansformer) model from the Hugging Face library. It processes video frames and saves the processed video with bounding boxes and labels.

## Key Features
- **Imports**: Includes PyTorch, OpenCV, and Hugging Face libraries.
- **Device Selection**: Uses MPS backend for optimized performance.
- **Main Workflow**:
  - Loads the DETR model and processes each video frame.
  - Visualizes detections with bounding boxes and labels.
