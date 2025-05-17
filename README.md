# Multi-Model Object Detection Suite  
**YOLOv8 + DETR + TensorFlow SSD | Real-Time & Batch Inference | Mac M1 Optimized**

---

## Overview

This project demonstrates a comprehensive object detection pipeline using three powerful models:

- **YOLOv8** from Ultralytics (for real-time webcam and video inference)
- **DETR (Detection Transformer)** from Hugging Face (for transformer-based object detection)
- **SSD MobileNet V2** from TensorFlow Hub (for efficient and scalable deployment)

The goal is to process both **webcam streams** and **video files** using pre-trained models across PyTorch and TensorFlow ecosystems. Optimized for Apple Silicon (Mac M1/M2) using Metal Performance Shaders (MPS), this suite serves as a reference implementation for scalable, cross-platform computer vision workflows.


---

## Features

- Real-time object detection from webcam using YOLOv8
- Batch video processing with DETR (Transformers) and SSD (TF Hub)
- Cross-framework support: PyTorch, TensorFlow, Hugging Face
- Frame-by-frame visualizations with bounding boxes and class labels
- Export results as annotated video and JSON/CSV data
- Optimized for **Mac M1** with **MPS backend**
- Modular Python design for extension and experimentation

---

## Model Architectures Used
- YOLOv8n / YOLOv8m	Ultralytics	
- DETR-ResNet50	Hugging Face Transformer Model	
- SSD MobileNet V2

## Tech Stack Used
- Python 3.10
- PyTorch
- TensorFlow
- Hugging Face Transformers
- TensorFlow Hub
- YOLOv8 (Ultralytics)
- DETR (facebook/detr-resnet-50)
- SSD MobileNet V2
- OpenCV
- Pillow (PIL)
- NumPy
- Matplotlib
- tqdm
- Apple Metal Performance Shaders (MPS)
- Conda
- Jupyter Notebook (optional)

## Mac M1 Optimization
All scripts check for MPS (Metal Performance Shaders) support and default to device='mps' if available. This ensures smooth and accelerated inference on MacBooks with Apple Silicon chips.

## Outputs
Each script generates:
- Annotated video frames
- Combined output video
- (Optional) JSON/CSV file with detected labels, boxes, and scores

## Code Breakdown

## **webcam_detection.py**

This code implements a real-time object detection system using the YOLO (You Only Look Once) model, specifically the YOLOv8n variant, with the help of the Ultralytics library. The code is structured within a class called DETRClass, which handles video capture, model loading, object detection, and visualization of results. Here’s a detailed breakdown of what each part of the code does:

**Imports**

- torch: The PyTorch library, used for tensor computations and model handling.
- numpy: A library for numerical operations, particularly for handling arrays.
- cv2: OpenCV library for image and video processing.
- time: A module to measure time intervals, useful for calculating frames per second (FPS).
- YOLO: The YOLO class from the Ultralytics library, which provides pre-trained models for object detection.

Class Definition: DETRClass
This class encapsulates the functionality for object detection.

**1. Init Method**

- **Parameters:** capture_index specifies the index of the video capture device (e.g., webcam).
- **Device Selection:**
    - self.device is set to use the Metal Performance Shaders (MPS) backend if available (for Mac M1), otherwise it defaults to CPU.
    - It prints the device being used.
- **Model Loading:**
    - The code attempts to load the YOLOv8n model from a file named yolov8n.pt.
    - If loading fails (e.g., the model file is not found), it catches the exception, prints an error message, and attempts to download the model automatically using YOLO('yolov8n').
- **Class Names:**
    - self.CLASS_NAMES_DICT stores the class names that the model can detect, which are printed to the console.

**2. Plot Bboxes Method**

- **Parameters**: Takes results (the output from the model) and frame (the current video frame).
- **Extract Detections**:
    - Retrieves bounding boxes, class IDs, confidences, and coordinates from the model's results.
- **Draw Bounding Boxes**:
    - For each detected object, it extracts the coordinates of the bounding box and draws a rectangle around the detected object using OpenCV.
    - It also creates a label that includes the class name and confidence score, which is displayed above the bounding box.
- **Return**: The modified frame with bounding boxes and labels drawn.

**3. Call Method**

- This method allows the class instance to be called like a function.
- **Video Capture:**
    - Initializes video capture from the specified index and sets the frame dimensions to 1280x720 pixels.
- **Main Loop:**
    - Continuously captures frames from the video:
        - **Frame Capture:** Reads a frame from the video. If reading fails, it breaks the loop.
        - **Object Detection:** Uses the model to predict objects in the current frame.
        - **Draw Bounding Boxes:** Calls plot_bboxs to draw the detected objects on the frame.
        - **FPS Calculation:** Measures the time taken to process the frame and calculates the frames per second (FPS).
        - **Display Frame:** Shows the processed frame in a window titled "DETR" with the FPS displayed.
        - **Exit Condition:** The loop breaks if the 'q' key is pressed.
- Cleanup: Releases the video capture and closes any OpenCV windows.

**4. Initialization and Execution**

- An instance of DETRClass is created with capture_index set to 0, which typically refers to the default webcam.
- The instance is called to start the object detection process.

**Summary**

Overall, this code sets up a real-time object detection system that captures video from a webcam, processes each frame to detect objects using a YOLO model, and displays the results with bounding boxes and labels in a window. The use of the MPS backend allows for optimized performance on compatible hardware. The program is designed to be user-friendly, allowing for easy termination by pressing the 'q' key.

## **video_detection.py**

This code implements a simple object detection system using the YOLO (You Only Look Once) model from the Ultralytics library. It processes a video file and displays the detected objects with bounding boxes and class labels in real-time. Here’s a detailed breakdown of what each part of the code does:

**1. Imports**

- cv2: The OpenCV library, used for image and video processing.
- YOLO: The YOLO class from the Ultralytics library, which provides pre-trained models for object detection.
- numpy: A library for numerical operations, particularly for handling arrays.

**2. Video Capture**

- **cap = cv2.VideoCapture():** This line initializes video capture from the specified video file. The cap object will be used to read frames from the video.

**3. Model Loading**

- **model = YOLO():** This line loads the YOLOv8 model weights. This model will be used for detecting objects in the video frames.

**4. Main Loop**

- while True:: This starts an infinite loop that will continue until explicitly broken.

**5. Frame Reading**

- **ret, frame = cap.read() Reads a frame from the video.
    - ret is a boolean that indicates whether the frame was successfully read.
    - frame contains the actual image data of the frame.
- **if not ret: break:** If the frame was not successfully read (e.g., end of the video), the loop breaks.

**6. Object Detection**

- **results = model(frame, device="mps"):** This line passes the current frame to the YOLO model for object detection. The device="mps" argument specifies that the Metal Performance Shaders (MPS) backend should be used for computation (typically for Mac M1 devices).
- **result = results[0]:** Extracts the first result from the model's output. The model can return multiple results, but in this case, we are only interested in the first one.
- **bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int"):** Retrieves the bounding box coordinates (in the format [x1, y1, x2, y2]) for detected objects and converts them to a NumPy array of integers.
- **classes = np.array(result.boxes.cls.cpu(), dtype="int"):** Retrieves the class IDs for the detected objects and converts them to a NumPy array of integers.

**7. Drawing Bounding Boxes and Labels**

- **for cls, bbox in zip(classes, bboxes)::** Iterates over the detected classes and their corresponding bounding boxes.
- **x, y, x2, y2 = bbox:** Unpacks the bounding box coordinates.
- **cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)**: Draws a rectangle around the detected object on the frame. The rectangle is colored in blue (BGR format) with a thickness of 2 pixels.
- **cv2.putText(frame, str(cls), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 225), 2):** Places the class label above the bounding box. The label is drawn in blue with a font size of 2.

**8. Displaying the Frame**

- **cv2.imshow("Img", frame):** Displays the current frame with the drawn bounding boxes and labels in a window titled "Img".

**9. Exit Condition**

- **key = cv2.waitKey(1):** Waits for 1 millisecond for a key press. This allows the window to refresh and display the current frame.
- **if key == 27: break:** If the 'Esc' key (ASCII code 27) is pressed, the loop breaks, ending the video processing.

**10. Cleanup**

- **cap.release():** Releases the video capture object, freeing up resources.
- **cv2.destroyAllWindows():** Closes all OpenCV windows that were opened during the execution.

## **object_detection_tf.py**

This code implements an object detection system using a pre-trained model from TensorFlow Hub, specifically the SSD MobileNet V2 model. It processes a video file, detects objects in each frame, and saves the output video with bounding boxes and labels around detected objects. Here’s a detailed breakdown of what each part of the code does:

**1. Imports**

- os: Provides functions for interacting with the operating system, such as file path manipulation.
- cv2: The OpenCV library, used for image and video processing.
- tensorflow: The TensorFlow library, used for machine learning and deep learning tasks.
- tensorflow_hub: A library for loading pre-trained models from TensorFlow Hub.
- numpy: A library for numerical operations, particularly for handling arrays.
- datetime: A module for working with dates and times, used for timestamping the output video.

**2. Configuration**

- VIDEO_NAME: The name of the video file to be processed.
- CURRENT_DIR: The current working directory.
- INPUT_VIDEO_PATH: The full path to the input video file, constructed using the current directory and video name.
- OUTPUT_DIR: The path to the output directory where the processed video will be saved. It is created if it does not already exist.

**3. Function Definitions**

**load_model()**

- This function loads the SSD MobileNet V2 model from TensorFlow Hub.
- It prints messages indicating the loading status and returns the loaded model.

**4. process_frame(frame, model)**

- **Parameters:** Takes a single video frame and the loaded model as input.
- **Preprocessing:**
    - Converts the frame from BGR (OpenCV format) to RGB (TensorFlow format).
    - Expands the dimensions of the tensor to match the model's input requirements.
- **Object Detection:**
    - Passes the preprocessed frame to the model to get predictions.
    - Extracts bounding boxes, detection scores, and class IDs from the model's output.
- **Filtering Detections:**
    - Filters out detections with scores below 0.5 (i.e., only keeps detections with a confidence score of 50% or higher).
- **Drawing Results:**
    - Iterates over the valid detections and draws bounding boxes and labels on the frame.
    - The bounding box coordinates are scaled back to the original frame dimensions.
- **Return:** The modified frame with drawn bounding boxes and labels.

**5. process_video()**

- This function orchestrates the video processing.
- File Existence Check: Checks if the input video file exists. If not, it prints an error message and exits.
- **Model Loading:** Calls load_model() to load the object detection model.
- **Video Capture:** Opens the video file using OpenCV. If it fails, it prints an error message and exits.
- **Video Properties:** Retrieves the width, height, and frames per second (FPS) of the video.
- **Output Video Writer:** Creates a VideoWriter object to save the processed video. It attempts to use the 'mp4v' codec and falls back to 'XVID' if it fails.
- **Frame Processing Loop:**
    - Continuously reads frames from the video.
    - Calls process_frame() to process each frame and write the processed frame to the output video.
    - Prints progress every 10 frames.
- **Resource Cleanup:** Releases the video capture and writer resources after processing is complete and prints summary information about the processing.

**6. Main Execution**

- The script checks if it is being run as the main module and calls process_video() to start the detection process.

**Summary**

Overall, this code processes a video file, detects objects in each frame using a pre-trained SSD MobileNet V2 model, and saves the results to a new video file with bounding boxes and labels around detected objects. The program is structured to handle errors gracefully and provides feedback on the processing status.

## **object_detection_hf.py**

This code implements an object detection system that processes a video file using the DETR (DEtection TRansformer) model from the Hugging Face Transformers library. It detects objects in the video frames, visualizes the results with bounding boxes and labels, and saves the processed video. Here’s a detailed breakdown of what each part of the code does:

**1. Imports**

- cv2: OpenCV library for image and video processing.
- torch: PyTorch library for deep learning.
- numpy: Library for numerical operations, particularly for handling arrays.
- DetrImageProcessor, DetrForObjectDetection: Classes from the Hugging Face Transformers library for processing images and performing object detection with the DETR model.
- Image: PIL (Python Imaging Library) for image handling.
- tqdm: A library for creating progress bars in loops.
- os: Provides functions for interacting with the operating system, such as file path manipulation.

**2. Debug Information**

- The script prints the current working directory and lists any .mp4 files in that directory at startup.

**3. Class Definition: VideoObjectDetector**

This class encapsulates the functionality for object detection in video frames.

**4. init Method**

- **Parameters:** model_name specifies the pre-trained model to use (default is "facebook/detr-resnet-50").
- **Device Selection:** Checks if the Metal Performance Shaders (MPS) backend is available (for M1 Macs) and sets the device accordingly.
- **Model and Processor Loading:** Loads the DETR model and its associated image processor from Hugging Face.

**5. get_video_properties(video_path)**

- Parameters: Takes the path to the video file.
- File Existence Check: Checks if the video file exists and raises an error if it does not.
- Video Capture: Opens the video file using OpenCV and retrieves its properties (width, height, FPS, and frame count).
- Return: Returns the video properties.

**6. detect_objects(image)**

- Parameters: Takes a single video frame (as a NumPy array).
- Image Conversion: Converts the NumPy array to a PIL Image.
- Image Processing: Uses the processor to prepare the image for the model.
- Inference: Runs the model to detect objects in the image.
- Post-processing: Filters the results based on a confidence threshold (0.7) and prepares the results for visualization.
- Return: Returns the detection results.

**7. visualize_predictions(image, results)**

- Parameters: Takes the original image and the detection results.
- Drawing Boxes and Labels: Iterates over the detected objects and draws bounding boxes and labels on the image.
- Return: Returns the modified image with visualized predictions.

**8. Function: process_video(video_path, output_path, process_every_n_frames)**

- Parameters: Takes the input video path, output video path, and a parameter to process every nth frame.
- Detector Initialization: Creates an instance of VideoObjectDetector.
- Video Properties: Retrieves the properties of the input video.
- Video Capture and Writer: Initializes video capture and sets up a video writer to save the output.
- Frame Processing Loop:
    - Reads frames from the video.
    - Processes every nth frame to maintain performance.
    - Converts frames from BGR to RGB for processing, detects objects, visualizes predictions, and writes the processed frames to the output video.
    - Updates a progress bar for visual feedback.
- Resource Cleanup: Releases the video capture and writer resources after processing is complete.

**9. Main Execution**

- The script defines the input and output video paths based on the script's directory.
- It calls process_video() to start the detection process.
- It prints success or failure messages based on the outcome of the processing.

**Summary**

Overall, this code processes a video file, detects objects in each frame using the DETR model, visualizes the results with bounding boxes and labels, and saves the processed video. The program is structured to handle errors gracefully and provides feedback on the processing status. It also optimizes performance by processing every nth frame, which is useful for longer videos.
