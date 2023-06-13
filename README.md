# Object Tracking using Faster R-CNN

This repository contains a Python script that implements object tracking using the Faster R-CNN (Region-based Convolutional Neural Network) model. The script utilizes the PyTorch library and the torchvision package for computer vision tasks.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- OpenCV
- NumPy

## Installation
1. Install Python 3.x from the official Python website: [Python Downloads](https://www.python.org/downloads/)
2. Install PyTorch and torchvision by following the instructions provided at: [PyTorch - Get Started](https://pytorch.org/get-started/locally/)
3. Install OpenCV using pip:
   ```
   pip install opencv-python
   ```
4. Install NumPy using pip:
   ```
   pip install numpy
   ```

## Usage
1. Place the video files you want to process in the `trimmed` folder.
2. Run the script using the following command:
   ```
   python object_tracking.py
   ```
3. The processed videos will be saved in the `processed_over` folder.
4. The script currently processes only the first 6 videos. You can modify the code to process more or all videos by changing the value of `i` in the code.

## Description
The code performs the following steps:

1. Loads the pre-trained Faster R-CNN model using `torchvision.models.detection.fasterrcnn_resnet50_fpn()`.
2. Sets the colors for drawing bounding boxes around objects detected in the frames.
3. Defines the paths to the input videos folder (`trimmed`) and the output videos folder (`processed_over`).
4. Creates the output videos folder if it doesn't already exist.
5. Retrieves the list of video files in the input folder.
6. Processes each video file:
   - Sets the color for drawing the bounding box for the current video.
   - Creates a frames folder for the current video.
   - Opens the video file using `cv2.VideoCapture`.
   - Reads the frames from the video and saves each frame as an image in the frames folder.
   - Releases the video capture object.
   - Gets the properties of the first frame (height, width, channels).
   - Defines the output video writers for the current video.
   - Processes each frame:
     - Reads the frame from the frames folder.
     - Preprocesses the frame using torchvision transforms.
     - Performs inference using the pre-trained Faster R-CNN model.
     - Filters the predictions to keep only the detections with a label of 37 (ball) and a score higher than 0.5.
     - Draws bounding boxes and circles around the detected objects.
     - Writes the frame with detections to the output videos.
   - Releases the video writers.

## Output
The code generates three types of output videos for each input video file:
1. `output_[video_file].mp4`: This video shows the original frames with bounding boxes around the detected objects.
2. `output_[video_file]_line.mp4`: This video shows the motion tracking of the detected objects as lines connecting their positions in consecutive frames.
3. `output_multiline_4.mp4`: This video combines the motion tracking of all processed videos by overlaying them on a single frame.

## Note
- The code currently processes only the first 6 videos. You can modify the code to process more or all videos by changing the value of `i` in the code.
- The code assumes that the `trimmed` and `processed_over` folders already exist. If they don't exist, the code will create them.
