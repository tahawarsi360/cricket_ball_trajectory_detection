import torch
import torchvision
import cv2
import os
import numpy as np

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 255, 0), (100, 255, 100), (100, 100, 100)]

# Path to the input videos folder
videos_folder = 'trimmed'
# empty_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 255, 0), (100, 255, 100), (100, 100, 100)]
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Path to the output videos folder
output_folder = 'processed_over'
# Height: 576, Width: 720
empty_frame_all = np.ones((576, 720, 3), dtype=np.uint8) * 255
# Create the output videos folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of video files in the input folder
video_files = os.listdir(videos_folder)

output_video_path_for_multiline = os.path.join(output_folder, 'output_multiline_4.mp4')
output_video_multline = cv2.VideoWriter(output_video_path_for_multiline, cv2.VideoWriter_fourcc(*'mp4v'), 25, (720, 576))
i = 0
# Process each video file
for video_file in video_files:
    if i <= 5:
        color_n = COLORS[i]
    else:
        i = 2
        
    i += 1
    # print(color_n)
    # Path to the current video file
    video_path = os.path.join(videos_folder, video_file)

    # Create the frames directory for the current video if it doesn't exist
    frames_folder = os.path.join(output_folder, f'{video_file}_frames')
    os.makedirs(frames_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize frame counter
    cnt = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_review = frame

        if ret:
            # Save the frame as an image in the frames folder
            frame_path = os.path.join(frames_folder, f'{cnt}.png')
            cv2.imwrite(frame_path, frame)
            cnt += 1
        else:
            break

    # Release the video capture object
    cap.release()

    # Get the first frame to obtain video properties
    frame = cv2.imread(os.path.join(frames_folder, '0.png'))
    height, width, channels = frame.shape
    empty_frame = np.ones((height, width, 3), dtype=np.uint8) * 255


    # Define the output video writer
    output_video_path = os.path.join(output_folder, f'{video_file}_output.mp4')
    output_video_path_line = os.path.join(output_folder, f'{video_file}_output_line.mp4')
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    output_video_line = cv2.VideoWriter(output_video_path_line, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    # Process each frame
    tracking = []
    for i in range(cnt):
        frame_path = os.path.join(frames_folder, f'{i}.png')
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        input_image = transform(frame_rgb)
        input_image = input_image.unsqueeze(0)

        # Run the inference
        with torch.no_grad():
            predictions = model(input_image)

        # Filter the predictions to keep only the ball detections with label 37
        labels = predictions[0]['labels'].tolist()
        scores = predictions[0]['scores'].tolist()
        boxes = predictions[0]['boxes'].tolist()
        # tracking = []

        for label, box, score in zip(labels, boxes, scores):
            if label == 37 and score > 0.5:
                # print(label)
                # print(box)
                pred_class = COCO_INSTANCE_CATEGORY_NAMES[label]
                # pred_box = box
                x1, y1, x2, y2 = map(int, box)
                x, y = int((x1 + x2)/2), int((y1 + y2)/2)
                tracking.append([x,y])
                org = (x1, y1 - 10)
                org_label = (x1 + 10, y1 - 10)

                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for center in tracking:
                    print(center)
                # cv2.circle(frame, (center[0], center[1]), radius = 5, color = (255, 255, 255), thickness=-1)
                    cv2.circle(frame, (center[0], center[1]), radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.circle(empty_frame, (center[0], center[1]), radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.circle(frame_review, (center[0], center[1]), radius=5, color=(255, 255, 0), thickness=-1)

        # Write the frame with detections to the output video
        output_video.write(frame)
        output_video_line.write(empty_frame)
        output_video_multline.write(frame_review)

    # Release the video writer
    output_video.release()
    output_video_line.release()

output_video_multline.release()

# Destroy all opened windows
cv2.destroyAllWindows()