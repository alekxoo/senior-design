import yolov5
import cv2
import numpy as np

# Load the YOLOv5 model
model = yolov5.load('./models/yolov5s.pt')

# Set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Start webcam capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Perform inference on the captured frame
    results = model(frame)

    # Get predictions and boxes
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Display detection results on the frame
    for box, score, category in zip(boxes, scores, categories):
        # Draw the bounding box on the frame
        x1, y1, x2, y2 = box.int().tolist()
        label = f'{model.names[int(category)]} {score:.2f}'
        color = (0, 255, 0)  # Green color for bounding boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('YOLOv5 Inference - Webcam', frame)

    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
