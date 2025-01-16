import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov9c.pt")

cap = cv2.VideoCapture(0) # continuously captures frames in the background

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Test Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read() # grabs the most recent frame from the buffer
    if not ret:
        print("Error: Could not read frame.")
        break
    if ret: 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, classes = [2])
        annotated_frame = results[0].plot() # returns numpy array 

        print("Regular Frame:", frame.dtype, frame.shape, frame.flags['C_CONTIGUOUS'])
        print("Annotated frame:", annotated_frame.dtype, annotated_frame.shape, annotated_frame.flags['C_CONTIGUOUS'])
        cv2.imshow("Test Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
