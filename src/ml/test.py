import torchvision
import cv2
import time

# Try creating two windows
cv2.namedWindow("Window1", cv2.WINDOW_NORMAL)
time.sleep(1)  # Wait to see if first window appears
cv2.namedWindow("Window2", cv2.WINDOW_NORMAL)