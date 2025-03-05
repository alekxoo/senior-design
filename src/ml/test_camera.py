import cv2
from object_detection.video_processor import VideoProcessor
import time

def main():
    # Force window creation before capture
    cv2.namedWindow('Car Detection', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Car Detection', 0, 0)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    processor = VideoProcessor()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                            
            cv2.imshow('Car Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Now process the frame
            start_time = time.time()
            detections, annotated_frame = processor.process_frame(frame)
                        
            # Calculate FPS
            # fps = 1 / (time.time() - start_time)
            # cv2.putText(annotated_frame, 
            #            f"FPS: {fps:.1f}", 
            #            (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 
            #            1, 
            #            (0, 255, 0), 
            #            2)
            
            # Show the processed frame
            cv2.imshow('Car Detection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
    # finally:
    #     print("Cleaning up...")
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     # Force destroy on Linux
    #     for i in range(4):
    #         cv2.waitKey(1)

if __name__ == "__main__":
    main()