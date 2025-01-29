import cv2
import numpy as np
from embeddings import extract_embedding, car_embeddings, index, store_client_car
from detection import detect_cars
from utils import draw_box

def track_client_car(client_id, video_source=0):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_cars(frame)
        for (x1, y1, x2, y2) in detections:
            car_crop = frame[y1:y2, x1:x2]
            detected_embedding = extract_embedding(car_crop)

            # Search FAISS index
            D, I = index.search(detected_embedding.reshape(1, -1), 1)
            matched_embedding = car_embeddings[client_id][I[0][0]]
            similarity = np.dot(detected_embedding, matched_embedding)

            # Determine if it's the client's car
            if similarity > 0.8:
                label = f"Client {client_id}'s Car ({similarity:.2f})"
                color = (0, 255, 0)
            else:
                label = "Not the client's car"
                color = (0, 0, 255)

            draw_box(frame, (x1, y1, x2, y2), label, color)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run tracking
if __name__ == "__main__":
    client_id = "client_123"
    client_image_path = "../dataset/"
    store_client_car(client_id, client_image_path)
    track_client_car(client_id)
