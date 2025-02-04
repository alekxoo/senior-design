import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ultralytics import YOLO  # YOLOv9
import yaml
import warnings
import torch
import torch.nn as nn
import torchvision.models as models

warnings.filterwarnings("ignore", category=FutureWarning)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load pre-trained ResNet18 and remove the fully connected layer (fc)
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Exclude fc layer

        # Add a new fully connected layer for the output
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 has 512 features at the final layer before fc
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # This will give us the similarity score
        )

    def forward_one(self, x):
        """Pass a single image through the network to extract features."""
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        return x

    def forward(self, input1, input2):
        """Pass two images through the network, and compute the similarity score."""
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        distance = torch.abs(output1 - output2)  # Calculate L1 distance (can also try L2)
        output = self.fc(distance)  # Feed it through the fully connected layers
        return output

def load_yaml(file_path):
    """ Load class labels and number of classes from YAML file. """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def parse_class_data(data):
    """ Extract class labels and count from YAML data. """
    class_labels = [cls['label'] for cls in data['classes']]
    num_classes = data['num_classes']
    return class_labels, num_classes

def load_siamese_model(model_path, device):
    """ Load the trained Siamese model. """
    try:
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Siamese classification model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Siamese classification model: {e}")
        return None

def load_yolov9_model(model_path):
    """ Load YOLOv9 model. """
    try:
        yolov9_model = YOLO(model_path)  # Load YOLOv9 model
        print("YOLOv9 model loaded successfully.")
        return yolov9_model
    except Exception as e:
        print(f"Error loading YOLOv9 model: {e}")
        return None

def generate_reference_embeddings(model, reference_images, transform, device):
    """ Generate embeddings for reference images of each class. """
    embeddings = {}

    for class_name, image_path in reference_images.items():
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.forward_one(image)  # Extract feature vector
            embeddings[class_name] = embedding

    print("Reference embeddings generated.")
    return embeddings

def infer_webcam(yolov9_model, classification_model, device, reference_embeddings, transform):
    """ Perform real-time inference using YOLOv9 and Siamese Network. """
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            results = yolov9_model.predict(img_rgb, classes=[2])  # Detect cars (YOLOv9 class index for 'car')
            annotated_frame = results[0].plot()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()

                    if conf > 0.7:  # Ensure high confidence
                        roi = img_rgb[y1:y2, x1:x2]
                        roi_pil = Image.fromarray(roi)
                        roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            query_embedding = classification_model.forward_one(roi_tensor)  # Get feature vector

                            best_match = None
                            best_score = float("inf")

                            for class_name, ref_embedding in reference_embeddings.items():
                                score = F.pairwise_distance(query_embedding, ref_embedding).item()
                                if score < best_score:
                                    best_score = score
                                    best_match = class_name

                        label = f"{best_match} ({conf:.2f})"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow("YOLOv9/Siamese Test Interface", annotated_frame)

        except Exception as e:
            print(f"Error during inference: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """ Main function to load models, generate embeddings, and start inference. """
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    yolov9_model_path = "./config/yolov9c.pt"
    siamese_model_path = "./CNNModels/siamese_best.pt"

    # Load YOLOv9 model
    yolov9_model = load_yolov9_model(yolov9_model_path)
    classification_model = load_siamese_model(siamese_model_path, device)

    if yolov9_model is None or classification_model is None:
        print("Failed to load one or more models.")
        return

    # Define transformation for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Reference images for each class
    reference_images = {
        "bmw": "./test/testBMW.jpg",
        "lexus": "./test/testLexus.webp",
        "toyota": "./test/testT.webp"
    }

    reference_embeddings = generate_reference_embeddings(classification_model, reference_images, transform, device)

    infer_webcam(yolov9_model, classification_model, device, reference_embeddings, transform)

if __name__ == '__main__':
    main()
