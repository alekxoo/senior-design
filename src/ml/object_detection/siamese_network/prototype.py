import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Navigate up to the directory containing 'senior-design' and 'yolov5'
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..'))
print(f"Project root directory: {project_root}")

# Add the project root directory and YOLOv5 directory to the Python path
sys.path.insert(0, project_root)
yolov5_dir = os.path.join(project_root, 'yolov5')
sys.path.insert(0, yolov5_dir)

# Print the sys.path for debugging
print(f"Current sys.path: {sys.path}")

# Print the contents of the YOLOv5 directory
print(f"Contents of {yolov5_dir}:")
for item in os.listdir(yolov5_dir):
    print(f"  {item}")

# Now try to import YOLOv5 modules
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    print("Successfully imported YOLOv5 modules.")
except ImportError as e:
    print(f"Error: Unable to import YOLOv5 modules. {e}")
    print("Please ensure YOLOv5 is installed correctly and in the right location.")
    sys.exit(1)

# Rest of your imports
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random

# Load YOLOv5 model
try:
    yolo_weights_path = os.path.join(project_root, 'yolov5s.pt')
    yolo_model = attempt_load(yolo_weights_path)
    yolo_model.eval()
    print(f"Successfully loaded YOLOv5 model from {yolo_weights_path}")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    print(f"Attempted to load weights from: {yolo_weights_path}")
    sys.exit(1)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True)
        )

        # Calculate the size of the output from the last convolutional layer
        self.fc_input_dims = self._get_conv_output((3, 224, 224))

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_dims, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256)
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.cnn1(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2=None):
        output1 = self.forward_once(input1)
        if input2 is not None:
            output2 = self.forward_once(input2)
            return output1, output2
        return output1

siamese_model = SiameseNetwork()
siamese_model_path = os.path.join(project_root, 'best_siamese_model.pth')
try:
    siamese_model.load_state_dict(torch.load(siamese_model_path))
    siamese_model.eval()
    print(f"Successfully loaded Siamese model from {siamese_model_path}")
except Exception as e:
    print(f"Error loading Siamese model: {e}")
    print(f"Attempted to load model from: {siamese_model_path}")
    sys.exit(1)

# Preprocess function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

# Load and preprocess the reference car images
reference_cars = {}
dataset_dir = os.path.join(project_root, 'vehicle_images_vault')
print(f"Looking for dataset in: {dataset_dir}")

if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory not found at {dataset_dir}")
    sys.exit(1)

# Use a subset of images (e.g., 5 per category) as reference
for category in os.listdir(dataset_dir):
    category_dir = os.path.join(dataset_dir, category)
    if os.path.isdir(category_dir):
        image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_images = random.sample(image_files, min(5, len(image_files)))
        for image_file in selected_images:
            image_path = os.path.join(category_dir, image_file)
            car_image = cv2.imread(image_path)
            if car_image is None:
                print(f"Warning: Unable to read image {image_path}")
                continue
            car_tensor = preprocess_image(car_image)
            with torch.no_grad():
                reference_cars[f"{category}_{image_file}"] = siamese_model.forward_once(car_tensor)

print(f"Loaded {len(reference_cars)} reference images.")

def detect_cars(frame):
    img = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img = img.to(next(yolo_model.parameters()).device)
    
    with torch.no_grad():

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=[2])  # class 2 is typically 'car' in COCO
    
    if len(pred) > 0 and len(pred[0]) > 0:
        return pred[0].cpu().numpy()
    else:
        return np.array([])

def compare_car(car_patch):
    car_tensor = preprocess_image(car_patch)
    with torch.no_grad():
        car_features = siamese_model(car_tensor)
    
    similarities = {}
    for car_id, ref_features in reference_cars.items():
        similarity = F.pairwise_distance(ref_features, car_features).item()
        similarities[car_id] = similarity
    
    return min(similarities.items(), key=lambda x: x[1])

def process_frame(frame):
    detections = detect_cars(frame)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        car_patch = frame[int(y1):int(y2), int(x1):int(x2)]
        if car_patch.size == 0:
            continue
        car_id, similarity = compare_car(car_patch)
        color = (0, 255, 0) if similarity < 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'{car_id}: {similarity:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            processed_frame = process_frame(frame)
            cv2.imshow('Frame', processed_frame)Using device: cpu
Training:   0%|                                                                                                                                                                                                                                                                                       | 0/8 [00:00<?, ?it/s]Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/TOYOTA_Rav4_2018Present/TOYOTA-Rav4-6293_95.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/TOYOTA_Rav4_2018Present/TOYOTA-Rav4-6293_15.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_11.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_43.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_74.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/LEXUS_NX_20142017/LEXUS-NX-5210_29.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_184.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/2020_Mazda_MX30/MAZDA-MX-30-6707_174.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_77.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/TOYOTA_Rav4_2018Present/TOYOTA-Rav4-6293_24.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_140.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_99.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/LEXUS_NX_20142017/LEXUS-NX-5210_52.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_173.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/2020_Mazda_MX30/MAZDA-MX-30-6707_103.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/TOYOTA_Rav4_2018Present/toyota-rav4-2018-6293_172.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_96.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_89.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/2020_Mazda_MX30/MAZDA-MX-30-6707_136.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_116.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_58.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_164.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_96.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/TOYOTA_Rav4_2018Present/TOYOTA-Rav4-6293_94.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_9.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_138.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/2020_Mazda_MX30/MAZDA-MX-30-6707_170.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/LEXUS_NX_20142017/LEXUS-NX-5210_90.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/2020_Mazda_MX30/MAZDA-MX-30-6707_115.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/AUDI_A7_Sportback_2017Present/AUDI-A7-Sportback-6152_67.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_93.jpg or /
Invalid file path: /home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault/BMW_7_Series_2022Present/bmw-7-series-2022-7253_120.jpg or /
Training:   0%|                                                                                                                                                                                                                                                                                       | 0/8 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "senior-design/src/ml/object_detection/siamese_network/siamese_train_script.py", line 267, in <module>
    main()
  File "senior-design/src/ml/object_detection/siamese_network/siamese_train_script.py", line 234, in main
    train_loss = train(model, train_loader, criterion, optimizer, device)
  File "senior-design/src/ml/object_detection/siamese_network/siamese_train_script.py", line 141, in train
    for img1, img2, label in tqdm(train_loader, desc="Training"):
  File "/home/machvision/Documents/senior_design/venv/lib/python3.8/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/home/machvision/Documents/senior_design/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/home/machvision/Documents/senior_design/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 673, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/machvision/Documents/senior_design/venv/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
  File "senior-design/src/ml/object_detection/siamese_network/siamese_train_script.py", line 216, in collate_fn
    return torch.utils.data.dataloader.default_collate(batch)
  File "/home/machvision/Documents/senior_design/venv/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py", line 317, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
  File "/home/machvision/Documents/senior_design/venv/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py", line 137, in collate
    elem = batch[0]
IndexError: list index out of range
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()