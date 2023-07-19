from ultralytics import YOLO
import logging 
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
import torchvision
from torchvision import transforms, models
import math
from PIL import Image
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# Disables console logs to improve speed
logging.disable(logging.INFO)


colors = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

color_transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        # first layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, int(len(colors)))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)
color_classifier = NET()


color_classifier.load_state_dict(torch.load("./Models/color_model.pt"))


# Identifies classes to be searched for that the model was trained on
classes = ['Convertible', 'Coupe', 'Minivan', 'SUV', 'Sedan', 'Truck', 'Van']

# Declares a transformer to make all images fit the input size of the trained model
transformer = torchvision.transforms.Compose([
    transforms.Resize(size=(int(224), int(224))),
    transforms.ToTensor(),
])

# Transfer Learning Model
classifier = models.resnet34()
num_ftrs = classifier.fc.in_features
classifier.fc = nn.Linear(num_ftrs, len(classes))

# Loads the model that is trained on our data
classifier.load_state_dict(torch.load("./Models/car_model_TL.pt"))

def get_probs(output):
    probs = nn.functional.softmax(output, dim=1)
    probs = probs.tolist()
    p = max(probs[0])
    p = p * 100
    return str(format(p, '.1f'))

def predict(img):
    classifier.eval()
    color_classifier.eval()
    color_img = color_transformer(img).float()

    img_normalized = transformer(img).float()

    img_normalized = img_normalized[np.newaxis, ...]
    color_img = color_img[np.newaxis, ...]

    output = classifier(img_normalized)
    color_output = color_classifier(color_img)

    prob = get_probs(output)
    output = torch.argmax(output, 1)
    
    color_output = torch.argmax(color_output, 1)

    return str(colors[color_output]).title() + " " + str(classes[output]) + " " + prob + "%"

WINDOW_NAME = "Video Classifier"
model = YOLO('./YOLOModels/yolov8n.pt')

cap = cv2.VideoCapture("C:/Users/matthew.hui/Documents/AutoSense _old/vid.mp4")
# cap.set(cv2.CAP_PROP_POS_FRAMES, 3050)

# Fullscreen
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_detections = defaultdict(dict)  # Dictionary to store previous frame's detections for each car ID
prev_confidence_scores = defaultdict(dict)  # Dictionary to store previous frame's detections for each car ID

car_counter = 0  # Counter for assigning unique IDs to cars
threshold_distance = 50  # Threshold for matching previous and current detections
confidence_threshold = 80.0
while True:
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)

    new_detections = defaultdict(dict)  # Dictionary to store current frame's detections for each car ID
    new_confidence_scores = defaultdict(dict)
    for r in results:
        annotator = Annotator(frame)
        boxes = r.boxes

        # Create a distance matrix between previous and current detections
        num_prev_detections = len(prev_detections)
        num_current_detections = len(boxes)
        distance_matrix = np.zeros((num_prev_detections, num_current_detections))

        for i, (prev_id, prev_detection) in enumerate(prev_detections.items()):
            for j, box in enumerate(boxes):
                b = box.xyxy[0].tolist()
                current_center = np.array([(b[2] + b[0]) / 2, (b[3] + b[1]) / 2])
                prev_center = prev_detection['center']
                distance = np.linalg.norm(current_center - prev_center)
                distance_matrix[i, j] = distance

        # Assign detections using the Hungarian algorithm
        prev_indices, current_indices = linear_sum_assignment(distance_matrix)
        used_current_indices = set()

        for i, prev_idx in enumerate(prev_indices):
            current_idx = current_indices[i]
            if distance_matrix[prev_idx, current_idx] > threshold_distance:
                continue

            used_current_indices.add(current_idx)
            prev_id = list(prev_detections.keys())[prev_idx]
            car_id = prev_id
            new_detections[car_id] = prev_detections[prev_id]
            new_confidence_scores[car_id] = prev_confidence_scores[prev_id]

            # class_id = r.names[box.cls[0].item()]
            # if str(class_id) == "car" or str(class_id) == "bus" or str(class_id) == "truck":
            b = boxes[current_idx].xyxy[0].tolist()
            annotator.box_label(b, new_detections[car_id]['label'])

        # Add new detections as new cars
        for j, box in enumerate(boxes):
            if j not in used_current_indices:
                b = box.xyxy[0].tolist()
                car_id = str(car_counter)
                car_counter += 1
                im_pil = Image.fromarray(img)
                im_pil = im_pil.crop([round(x) for x in b])
                class_id = r.names[box.cls[0].item()]
                if str(class_id) == "car" or str(class_id) == "bus" or str(class_id) == "truck":
                    # if float(new_detections[car_id]['label'].split(" ")[2].split('%')[0]) < 83.0
                    c = predict(im_pil)
                    
                    new_detections[car_id] = {'label': c, 'center': np.array([(b[2] + b[0]) / 2, (b[3] + b[1]) / 2])}
                    annotator.box_label(b, new_detections[car_id]['label'])
                    # frame = cv2.rectangle(frame, b, (36,255,12), 1)
                    # cv2.putText(frame, new_detections[car_id]['label'], b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    prev_detections = new_detections  # Update the previous detections for the next frame

    frame = annotator.result()
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
