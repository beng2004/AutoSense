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

color_classifier.load_state_dict(torch.load("C:/Users/benjamin.guerrieri/Documents/AutoSenseBackup/Models/color_model.pt"))

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
classifier.load_state_dict(torch.load("C:/Users/benjamin.guerrieri/Documents/AutoSenseBackup/Models/car_model_TL.pt"))

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
path = r'C:\Users\benjamin.guerrieri\Documents\AutoSenseBackup\SingleTestImages\NotCropped\5cars.jpg'

img = cv2.imread(path)

# cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_FREERATIO)
# cv2.setWindowProperty(WINDOW_NAME, cv2.WINDOW_AUTOSIZE, cv2.WINDOW_FREERATIO)
# cv2.imshow(WINDOW_NAME, img)

frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model.predict(frame)

for r in results:
    annotator = Annotator(img)
    boxes = r.boxes

    for j, box in enumerate(boxes):
        b = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
        class_id = r.names[box.cls[0].item()]
        if str(class_id) == "car" or str(class_id) == "bus" or str(class_id) == "truck":    
            im_pil = Image.fromarray(frame)
            im_pil = im_pil.crop([round(x) for x in b])
            corner = [round(b[0]), round(b[1])]
            c = predict(im_pil)

    
    
            annotator.box_label(b, c)    
            img = annotator.result()

cv2.imshow(WINDOW_NAME, img)
cv2.waitKey(0)
cv2.destroyAllWindows()

