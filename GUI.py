import tkinter as tk
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import logging 
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms, models
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import time

# Disables console logs to improve speed
logging.disable(logging.INFO)

colors = ['Beige', 'Black', 'Blue', 'Brown', 'Gold', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'Tan', 'White', 'Yellow']
body_types = ['Convertible', 'Coupe', 'Minivan', 'SUV', 'Sedan', 'Truck', 'Van']

color_transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transformer = torchvision.transforms.Compose([
    transforms.Resize(size=(int(224), int(224))),
    transforms.ToTensor(),
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
    
    # Transfer Learning Model
classifier = models.resnet34()
num_ftrs = classifier.fc.in_features
classifier.fc = nn.Linear(num_ftrs, len(body_types))
classifier.load_state_dict(torch.load("./Models/car_model_TL.pt"))

color_classifier = NET()
color_classifier.load_state_dict(torch.load("./Models/color_model.pt"))

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

    return str(colors[color_output]).title() + " " + str(body_types[output]) + " " + prob + "%"


const_title = "AutoSense"
font = ""
background = "#FFFFFF"
text_background = "#FFFFFF" 

WINDOW_NAME = "Video Scanner"

# Create the tkinter application
class VideoClassifierApp:
    def __init__(self, root, video_source):
        self.root = root
        self.root.title("Video Classifier")

        # self.config(background=background)

        self.title = tk.Label(text=const_title, background=text_background)
        self.title.pack()

        self.counter = 0

        self.file = ""
        self.selected_file = tk.Label(text="Current video selected: ", background=text_background)
        self.selected_file.pack()

        self.initial_color = StringVar()
        self.initial_color.set(colors[2]) # Default value of Blue

        self.color_menu = OptionMenu(self.root, self.initial_color, *colors)
        self.color_menu.pack()

        self.initial_body = StringVar()
        self.initial_body.set(body_types[2])

        self.body_menu = OptionMenu(self.root, self.initial_body, *body_types)
        self.body_menu.pack()

        self.video_source = video_source
        self.model = YOLO('./YOLOModels/yolov8n.pt')

        black = Image.open('C:/Users/matthew.hui/Documents/AutoSense/Utilities/blackBackground.png')
        blacker = ImageTk.PhotoImage(black)
        self.background = tk.Label(self.root, background="#000000")
        # self.background.configure(image=ImageTk.PhotoImage(Image.open('C:/Users/matthew.hui/Documents/AutoSense/Utilities/blackBackground.png')))
        self.background.pack(fill=tk.BOTH, expand=True)

        self.video_select = Button(self.root, text='SELECT VIDEO', command=self.prompt_video, background=text_background)
        self.video_select.pack()

        self.start_detection = Button(self.root, text='BEGIN DETECTION', command=self.detect, background=text_background)
        self.start_detection.pack()

        # Fullscreen
        # root.attributes("-fullscreen", True)

        self.prev_detections = defaultdict(dict)
        self.car_counter = 0
        self.threshold_distance = 50

    def prompt_video(self):
        self.file = filedialog.askopenfilename(initialdir="c:/Users/matthew.hui/Documents/AutoSense", title="Select Video", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        split = self.file.split('/')[-2:]
        self.file_dir = "Current video selected: " + (".../" + split[0] + "/" + split[1])
        self.selected_file.configure(text=self.file_dir)
        # time.sleep(2.5)
        self.cap = cv2.VideoCapture(self.video_source)
        self.update()

    def detect():
        print("detecting")

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    # Pass in a PIL image
    def resize_with_aspect_ratio(self, image, desired_height):
        width, height = image.size 

        # if width > max_width or height > max_height:
        #     ratio = min(max_width / width, max_height / height)
        #     new_width = int(width * ratio)
        #     new_height = int(width * ratio)

        ratio = width // height
        new_width = ratio * desired_height

        return image.resize((new_width, desired_height), Image.ANTIALIAS)
        # return image

    def update(self):
        frame = self.get_frame()

        if frame is not None:
            img = Image.fromarray(frame)
            results = self.model.predict(frame)

            new_detections = defaultdict(dict)
            for r in results:
                annotator = Annotator(frame)
                boxes = r.boxes

                num_prev_detections = len(self.prev_detections)
                num_current_detections = len(boxes)
                distance_matrix = np.zeros((num_prev_detections, num_current_detections))

                for i, (prev_id, prev_detection) in enumerate(self.prev_detections.items()):
                    for j, box in enumerate(boxes):
                        b = box.xyxy[0].tolist()
                        current_center = np.array([(b[2] + b[0]) / 2, (b[3] + b[1]) / 2])
                        prev_center = prev_detection['center']
                        distance = np.linalg.norm(current_center - prev_center)
                        distance_matrix[i, j] = distance

                prev_indices, current_indices = linear_sum_assignment(distance_matrix)
                used_current_indices = set()

                for i, prev_idx in enumerate(prev_indices):
                    current_idx = current_indices[i]
                    if distance_matrix[prev_idx, current_idx] > self.threshold_distance:
                        continue

                    used_current_indices.add(current_idx)
                    prev_id = list(self.prev_detections.keys())[prev_idx]
                    car_id = prev_id
                    new_detections[car_id] = self.prev_detections[prev_id]

                    b = boxes[current_idx].xyxy[0].tolist()
                    annotator.box_label(b, new_detections[car_id]['label'])

                for j, box in enumerate(boxes):
                    if j not in used_current_indices:
                        b = box.xyxy[0].tolist()
                        car_id = str(self.car_counter)
                        self.car_counter += 1
                        im_pil = Image.fromarray(frame)
                        im_pil = im_pil.crop([round(x) for x in b])
                        class_id = r.names[box.cls[0].item()]
                        if str(class_id) == "car" or str(class_id) == "bus" or str(class_id) == "truck":
                            c = predict(im_pil)

                            new_detections[car_id] = {'label': c, 'center': np.array([(b[2] + b[0]) / 2, (b[3] + b[1]) / 2])}
                            annotator.box_label(b, new_detections[car_id]['label'])

                self.prev_detections = new_detections
                frame = annotator.result()

            # picsize = self.canvas.size
            screen_width = int(root.winfo_width() // 1.5)
            screen_height = int(root.winfo_height() // 1.5)
            # processed_frame = self.resize_with_aspect_ratio(Image.fromarray(frame), 400)
            processed_frame = Image.fromarray(frame).resize((screen_width, screen_height), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(image=processed_frame)
            
            # self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.background.configure(image=self.photo, height=screen_height, width=screen_width)

        self.root.after(20, self.update)

    def close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoClassifierApp(root, "C:/Users/matthew.hui/Documents/AutoSense _old/20230711_122653.mp4")
    root.mainloop()
