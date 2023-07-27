import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.font import Font
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
font = "Calibri"
background = "#FFFFFF"
prev_width, prev_height = 0, 0
first_frame = True

# Create the tkinter application
class VideoClassifierApp:
    def __init__(self, root):
        # Fullscreen
        # root.attributes("-fullscreen", True)

        self.root = root
        self.root.title(const_title)
 
        main_container = tk.Frame(root)
        # main_container.pack(fill='both', expand=True, padx=(60, 0), pady=40)
        main_container.place(relx=0.5, rely=0.5, anchor=CENTER)
        # main_container.pack_propagate(False)

        self.title = tk.Label(main_container, text=const_title)
        self.title.config(font=(font, 35))
        self.title.pack(anchor='w')

        self.video_file = ""
        self.selected_video = tk.Label(main_container, text="Current video selected:    ")
        self.selected_video.config(font=(font, 14))
        self.selected_video.pack(anchor='w')

        self.image_file = ""
        self.selected_image = tk.Label(main_container, text="Current image selected:    ")
        self.selected_image.config(font=(font, 14))
        self.selected_image.pack(anchor='w')

        content_container = tk.Frame(main_container)
        content_container.pack(anchor='w')

        self.background = tk.Label(content_container, background="#A9B0BA", width=120, height=40)
        # self.background.configure(image=ImageTk.PhotoImage(Image.open('C:/Users/matthew.hui/Documents/AutoSense/Utilities/blackBackground.png')))
        self.background.pack(fill=tk.BOTH, expand=True, side='left', padx=(0, 30), pady=20)
        self.background.pack_propagate(False)
        # self.background.pack(anchor='w', side='left', padx=(0, 50), pady=20)

        right_container = tk.Frame(content_container)
        right_container.pack(anchor='e', side='left')

        self.initial_color = StringVar()
        self.initial_color.set(colors[2]) # Default value of Blue
        self.initial_body = StringVar()
        self.initial_body.set(body_types[2])

        self.color_menu = OptionMenu(right_container, self.initial_color, *colors)
        self.color_menu.config(font=(font, 12))
        self.color_menu.pack(pady=10)

        self.body_menu = OptionMenu(right_container, self.initial_body, *body_types)
        self.body_menu.config(font=(font, 12))
        self.body_menu.pack(pady=10)

        self.start_detection = Button(right_container, text='STOP DETECTION', command=self.stop_update)
        self.start_detection.config(font=(font, 12))
        self.start_detection.pack(pady=10)

        self.model = YOLO('./YOLOModels/yolov8n.pt')

        self.video_select = Button(main_container, text='SELECT VIDEO', command=self.prompt_video)
        self.video_select.config(font=(font, 12))
        self.video_select.pack(side='left', padx=(0, 20))
        
        self.img_select = Button(main_container, text='SELECT IMAGE', command=self.prompt_image)
        self.img_select.config(font=(font, 12))
        self.img_select.pack(side='left', padx=(0, 20))

        self.prev_detections = defaultdict(dict)
        self.car_counter = 0
        self.threshold_distance = 160
        self.loop_update = False

    def prompt_video(self):
        self.video_file = filedialog.askopenfilename(initialdir="c:/Users/matthew.hui/Documents/AutoSense", title="Select Video", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        print(self.video_file)
        
        split = self.video_file.split('/')[-2:]
        file_dir = "Current video selected: " + (".../" + split[0] + "/" + split[1])
        self.selected_video.configure(text=file_dir)
        # time.sleep(2.5)
        self.cap = cv2.VideoCapture(self.video_file)
        self.loop_update = True
        self.update()

    def prompt_image(self):
        self.image_file = filedialog.askopenfilename(initialdir="c:/Users/matthew.hui/Documents/AutoSense", title="Select Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
        split = self.image_file.split('/')[-2:]
        file_dir = "Current image selected: " + (".../" + split[0] + "/" + split[1])
        self.selected_image.configure(text=file_dir)

        self.process_image()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    # Pass in a PIL image
    def resize_with_aspect_ratio(self, image, desired_height):
        width, height = image.size 
        ratio = width // height
        new_width = ratio * desired_height

        return image.resize((new_width, desired_height), Image.ANTIALIAS)

    def process_image(self):
        img = cv2.imread(self.image_file)
        img_height, img_width, _ = img.shape

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(frame)

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

        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        processed_img = Image.fromarray(bgr_img).resize((img_width, img_height), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image=processed_img)
        self.background.config(image=self.photo, width=img_width, height=img_height)

    # def stop_update(self):
    #     print("stopping update")
    #     if self.video_file:
    #         self.loop_update = False

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
                    target_car = " ".join(str(x) for x in new_detections[car_id]['label'].split()[0:2]).upper() #COLOR BODY

                    
                    if target_car == self.initial_color.get().upper() + " " + self.initial_body.get().upper():
                        annotator.box_label(b, new_detections[car_id]['label'], color=(200,0,0))
                    else:
                        annotator.box_label(b, new_detections[car_id]['label'] )
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
                            target_car = " ".join(str(x) for x in new_detections[car_id]['label'].split()[0:2]).upper()

                            if target_car == self.initial_color.get().upper() + " " + self.initial_body.get().upper():
                                annotator.box_label(b, new_detections[car_id]['label'], color=(200,0,0))
                            else:
                                annotator.box_label(b, new_detections[car_id]['label'] )

                self.prev_detections = new_detections
                frame = annotator.result()

            # picsize = self.canvas.size
            global first_frame, prev_width, prev_height
            if first_frame:
                prev_width = int(root.winfo_width() // 1.5)
                prev_height = int(root.winfo_height() // 1.5)
                first_frame = False

            screen_width = int(root.winfo_width() // 1.5) if prev_width != int(root.winfo_width() // 1.5) else prev_width
            screen_height = int(root.winfo_height() // 1.5) if prev_height !=  int(root.winfo_height() // 1.5) else prev_height
            # screen_width = 900
            # screen_height = 400
            # processed_frame = self.resize_with_aspect_ratio(Image.fromarray(frame), 400)
            processed_frame = Image.fromarray(frame).resize((screen_width, screen_height), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(image=processed_frame)
            
            # self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.background.configure(image=self.photo, height=screen_height, width=screen_width)

        global update_id
        update_id = self.root.after(10, self.update)

    def stop_update(self):
        print("STOPPING UPDATE")
        if update_id:
            self.root.after_cancel(update_id)

    def close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoClassifierApp(root)
    root.bind("<space>", VideoClassifierApp.stop_update)
    root.mainloop()
