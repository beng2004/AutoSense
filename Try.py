import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# Constants
colors = ['Beige', 'Black', 'Blue', 'Brown', 'Gold', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'Tan', 'White', 'Yellow']
body_types = ['Convertible', 'Coupe', 'Minivan', 'SUV', 'Sedan', 'Truck', 'Van']
const_title = "AutoSense"
background_color = "#FFFFFF"
text_color = "#000000"

# Helper function to get car probabilities
def get_probs(output):
    probs = torch.nn.functional.softmax(output, dim=1)
    probs = probs.tolist()
    p = max(probs[0])
    p = p * 100
    return str(format(p, '.1f'))

# Helper function for car prediction
def predict_car(img, classifier, color_classifier):
    classifier.eval()
    color_classifier.eval()
    color_img = color_transformer(img).float()
    img_normalized = transformer(img).float()

    img_normalized = img_normalized.unsqueeze(0)
    color_img = color_img.unsqueeze(0)

    output = classifier(img_normalized)
    color_output = color_classifier(color_img)

    prob = get_probs(output)
    output = torch.argmax(output, 1)
    color_output = torch.argmax(color_output, 1)

    return str(colors[color_output.item()]).title(), str(body_types[output.item()]), prob + "%"

# Helper function for object detection and tracking
def detect_and_track(model, frame):
    global prev_detections, prev_confidence_scores, car_counter

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)

    # Rest of the code for object detection and tracking...

# Set up GUI
class AutoSenseApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(const_title)
        self.configure(background=background_color)

        self.file = ""
        self.video_label = tk.Label(text="Current video selected: ", background=background_color, fg=text_color)
        self.video_label.pack()

        self.color_var = tk.StringVar(self)
        self.color_var.set(colors[0])  # Default color selection

        self.body_var = tk.StringVar(self)
        self.body_var.set(body_types[0])  # Default body type selection

        self.color_menu = tk.OptionMenu(self, self.color_var, *colors)
        self.color_menu.pack()

        self.body_menu = tk.OptionMenu(self, self.body_var, *body_types)
        self.body_menu.pack()

        self.display_label = tk.Label()
        self.display_label.pack()

        self.video_select_btn = tk.Button(text='SELECT VIDEO', command=self.prompt_video, background=background_color, fg=text_color)
        self.video_select_btn.pack()

        self.detect_btn = tk.Button(text='BEGIN DETECTION', command=self.detect, background=background_color, fg=text_color)
        self.detect_btn.pack()

        # Initialize the YOLO model
        self.model = YOLO('./YOLOModels/yolov8n.pt')

        # Load the color classification model
        self.color_transformer = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.color_classifier = self.load_model_color('./Models/color_model.pt')

        # Load the body type classification model
        self.transformer = torchvision.transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ])
        self.classifier = self.load_model_body('./Models/car_model_TL.pt')

    def load_model_color(self, model_path):
        color_classifier = NET()
        color_classifier.load_state_dict(torch.load(model_path))
        return color_classifier

    def load_model_body(self, model_path):
        classifier = models.resnet34()
        num_ftrs = classifier.fc.in_features
        classifier.fc = torch.nn.Linear(num_ftrs, len(body_types))
        classifier.load_state_dict(torch.load(model_path))
        return classifier

    def prompt_video(self):
        self.file = filedialog.askopenfilename(initialdir="c:/Users/matthew.hui/Documents/AutoSense", title="Select Video", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        if self.file:
            split = self.file.split('/')[-2:]
            self.video_label.configure(text="Current video selected: .../" + split[0] + "/" + split[1])
            self.show_frame()

    def detect(self):
        if self.file:
            self.show_frame()

    def show_frame(self):
        try:
            cap = cv2.VideoCapture(self.file)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_with_detection = detect_and_track(self.model, frame)

                # Convert the frame with detection results back to PIL and display
                display_img = Image.fromarray(frame_with_detection)
                imgtk = ImageTk.PhotoImage(image=display_img)
                self.display_label.imgtk = imgtk
                self.display_label.configure(image=imgtk)

                self.update_idletasks()
        except Exception as e:
            print("Error:", e)
        finally:
            cap.release()

if __name__ == "__main__":
    app = AutoSenseApp()
    app.mainloop()
