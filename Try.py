import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.yolo.utils.plotting import Annotator
from ultralytics import YOLO
import torch.nn as nn
import torch
import torchvision.transforms as transforms

# ... (the rest of your code remains the same)

# Create the tkinter application
class VideoClassifierApp:
    def __init__(self, root, video_source):
        self.root = root
        self.root.title("Video Classifier")
        self.video_source = video_source
        self.model = YOLO('./YOLOModels/yolov8n.pt')
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Fullscreen
        root.attributes("-fullscreen", True)

        self.prev_detections = defaultdict(dict)
        self.car_counter = 0
        self.threshold_distance = 50

        self.cap = cv2.VideoCapture(self.video_source)
        self.update()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

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

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update)

    def close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoClassifierApp(root, "C:/Users/matthew.hui/Documents/AutoSense _old/vid.mp4")
    root.mainloop()
