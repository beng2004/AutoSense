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
#disables console logs to improve speed
logging.disable(logging.INFO)

#identifies classes to be searched for that the model was trained on
classes =['Convertible', 'Coupe', 'Minivan', 'SUV', 'Sedan', 'Truck', 'Van']

#declares a transformer to make all fit the input size of the trained model
transformer = torchvision.transforms.Compose([
    transforms.Resize(size = (int(224), int(224))),
    transforms.ToTensor(),
])

#Transfer Learning Model
classifier = models.resnet34()
num_ftrs = classifier.fc.in_features
classifier.fc = nn.Linear(num_ftrs, len(classes))

#loads the model that is trained on our data
classifier.load_state_dict(torch.load("C:/Users/benjamin.guerrieri/Documents/AutoSenseBackup/Models/car_model_TL.pt"))

#necessary imports
from PIL import Image
import numpy as np
def get_probs(output):
    probs = nn.functional.softmax(output,dim=1)
    probs = probs.tolist()
    p = max(probs[0])
    p = p*100
    return str(format(p, '.1f'))

def predict(img):
    classifier.eval()
    img_normalized = transformer(img).float()
    img_normalized = img_normalized[np.newaxis, ...]

    output = classifier(img_normalized)
    prob = get_probs(output)
    output = torch.argmax(output, 1)
    return str(classes[output]) + " " + prob + "%"



WINDOW_NAME = "Video Classifier"
model = YOLO('./YOLOModels/yolov8n.pt')


cap = cv2.VideoCapture("./TestVideos/TrafficVideo.mp4")
#fullscreen
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

memory = []
#remember if it is first time running
first_time = True
append = False
num_cars = 0
while True:
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)

    for r in results:
        #only classify once
        annotator = Annotator(frame)   
        boxes = r.boxes
        
        for i, box in enumerate(boxes):
            b = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
                        
            class_id = r.names[box.cls[0].item()]
            if str(class_id) == "car" or str(class_id) == "bus" or str(class_id) == "truck":    
                # print(class_id)
                im_pil = Image.fromarray(img)
                im_pil = im_pil.crop([round(x) for x in b])
                corner = [round(b[0]), round(b[1])]
                # frame = cv2.circle(frame, (corner[0],corner[1]), radius=5, color=(0, 0, 255), thickness=-1)

                # print(b)
                if first_time:
                    c = predict(im_pil)
                    memory.append([corner, c])
                    # print(corner)

                else:
                    # print(`   corner)
                    #every other time
                    closest_corner= 100000
                    for i, row in enumerate(memory):
                        #calculate closes must check all before adding

                        if math.dist(row[0],corner) < closest_corner:
                            closest_corner = math.dist(row[0],corner)  
                            # frame = cv2.line(frame, row[0],corner,(255, 0, 255),1)
 
                            new_corner_index = i
                            new_corner = corner  

                    if math.dist(memory[new_corner_index][0], new_corner) < 13:
                        # print(math.dist(memory[new_corner_index][0], new_corner))

                        memory[new_corner_index][0] = [new_corner]
                    else:
                        if(corner[1] < 8):
                            memory.append([new_corner])


                #PSEUDOCODE
                """
                first time through: save the single top right cord from each car on the screen and store into list acting as our temporary memory
                next frame find the closest cordinates (must compare every new cordinate to every old) and update the memory to the new closest cord
                along with the cord we must insert the classification information into the correct index the memory will be a list of lists where index 0 of the list of lists is the cord and 1 is the data
                if there are any cords that have not found their closest cord that means that this cord represents a new car and we should add a new list to the memory
                if the car exits the screen bounds delete that entry from memory
                """
                # print(i)
        
                annotator.box_label(b, c + " " + str(i))

        print(memory)  
        
        first_time = False
        # print(memory)
    frame = annotator.result()  
    
    cv2.imshow(WINDOW_NAME, frame)     

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()