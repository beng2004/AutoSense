from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
import torchvision
from torchvision import transforms

classes =['Convertible', 'Coupe', 'Minivan', 'SUV', 'Sedan', 'Truck', 'Van']
#identifies a transformer to make all images the same size
transformer = torchvision.transforms.Compose([
    transforms.Resize(size = (int(120), int(180))),
    transforms.ToTensor(),
])

#ResNet/Best Model 80%
import torch.nn as nn
import torch
from torch import Tensor
from typing import Type

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out
    
class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
classifier = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes= len(classes))
classifier.load_state_dict(torch.load("C:/Users/benjamin.guerrieri/Documents/AutoSenseBackup/Models/car_model_no_hatch.pt"))

from PIL import Image
import numpy as np

def predict(img):
    classifier.eval()
    img_normalized = transformer(img).float()
    img_normalized = img_normalized[np.newaxis, ...]

    output = classifier(img_normalized)
    output = torch.argmax(output, 1)
    return str(classes[output])







model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("./TestVideos/TrafficVideo3.mp4")
cap.set(3, 640)
cap.set(4, 480)


prev_num_detections = -1
num_detections = 0
classify = True
while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    #only runs classification if a new car comes on the screen
    

    labels = []
    for r in results:
        #only classify once
        annotator = Annotator(frame)
        
        boxes = r.boxes

        for i, box in enumerate(boxes):

            b = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
            
            
            # new_img = img[int(b[0]):int(b[1]), int(b[2]):int(b[3])]
            
            class_id = r.names[box.cls[0].item()]
            if str(class_id) == "car" or str(class_id) == "bus" or str(class_id) == "truck":    
                print(class_id)
                im_pil = Image.fromarray(img)
                im_pil = im_pil.crop([round(x) for x in b])
                c = predict(im_pil)
                
                annotator.box_label(b, c)
        

    frame = annotator.result()  

    cv2.imshow('YOLO V8 Detection', frame)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()