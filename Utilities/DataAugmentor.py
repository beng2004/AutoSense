#script that flips all images in dataset horiontally
#DO NOT RUN

import os
import cv2

# PATH = "C:/Users/benjamin.guerrieri/Documents/AutoSenseBackup/FinalDataset"
directorys = os.listdir(PATH)

for directory in directorys:
    cd = PATH + "/" + directory
    for file in os.listdir(cd):
        img = cv2.imread(cd + "/"+file)
        horizontal_img = cv2.flip( img, 1)

        #saving now
        cv2.imwrite(os.path.join(cd, file + '_flip' + '.jpg'), horizontal_img)


