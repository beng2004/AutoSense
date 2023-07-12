import requests
import os
import re
import shutil
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import base64
import urllib.request
import time
from PIL import Image
from numpy import asarray
import io

# starting with the boxing script for automatic cropping
from ultralytics import YOLO
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

model = YOLO("yolov8m.pt") #nano version for faster computing

def compareImg(existing, new):
    firstImg = Image.open(existing)
    numpy1 = asarray(firstImg).ravel()
    numpy2 = asarray(new).ravel()

    for (cell1, cell2) in zip(numpy1, numpy2):
        # print(cell1, cell2)
        if cell1 != cell2:
            return False
        
    return True

def crop(img_dir):
    results = model.predict(img_dir) #
 
    result = results[0]
    if len(result.boxes) != 1: # remove if zero or multiple boxes
        os.remove(img_dir)
        return
    box = result.boxes[0]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = result.names[box.cls[0].item()]

    if str(class_id) != "car" and str(class_id) != "truck":
        print(str(class_id))
        print("Didn't detect car in: " + img_dir)
        print("Removing image ... ")
        
        os.remove(img_dir)
        return
    
    conf = round(box.conf[0].item(), 2)
   
    img= Image.open(img_dir)
    img = img.crop(cords)
    img.save(img_dir)
    return cords, class_id, conf

DRIVER_PATH = 'C:/Users/matthew.hui/Documents/code/chromedriver.exe'
options = Options()
service = Service(DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)
# Grab images
URL = 'https://www.google.com/search?q=convertible+carfax&tbm=isch&source=hp&biw=1920&bih=937&ei=aUKcZN_mMPyv5NoPopaFKA&iflsig=AOEireoAAAAAZJxQeee0hYto3Ul2GntKt1Qw-qfND-VP&ved=0ahUKEwifv4ujleb_AhX8F1kFHSJLAQUQ4dUDCAc&uact=5&oq=convertible+carfax&gs_lcp=CgNpbWcQAzIHCAAQGBCABDoFCAAQgAQ6CAgAEIAEELEDOggIABCxAxCDAVAAWMoPYJgSaABwAHgAgAGVAYgBkBKSAQQwLjE4mAEAoAEBqgELZ3dzLXdpei1pbWc&sclient=img'
# URL = "https://www.carfax.com/Used-Minivans_bt5"

driver.get(URL)

current_dir = 'C:/Users/matthew.hui/Documents/code/convertibles'

def scroll_to_end():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    print('scroll done')

# scroll
counter = 0
for i in range(1, 15):     
    scroll_to_end()
    # image_elements = driver.find_elements_by_class_name('rg_i')
    # print(len(image_elements))
    # for image in image_elements: 
    #     if (image.get_attribute('src') is not None):
    #         imageSources.append(image)

# print(imageSources)

# BEAUTIFUL SOUP VERSION -------------------------------------------------------------------------
# getURL = requests.get(URL, verify=False)
soup = BeautifulSoup(driver.page_source, 'html.parser')
images = soup.find_all('img', class_='rg_i')

# Store image links for extraction
imageSources = []
for image in images:
    # print(image)
    link = str(image.get('data-src'))
    if link.startswith("https://") or link.startswith("data:image/"):
        imageSources.append(link)

# --------------------------- CLEAR THE FOLDER ------------------------------------------------------
# for file in os.listdir(current_dir):
#     os.remove(os.path.join(current_dir, file))
    
for i, image in enumerate(imageSources):
    dir = current_dir + '/convertible_carfax' + str(i) + '.jpg'
    
    picture = requests.get(image, stream=True, verify=False).content
    found = False
    for existingPic in os.listdir(current_dir):
        # if dir == ('./minivans/'+ existingPic):
        #     continue
        if compareImg(current_dir + "/" + existingPic, Image.open(io.BytesIO(picture))):
            print(image, current_dir, existingPic)
            found = True
            break;
    
    if not found:
        with open(dir, 'wb') as file:
            file.write(picture)        

# for file in os.listdir(current_dir):
#     crop(os.path.join(current_dir, file))

# for i, file in enumerate(os.listdir(current_dir)):
#     os.rename(os.path.join(current_dir, file), os.path.join(current_dir, 'minivan_test' + str(i) + '.jpg'))
  
