import os 
import cv2

image_size = (128, 128)

dir = "RawData/"
dest = "Data/Images/"

def processImage(path, dest, shape) : 
    global cnt
    img = cv2.resize(cv2.imread(path), shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY) 
    cv2.imwrite(f"{dest}{cnt}.png", img)    
    cnt += 1 

for label in os.listdir(dir) :
    cnt = 1
    for i in os.listdir(dir + label) :
        processImage(f"{dir}/{label}/{i}", f"{dest}/{label}/", image_size)
