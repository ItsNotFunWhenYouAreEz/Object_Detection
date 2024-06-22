import cv2
import os 
import random
import numpy as np 
import xml.etree.ElementTree as ET
from tqdm import tqdm

def readAnnottion(annotate_file):
    for obj in ET.parse(annotate_file).getroot().iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        box = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text) ,
             int(xmlbox.find('xmax').text) , int(xmlbox.find('ymax').text)]
    return box, cls

def createAnnotationFile(annotate_file, cls, x1, y1, x2, y2):
    with open(annotate_file, "w") as f :
        f.write("<annotation>\n")
        f.write("\t<object>\n")
        f.write(f"\t\t<name>{cls}</name>\n")
        f.write("\t\t<bndbox>\n")
        f.write(f"\t\t\t<xmin>{x1}</xmin>\n")
        f.write(f"\t\t\t<ymin>{y1}</ymin>\n")
        f.write(f"\t\t\t<xmax>{x2}</xmax>\n")
        f.write(f"\t\t\t<ymax>{y2}</ymax>\n")
        f.write("\t\t</bndbox>\n")
        f.write("\t</object>\n")
        f.write("</annotation>")

def move(img, x, y) :
    w, h, _ = img.shape
    translation_matrix = np.float32([ [1, 0, x], [0, 1, y] ])
    img = cv2.warpAffine(img, translation_matrix, (h, w), borderValue=(255,255,255))
    return img

def addRandomNoise(img, value): 
    h, w, c = img.shape
    for i in range(value) :
        img[random.randint(0, h-1), random.randint(0, w-1)] = 1
    return img

def addRandomShape(img, size, x, y) :

    img = cv2.rectangle(img, (x - size, y - size), (x + size, y - size), (255, 255, 255), -1) 
    return img

image_dir = "Data/Images" 
annotation_dir = "Data/Annotation"
noise_amount = [0, 25]

for label in os.listdir(image_dir):
    dir = os.listdir(f"{image_dir}/{label}")
    cnt = len(dir) + 1 
    for img_name in dir : 

        img = cv2.imread(f"{image_dir}/{label}/{img_name}")

        if label != "B" :
            
            annotation_file = img_name.split(".")[0]
            annotation_path = f"{annotation_dir}/{label}/{annotation_file}.xml"
            box, cls = readAnnottion(annotation_path)
            x1, y1, x2, y2  = box

            for i in range(20) : 
                x1, y1, x2, y2  = box

                if False :
                    x, y = random.randint(-x1, 128 - x2), random.randint(-y1, 128 - y2)
                    im = move(img, x, y)
                    x1, y1, x2, y2 = x1 + x, y1 + y, x2 + x, y2 + y
                    angle = random.randint(1, 360)
                    bb = np.array(((x1,y1),(x2,y1),(x2, y2),(x1, y2))) 
                    mat = cv2.getRotationMatrix2D((64, 64), angle, 1.0) 

                    img_rotated = cv2.warpAffine(im, mat, img.shape[1::-1], borderValue=(255,255,255)) 

                    bb_rotated = np.vstack((bb.T,np.array((1,1,1,1))))
                    bb_rotated = np.dot(mat,bb_rotated).T

                    x1, y1, x2, y2 = 128, 128, 0, 0
                    for i in bb_rotated :
                        if i[0] < x1 :
                            x1 = int(i[0])
                        if i[0] > x2 :
                            x2 = int(i[0])
                        if i[1] < y1 :
                            y1 = int(i[1])
                        if i[1] > y2 :
                            y2 = int(i[1])

                    createAnnotationFile(f"{annotation_dir}/{label}/{cnt}.xml", cls, x1, y1, x2, y2)
                    cv2.imwrite(f"{image_dir}/{label}/{cnt}.png", addRandomNoise(img_rotated, random.choice(noise_amount)))
                    cnt += 1
                else : 
                    x, y = random.randint(-x1, 128 - x2), random.randint(-y1, 128 - y2)
                    createAnnotationFile(f"{annotation_dir}/{label}/{cnt}.xml", cls, x1 + x, y1 + y, x2 + x, y2 + y)
                    cv2.imwrite(f"{image_dir}/{label}/{cnt}.png", addRandomNoise(move(img, x, y), random.choice(noise_amount)))
                    cnt += 1
        else : 
            for i in range(20) : 
                if random.choice([0, 1]) == 1 :
                    addRandomShape(img, random.randint(1, 10), random.randint(1,128), random.randint(1,128))
                x, y = random.randint(-50, 50), random.randint(-50, 50)
                cv2.imwrite(f"{image_dir}/{label}/{cnt}.png", addRandomNoise(move(img, x, y), random.choice(noise_amount)))
                cnt += 1



