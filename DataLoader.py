import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import tensorflow as tf

def readAnnottion(annotate_file, classes,):
    w, h  = (128, 128 )
    for obj in ET.parse(annotate_file).getroot().iter('object'):
        cls = obj.find('name').text
        cls_id = classes.index(cls)
        classes = np.zeros(len(classes))
        classes[cls_id] = 1
        xmlbox = obj.find('bndbox')
        box = np.asarray([int(xmlbox.find('xmin').text) / w, int(xmlbox.find('ymin').text) /h ,
             int(xmlbox.find('xmax').text) / w, int(xmlbox.find('ymax').text) / h])
        return [box[0], box[1], box[2], box[3]], classes

def readTextAnnotation(annotate_file) :
    with open(annotate_file, "r") as f : 
        x = f.read()
        box = x.split() 
        return [int(int(box[0]) / 128), int(int(box[1]) / 128), int(int(box[2]) / 128), int(int(box[3]) / 128)]


def creatDataSet(images_dir, annotate_dir, classes) :
    X = [ ]

    classes = []
    boxes = [ ]

    for label in os.listdir(images_dir) :   
            for i in os.listdir(images_dir + label) :   

                img = cv2.imread(images_dir + label + "/" + i)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, 2 )
                if label != "B" :
                    x = i.split(".")[0]
                    box, classifier = readAnnottion(f"{annotate_dir}{label}/{x}.xml", ["B", "H", "S", "U"])
                else :
                    box, classifier = [0, 0, 0, 0], [1, 0, 0, 0]
                X.append(img)
                boxes.append(box)
                classes.append(classifier)

    X = np.asarray(X)
    boxes = np.asarray(boxes)
    classes = np.asarray(classes)

    Y = {
        "cl_head": classes,
        "bb_head": boxes
    }

    return X, Y