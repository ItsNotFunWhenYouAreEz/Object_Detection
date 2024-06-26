import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

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
    return [classes[0], classes[1], classes[2], classes[3], box[0], box[1], box[2], box[3]]

def creatDataSet(images_dir, annotate_dir, classes) :
    X = [ ]
    Y = [ ]

    for label in os.listdir(images_dir) :   
            for i in tqdm(os.listdir(images_dir + label)) :   
                
                img = cv2.imread(images_dir + label + "/" + i)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, 2 )
                if label != "B" :
                    x = i.split(".")[0]
                    annotate = readAnnottion(f"{annotate_dir}{label}/{x}.xml", ["B", "H", "S", "U"])
                else :
                    annotate = [1, 0, 0, 0, 0, 0, 0, 0]
                X.append(img)
                Y.append(annotate)

    X = np.asarray(X)
    Y = np.asarray(Y)


    return X, Y