import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from model import buildModel
from DataLoader import creatDataSet
import cv2 
import numpy as np
from tensorflow.keras.models import load_model

BATCH_SIZE = 16
CLASSES = ["B" , "H", "S", "U"]
EPOCHS = 40


img_size = (128, 128, 1)
data_path = "Data/Images/"
annotation_path = "Data/Annotation/"

train_x, train_y = creatDataSet(data_path, annotation_path, CLASSES)
model = buildModel((128, 128, 1), 4)

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = {'cl_head' : tf.keras.losses.CategoricalCrossentropy(), 'bb_head' : tf.keras.losses.MSE },
            metrics = {'cl_head' : 'accuracy', 'bb_head' : tf.keras.losses.MSE })

model.summary()

steps_per_epoch = int(len(train_x) / BATCH_SIZE)
model.fit(train_x, train_y, batch_size = BATCH_SIZE, steps_per_epoch = steps_per_epoch , epochs = EPOCHS)

model.save('model.h5')


