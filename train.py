import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from model import buildModel
from DataLoader import creatDataSet

def schedule(epoch) :
    if epoch <= 10 :
        return 0.0001

    elif epoch <= 20 :
        return 0.000005
    
    elif epoch <= 30 :
        return 0.00005
    
    elif epoch < 40 : 
        return 0.000005
    
    elif epoch < 50 : 
        return 0.000001
    
    else :
        return 0.0000001

# def schedule(epoch):
#   def lr(epoch, start_lr, exp_decay):
#     return start_lr * math.exp(-exp_decay*epoch)
#   return lr(epoch, 0.0005, 0.1)

BATCH_SIZE = 16
CLASSES = ["B" , "H", "S", "U"]
EPOCHS = 100

img_size = (128, 128, 1)
data_path = "Data/Images/"
annotation_path = "Data/Annotation/"

train_x, train_y = creatDataSet(data_path, annotation_path, CLASSES)
model = buildModel((128, 128, 1), 4)

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = {'cl_head' : tf.keras.losses.CategoricalCrossentropy(), 'bb_head' : tf.keras.losses.MSE },
            metrics = {'cl_head' : 'accuracy', 'bb_head' : tf.keras.losses.MSE })

lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.00001, verbose = 1)

steps_per_epoch = int(len(train_x) / BATCH_SIZE)
model.fit(train_x, train_y, batch_size = BATCH_SIZE, steps_per_epoch = steps_per_epoch , epochs = EPOCHS, callbacks = [early_stopping])

model.save('model.h5')


