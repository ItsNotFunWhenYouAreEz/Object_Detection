import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, BatchNormalization, Rescaling
from tensorflow.keras import layers

def convBlock(last_layer, units, kernel_size = 3, activation='relu') :
    x = Conv2D(units, kernel_size, padding='same')(last_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    return x 

def buildModel(input_shape, n_classes):
    input = tf.keras.layers.Input(input_shape)
    x = Rescaling(1./255)(input)

    x = convBlock(input, 16)
    x = convBlock(x, 32)
    x = convBlock(x, 64)
    x = convBlock(x, 64)
    x = convBlock(x, 128)
    x = convBlock(x, 128)

    x = Flatten()(x)

    locator_branch = Dense(128, activation='relu')(x)
    locator_branch = Dense(64, activation='relu')(locator_branch)
    locator_branch = Dense(32, activation='relu')(locator_branch)
    locator_branch = Dense(4, activation='sigmoid', name='bb_head')(locator_branch)

    classifier = Dense(128, activation='relu')(x)
    classifier = Dense(n_classes, activation='softmax', name='cl_head')(classifier)

    return Model(input, outputs= [locator_branch, classifier])

if __name__ == "__main__" :
    model = buildModel((128, 128, 1) ,4)
    model.summary()
