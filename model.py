import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, BatchNormalization, Rescaling


def buildModel(input_shape, n_classes):
    input = Input(input_shape)
    x = Rescaling(1./255)(input)

    x = Conv2D(16, 3, padding='same', use_bias = False)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, padding='same', use_bias = False)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, padding='same', use_bias = False)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, padding='same', use_bias = False)(x)
    x = Conv2D(64, 3, padding='same', use_bias = False)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(92, 3, padding='same', use_bias = False)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, padding='same', use_bias = False)(x)
    x = MaxPooling2D()(x)

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

