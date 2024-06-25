import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Rescaling



def buildModel(input_shape):
    input = Input(input_shape)
    x = Rescaling(1./255)(input)

    x = Conv2D(16, 3, padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, 3, padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(48, 3, padding='same')(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(64, 3, padding='same')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation = "relu")(x)
    output = Dense(8, activation = "sigmoid")(x)

    return Model(input, output)

if __name__ == "__main__" :
    model = buildModel((128, 128, 1))
    model.summary()




