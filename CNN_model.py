"""
**
* Author            : A. Peter Hudson
* Filename          : CNN_model.py
* Functions         : buildModel(preCalculatedWeightPath=None, shape)
* Global Variables  : module_path
**
"""

import os, sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)

# Used Keras libraries to create model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# Setting the backend as 'th' for theano ('tf' for tensorflow)
K.set_image_dim_ordering('th')

# Architecture: This architecture contain an input layer, 9 convolutional layers, 2 dense layers and output layer.
"""
* Function Name : buildModel(preCalculatedWeightPath=None, shape)
* Input         : preCalculatedWeightPath - Pre-calculated weight if any
                  shape - dimension of the input image
* Output        : Returns a model
* Example Call  : buildModel(model.h5)
*
* This function accepts the set of images and creates a convolutional network model and returns the same.
"""

def buildModel(preCalculatedWeightPath=None, shape=(48, 48)):
    model = Sequential()

    model.add (ZeroPadding2D ((1, 1), input_shape=(1, 48, 48)))
    model.add (Conv2D (32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), input_shape=(1, 48, 48)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add (ZeroPadding2D ((1,1)))
    model.add (Conv2D (64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add (ZeroPadding2D ((1, 1)))
    model.add (Conv2D (128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    print ("Create model successfully")
    if preCalculatedWeightPath:
        model.load_weights(preCalculatedWeightPath)

    model.compile(optimizer='adam', loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return model
