from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

import os

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import numpy as np


class_to_label = {'BarCodes': 0, 'Graphs': 1, 'LogosAndStamps': 2, 'Photographs': 3, 'QRcodes': 4, 'Signatures': 5}
label_to_class = {v: k for k, v in class_to_label.items()}

num_classes = len(class_to_label)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

print(model.summary())
model.load_weights('src/model_weights.h5')