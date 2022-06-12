import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from src.settings import config
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19


def create_VGG16():
    model = Sequential()
    # !VGG16 solo hace match con 3 canales de color
    VGG16_base = MobileNetV2(weights="imagenet",
                             input_shape=(224, 224, 1),
                             include_top=False)

    model.add(VGG16_base)
    model.add(Flatten())
    fully_connected = Sequential(name="Fully_Connected")
    fully_connected.add(Dropout(0.2, seed=111, name="Dropout_1"))
    fully_connected.add(Dense(units=512, activation='relu', name='Dense_1'))
    fully_connected.add(Dense(units=32, activation='relu', name='Dense_2'))
    fully_connected.add(Dense(2, activation='softmax'))
    model.add(fully_connected)

    if config.DEBUG_MODE_MODELS:
        model.summary()
    return model
