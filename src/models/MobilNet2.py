import tensorflow as tf
# import tensorflow_hub as hub
from keras import Input
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, Concatenate
# from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNetV2

from src.settings import config


# url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# !Lo entrenamos completamente las capas no estan desactivadas
def create_mobilNet(mode_classification):
    # !MobilNet solo hace match con 3 canales de color
    image_input = Input(shape=(224, 224, 3))

    mobil_net2_base = MobileNetV2(weights="imagenet",
                                  include_top=False, input_tensor=image_input)
    # ?ultima de convolusión que es conv_pw_13_relu
    last_layer = mobil_net2_base.layers[-1].output
    x = Flatten(name='flatten')(last_layer)
    x = Dropout(rate=0.2)(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    # x = Dropout(rate=0.3)(x)
    x = Dense(32, activation='relu', name='fc2')(x)
    # x = Dropout(rate=0.3)(x)

    if mode_classification in config.MODES_BINARY_CLASS:
        out = Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='output')(x)

    else:  # MODE_3_CLASS = "ClassBMN"
        out = Dense(3, kernel_initializer="random_uniform", activation='softmax', name='output')(x)
    custom_model = Model(image_input, out)

    if config.DEBUG_MODE_MODELS:
        custom_model.summary()
    return custom_model
