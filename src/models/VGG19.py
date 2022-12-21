import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Concatenate
from tensorflow.keras import Input, Model
from src.settings import config
from tensorflow.keras.applications.vgg19 import VGG19


# ?Summary de VGG19
# https://iq.opengenus.org/vgg19-architecture/
def create_VGG19(mode_classification):
    # --Transfer learning con VGG19
    image_input = Input(shape=(224, 224, 3))
    # image_input = None
    # if config.USE_DESCRIPTOR:
    #     image_input = Input(shape=(224, 224, 3))
    # else:
    #     # --Transfer learning con VGG16
    #     image_input = Input(shape=(224, 224, 1))
    #     image_input = Concatenate()([image_input, image_input, image_input])
    VGG19_base = VGG19(weights="imagenet",
                       input_tensor=image_input,
                       include_top=False)

    last_layer = VGG19_base.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(rate=0.3)(x)
    # --Capa final conmpletamente conectada
    if mode_classification in config.MODES_BINARY_CLASS:
        out = Dense(1, activation='sigmoid', name='output')(x)

    else:  # MODE_3_CLASS = "ClassBMN"
        out = Dense(3, activation='softmax', name='output')(x)

    custom_model = Model(image_input, out)
    # ?Congelar los pesos menos las 6 capas
#     for layer in custom_model.layers[:-6]:
#         layer.trainable = False

    if config.DEBUG_MODE_MODELS:
        custom_model.summary()
    return custom_model
