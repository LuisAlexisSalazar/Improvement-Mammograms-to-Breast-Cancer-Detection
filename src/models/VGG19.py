import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout,Concatenate
from tensorflow.keras import Input, Model, Sequential

from src.settings import config
from tensorflow.keras.applications.vgg16 import VGG16


# ?Summary de VGG19
# https://iq.opengenus.org/vgg19-architecture/
def create_VGG19(mode_classification):
    # --Transfer learning con VGG16
    image_input = Input(shape=(224, 224, 1))
    img_conc = Concatenate()([image_input, image_input, image_input])
    # !VGG16 solo hace match con 3 canales de color
    VGG16_base = VGG16(weights="imagenet",
                       input_tensor=img_conc,
                       include_top=False)
    # VGG16_base = VGG16(input_tensor=image_input, include_top=True)

    # *Tomar desde la ultima capa antes de hacer flatten,desne,dense y predictore
    last_layer = VGG16_base.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    # +Capas completamente conectadas
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
    for layer in custom_model.layers[:-6]:
        layer.trainable = False

    if config.DEBUG_MODE_MODELS:
        custom_model.summary()
    return custom_model
