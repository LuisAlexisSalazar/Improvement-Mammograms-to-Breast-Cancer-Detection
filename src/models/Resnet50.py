import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Concatenate
from tensorflow.keras import Input, Model

from src.settings import config
from tensorflow.keras.applications.resnet50 import ResNet50


def create_ResNet50(mode_classification):
    # --Transfer learning con VGG16
    image_input = Input(shape=(224, 224, 1))
    img_conc = Concatenate()([image_input, image_input, image_input])

    ResNet50_base = ResNet50(weights="imagenet",
                             input_tensor=img_conc,
                             include_top=False)  # no incluya la ultima capa
    # *Tomar desde la ultima capa que es de activación_49
    last_layer = ResNet50_base.layers[-1].output
    x = Flatten(name='flatten')(last_layer)
    # --Capasa completamente conectadas
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(rate=0.3)(x)
    if mode_classification is config.MODES_BINARY_CLASS:
        out = Dense(1, activation='sigmoid', name='output')(x)

    else:  # MODE_3_CLASS = "ClassBMN"
        out = Dense(3, activation='softmax', name='output')(x)

    custom_model = Model(image_input, out)

    # *Congelar los pesos menos las 6 capas
    for layer in custom_model.layers[:-6]:
        layer.trainable = False

    if config.DEBUG_MODE_MODELS:
        custom_model.summary()
    return custom_model
