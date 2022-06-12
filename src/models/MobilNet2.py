import tensorflow as tf
# import tensorflow_hub as hub
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
# from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from src.settings import config


# url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"


def create_mobilNet():
    model = Sequential()
    # !MobilNet solo hace match con 3 canales de color
    # mobil_netv2_base = hub.KerasLayer(url, input_shape=(224, 224, 3))

    mobil_netv2_base = MobileNetV2(weights="imagenet",
                                   input_shape=(224, 224, 3),
                                   include_top=False)
    # Congelar el modelo descargado
    # mobil_netv2_base.trainable = False

    # Primera configuración que entrena se estanca en 0.52
    # mobil_netv2_base.trainable = False
    # model.add(mobil_netv2_base)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(2, activation='softmax'))

    # !Intento de 2da configuración
    model.add(mobil_netv2_base)
    model.add(Flatten())
    fully_connected = Sequential(name="Fully_Connected")
    fully_connected.add(Dropout(0.2, seed=111, name="Dropout_1"))
    fully_connected.add(Dense(units=512, activation='relu', name='Dense_1'))
    # fully_connected.add(Dropout(0.2, name="Dropout_2"))
    fully_connected.add(Dense(units=32, activation='relu', name='Dense_2'))
    fully_connected.add(Dense(2, activation='softmax'))
    model.add(fully_connected)
    
    if config.DEBUG_MODE_MODELS:
        model.summary()
    return model
