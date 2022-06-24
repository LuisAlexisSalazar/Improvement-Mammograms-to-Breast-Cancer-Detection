import os
from src.settings.config import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def definite_mode_read():
    return 'grayscale'

    # !Actualizado porque podemos cambiar la configuraciones de entrada
    # if is_transfer_learning:
    #     return 'rgb'
    # else:  # Own CNN
    #     return 'grayscale'


# https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
def generate_ImageDataGenerator(color_mode, validation_split=0.2):
    # *"grayscale"
    datagen = ImageDataGenerator(
        # * Es necesario para la lectura en rgb pero para color_mode en grayscale no
        # rescale=1. / 255,
        rotation_range=3,  # rotar en ±3 grados
        zoom_range=0.1,  # 1 es imagen sin zoom
        width_shift_range=0.1,  # 20% de recorrido
        height_shift_range=0.1,  # 20% de recorrido
        # https://stackoverflow.com/questions/62484597/understanding-width-shift-range-and-height-shift-range-arguments-in-kerass
        horizontal_flip=True,
        # vertical_flip=True,
        # https://stackoverflow.com/questions/57301330/what-exactly-the-shear-do-in-imagedatagenerator-of-keras
        shear_range=0.1,
        #     preprocessing_function=gray_to_rgb,
        validation_split=validation_split  # 20% para pruebas
        # https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
        # preprocessing_function=preprocess_input
    )
    if color_mode == "rgb":
        print("Es rgb Se le hara Reescalado")
        # !Es necesario para la lectura en rgb que este reescalado pero para en grayscale no
        datagen.rescale = 1. / 255

    return datagen
