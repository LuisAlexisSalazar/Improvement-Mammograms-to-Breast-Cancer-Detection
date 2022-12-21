import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, Concatenate
from tensorflow.keras import Input, Model, Sequential
from src.settings import config
from tensorflow.keras.applications.vgg16 import VGG16

from src.roi_preprocessing.brint import run_gpu_brint_m, run_gpu_brint_s


# !Error InvalidArgumentError:  In[0] mismatch In[1] shape: 3 vs. 1: [16,3] [64,1] 0 0
# ?Summary de VGG16
# https://www.kaggle.com/getting-started/178568
def create_VGG16(mode_classification):

    image_input = Input(shape=(224, 224, 3))
    # !Fallido intentar aplicar el filtro cuando llega la imagen
    # img_conc = None
    # if config.APPLY_DESCRIPTOR:
    #     img_brint_s = run_gpu_brint_s(image_input, radius=config.RADIUS, q_points=config.Q_POINTS)
    #     img_brint_m = run_gpu_brint_m(image_input, radius=config.RADIUS, q_points=config.Q_POINTS)
    #     img_conc = Concatenate()([image_input, img_brint_s, img_brint_m])
    # else:
    #     img_conc = Concatenate()([image_input, image_input, image_input])

    # img_conc = Concatenate()([image_input, image_input, image_input])
    # !VGG16 solo hace match con 3 canales de color
    VGG16_base = VGG16(weights="imagenet",
                       input_tensor=image_input,
                       include_top=True)

    last_layer = VGG16_base.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    # --Capasa completamente conectadas
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    if mode_classification in config.MODES_BINARY_CLASS:
        out = Dense(1, activation='sigmoid', name='output')(x)

    else:  # MODE_3_CLASS = "ClassBMN"
        out = Dense(3, activation='softmax', name='output')(x)

    custom_model = Model(image_input, out)
    # *Congelar los pesos menos las 3 capas densa ultimas
#     for layer in custom_model.layers[:-3]:
#         layer.trainable = False

    if config.DEBUG_MODE_MODELS:
        custom_model.summary()
    return custom_model
