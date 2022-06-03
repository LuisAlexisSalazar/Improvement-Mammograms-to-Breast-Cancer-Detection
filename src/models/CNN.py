# Tensorflow ; keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from src.settings import config


def create_basic_cnn_model(num_classes: int):
    """
    Function to create a basic CNN.
    :param num_classes: Numero de etiquetas.
    :return: A basic CNN model.
    """
    cnn_model = Sequential()
    # Capas de Convolusión
    cnn_model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(100, 100, 1)))
    cnn_model.add(MaxPool2D((2, 2)))
    cnn_model.add(Conv2D(64, (5, 5), activation='relu'))
    cnn_model.add(MaxPool2D((2, 2)))

    # Aplanar resultados
    cnn_model.add(Flatten())

    # FC
    cnn_model.add(Dense(100, activation='relu'))

    # Capa densa para producir la salida final
    cnn_model.add(Dense(1, activation='sigmoid'))

    if config.DEBUG_MODE_MODELS:
        cnn_model.summary()
    return cnn_model
