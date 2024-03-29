# Tensorflow ; keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from src.settings import config


# *Metricas que podemos usar: accuracy, recision, recall,ConfusionMatrixPlot ()
def create_basic_cnn_model(mode_class):
    """
    Function to create a basic CNN.
    :param num_classes: Numero de etiquetas.
    :return: A basic CNN model.
    """
    cnn_model = Sequential()
    # Capas de Convolusión
    # if config.USE_DESCRIPTOR:
    cnn_model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(224, 224, 3)))
    # else:
    cnn_model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(224, 224, 1)))
    cnn_model.add(MaxPool2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
    cnn_model.add(MaxPool2D((2, 2)))

    # Aplanar resultados
    cnn_model.add(Flatten())

    # Dropout
    # cnn_model.add(Dropout(0.5, seed=111, name="Dropout_1"))
    # FC
    cnn_model.add(Dense(16, activation='relu'))
    cnn_model.add(Dense(3, activation='softmax', name='output'))

    if config.DEBUG_MODE_MODELS:
        cnn_model.summary()
    return cnn_model
