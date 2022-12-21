# Tensorflow ; keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.regularizers import l1_l2
from src.settings import config


# *Metricas que podemos usar: accuracy, recision, recall,ConfusionMatrixPlot ()
def create_basic_cnn_model(mode_class):



    cnn_model = Sequential()
    cnn_model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    cnn_model.add(Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    cnn_model.add(Dense(3, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), name='output'))

    # MOdelo con overfitting
    # cnn_model = Sequential()
    # cnn_model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    # cnn_model.add(MaxPool2D(pool_size=(2, 2)))
    #
    # cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
    # cnn_model.add(MaxPool2D(pool_size=(2, 2)))
    #
    # cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    # cnn_model.add(MaxPool2D(pool_size=(2, 2)))
    #
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(64, activation='relu'))
    # cnn_model.add(Dropout(0.5))
    # cnn_model.add(Dense(3, activation='softmax'))
    if config.DEBUG_MODE_MODELS:
        cnn_model.summary()
    return cnn_model
    # !Antiguo modelo
    # cnn_model = Sequential()
    # cnn_model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(224, 224, 3)))
    # cnn_model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(224, 224, 1)))
    # cnn_model.add(MaxPool2D(pool_size=(2, 2)))
    # cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
    # cnn_model.add(MaxPool2D((2, 2)))
    # # Aplanar resultados
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(16, activation='relu'))
    # cnn_model.add(Dense(3, activation='softmax', name='output'))
