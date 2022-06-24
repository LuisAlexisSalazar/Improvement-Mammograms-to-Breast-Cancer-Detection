import os
from src.settings import config
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy, Precision, Recall, \
    SpecificityAtSensitivity
import time
from tensorflow_addons.metrics import F1Score
from mlflow.entities import Metric

# *Crear propias metricas para tensorboard
# https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
# *[opción escogida]Librerias complemneto [tensorflow_addons] de tensorflow con las metricas que necesitas
# https://stackoverflow.com/questions/70589698/tensorflow-compute-precision-recall-f1-score


start_time = None


def definite_class_mode(mode_class):
    if mode_class in config.MODES_BINARY_CLASS:
        return 'binary'
    else:
        return 'categorical'


def definition_loss(mode_classification):
    if mode_classification in config.MODES_BINARY_CLASS:
        return 'binary_crossentropy'
    else:
        return 'categorical_crossentropy'


# ?Metricas en base a la matrix de confusión
# https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
# Lista de metricas
# https://stackoverflow.com/questions/59353009/list-of-metrics-that-can-be-passed-to-tf-keras-model-compile
# !F1 esta mal no toma en cuenta la precisiones y recall correctas
# https://stackoverflow.com/questions/57910680/how-to-name-custom-metrics-in-keras-fit-output
# !Customizar o mantener el nombre de las metricas para sacar el f1
def get_list_metrics(mode_classification):
    # list_metric = [Precision(), Recall(),F1Score(num_classes=2, average='macro',threshold=0.5)]
    list_metric = [Precision(name="precision"), Recall(name="recall")]

    if mode_classification in config.MODES_BINARY_CLASS:
        list_metric = list_metric + [BinaryAccuracy()]
    else:
        list_metric = list_metric + [CategoricalAccuracy()]

    return list_metric


def name_metric(object_metric):
    return object_metric.name


def start_time():
    global start_time
    start_time = time.time()


def end_time():
    global start_time
    run_time = time.time() - start_time
    if config.DEBUG_MODE:
        print("Tiempo de entrenamiento: %s seconds ---" % (run_time))
    return run_time


# https://www.iartificial.net/precision-recall-f1-accuracy-en-clasificacion/
def getF1(precision, recall, name_F1="f1_score"):
    list_f1 = []
    # https://stackoverflow.com/questions/42259166/python-3-valueerror-not-enough-values-to-unpack-expected-3-got-2
    # ?enumerate and zip
    for step, (value_precision, value_recall) in enumerate(zip(precision, recall)):
        if value_precision == 0 or value_recall == 0:
            value = 0
        else:
            value = 2 / ((1 / value_precision) + (1 / value_recall))
        f1_metric = Metric(
            key=name_F1,
            value=value,
            timestamp=0,
            step=step,
        )
        list_f1.append(f1_metric)

    return list_f1


# Debes pasar el "history" que devuelve el metodo .fit del modelo
def get_F1_scores(history):
    precision = history.history["precision"]
    recall = history.history["recall"]
    f1_score = getF1(precision, recall)

    val_precision = history.history["val_precision"]
    val_recall = history.history["val_recall"]
    val_f1_score = getF1(val_precision, val_recall, name_F1="val_f1_score")
    return f1_score, val_f1_score
