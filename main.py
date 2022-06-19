# *Limpieza de las variables
import gc
gc.collect()
# *Visualización del modelo de keras
# https://github.com/paulgavrikov/visualkeras

# *MLFlow: Registrar las metricas y parametros de nuestro modelo
# https://medium.com/analytics-vidhya/mlflow-logging-for-tensorflow-37b6a6a53e3c
# https://medium.com/analytics-vidhya/mlflow-projects-24c41b00854
# https://medium.com/analytics-vidhya/tensorflow-model-tracking-with-mlflow-e9de29c8e542
# Ejemplos https://github.com/amesar/mlflow-examples
# Mejor configuración https://towardsdatascience.com/using-mlflow-to-track-and-version-machine-learning-models-efd5daa08df0
# Buena documentación https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html
#  --Imports to download data
from src.download_dataset.download import download_data
from src.settings.config import *
#
# # --Download Data MIAS and MINI-CBIS
# list_mode = ["BinaryNM", "BinaryBN", "Binary(BM)N", "ClassBMN"]
#
# for mode in list_mode:
#     download_data("MINI-DDSM", mode)


# !Seguir revisando el topico en github de brast cancer
# https://github.com/topics/mammogram-images

# !Revisar MLFLOW
# https://stackoverflow.com/questions/71846804/how-to-i-track-loss-at-epoch-using-mlflow-tensorflow/71850503
# Comparar los modelos a travez de los graficos Mflow
# https://towardsdatascience.com/experiment-tracking-template-with-keras-and-mlflow-36aa214896df
# *Guarda por cada epocas
# https://medium.com/@ij_82957/how-to-reduce-mlflow-logging-overhead-by-using-log-batch-b61301cc540f
# Con MLFlow grafica o guarda matrix de confusión
# https://www.linkedin.com/pulse/machine-learning-mlflow-managing-end-to-end-lifecycle-gaurav-pahuja