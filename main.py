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

# --Revisar MLFLOW
# Buen tutorial completo: https://towardsdatascience.com/experiment-tracking-with-mlflow-in-10-minutes-f7c2128b8f2c
# https://stackoverflow.com/questions/71846804/how-to-i-track-loss-at-epoch-using-mlflow-tensorflow/71850503
# Comparar los modelos a travez de los graficos Mflow
# https://towardsdatascience.com/experiment-tracking-template-with-keras-and-mlflow-36aa214896df
# *Guarda por cada epocas
# https://medium.com/@ij_82957/how-to-reduce-mlflow-logging-overhead-by-using-log-batch-b61301cc540f
# Con MLFlow grafica o guarda matrix de confusión
# https://www.linkedin.com/pulse/machine-learning-mlflow-managing-end-to-end-lifecycle-gaurav-pahuja
# Eliminar el experimento
# https://stackoverflow.com/questions/60088889/how-do-you-permanently-delete-an-experiment-in-mlflow
# Rm -r equivalente en windwo
# https://stackoverflow.com/questions/60088889/how-do-you-permanently-delete-an-experiment-in-mlflow
# rm -rf mlruns/.trash/*

# Cuando no muestra los experimentos mandados en la url
# https://stackoverflow.com/questions/71708147/mlflow-tracking-ui-not-showing-experiments-on-local-machine-laptop
# mlflow ui --backend-store-uri file:///Users/Swapnil/Documents/LocalPython/MLFLowDemo/mlrun
# mlflow ui --backend-store-uri file:///E:\U\Improvement-Mammograms-to-Breast-Cancer-Detection\mlruns

# --Matrix de confusión al guardar el modelo con tensorflow
# https://www.youtube.com/watch?v=EkAg51oIvQI&t=689s
# https://github.com/DavidReveloLuna/Bird_Classification-
# ------------Test Codigos-------------------


# ?Futuro hacer Yolo
# https://github.com/DavidReveloLuna/YoloV5


# ------------- Tabla Dinamica en jupyter notebook como google collaboraty----------
# https://github.com/quantopian/qgrid
# https://stackoverflow.com/questions/61709252/jupyterlab-table-dynamic-output-sorting-filterung
# from src.utils.utils import create_df_dataset_MIAS
# from src.settings.config import PATH_DATA_RAW, MODE_DOWNLOAD_DEFAULT
#
# df_mias = create_df_dataset_MIAS(PATH_DATA_RAW + "MIAS/info_MIAS.txt")
#
# import qgrid
#
# qgrid_widget = qgrid.show_grid(df_mias, show_toolbar=True)
# qgrid_widget
#
# from src.utils.utils import create_df_dataset_MINI_DDSM
# from src.settings.config import PATH_DATA_RAW, MODE_DOWNLOAD_DEFAULT
#
# df_mini_MIAS = create_df_dataset_MINI_DDSM()
#
# qgrid_widget = qgrid.show_grid(df_mini_MIAS, show_toolbar=True)
# qgrid_widget

# ?Ejecucición de mlflow en linea de comando
# mlflow ui --backend-store-uri mlruns
