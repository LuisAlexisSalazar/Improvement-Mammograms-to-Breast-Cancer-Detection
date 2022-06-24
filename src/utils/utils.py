import zipfile
import tarfile

import pandas as pd
import cv2
from src.settings.config import *
from src.utils.create_image import *
from src.utils.folders import *

import json
import io


# --Utilidades para la descarga de los datsets
def unzip_zip(path_data_set, name_file):
    try:
        with zipfile.ZipFile(path_data_set + name_file, 'r') as zip_ref:
            zip_ref.extractall(path_data_set)
    except Exception as e:
        print("Error al descomprimir el archivo zip")
        print("Error:", e)
        return False


def unzip_tar(path_data_set, name_file):
    try:
        print(path_data_set + name_file)
        print(os.listdir(path_data_set))
        file = tarfile.open(path_data_set + name_file)
        file.extractall(path_data_set)
        return True
    except Exception as e:
        print("Error al descomprimir el archivo")
        print("Error:", e)
        return False


# ? Leer información del MIAS
def create_df_dataset_MIAS(file_path: str = PATH_DATA_RAW + "MIAS/info_MIAS.txt") -> pd.DataFrame:
    # *Descripción de los campos del dataframe,
    # ?muchos campos seran Nan en la clase anormal porque no tienen anomalias son normales
    """
    name_file -> nombre del archivo sin saber el formato que es ".PGM"
    cjf -> Caracteristica del tejido de fondo
    meta_class -> clase de anormalidades que presenta (calcificaión,asimetrico,masas bien definidas)
    abn_class -> Gravedad de anormalidad
    x_abn , y_abn -> coordenadas de centro de al anormalidad
    radio -> radio de que encierra la anomalía
    """
    df_mammogram = pd.read_table(file_path, delimiter='\s', engine='python')
    df_mammogram.columns = ['name_file', 'cjf', 'meta_class', 'abn_class', 'x_abn', 'y_abn', 'radio']

    # ! Hay 3 clases que podemos definir en abn_class: B (Beningno), M(Maligno) y Nan (sin tumores)
    # ? Definir con que clases nos quedaremos
    # * por el momento sera con solo Beningno y Maligno por ello eliminamos los que son Nan
    # df_mammogram.dropna(subset=['abn_class'], inplace=True)
    return df_mammogram


# ?Leer la información del MINI-DDSM en la carpeta JPEG-8 por ser mas ligero
def create_df_dataset_MINI_DDSM(
        file_path: str = PATH_DATA_RAW +
                         "MINI-DDSM/MINI-DDSM-Complete-JPEG-8/DataWMask.xlsx") -> pd.DataFrame:
    """
    fullPath -> rota completa de donde se encuentra la imagen en jpg
    fileName -> Nombre del archivo de imagen .jpg
    View -> Vistas de la mamografia craneocaudales bilaterales (CC) y oblicuas mediolaterales (MLO), # En total 4
    imagenes por paciente porque 2 vistas por cada lado
    Side -> En que lado del ceno vamos a es tomado puede ser left o right
    Status -> Estado de la mamografía existe 3 tipos Benign, Cancer y Normal
    Tumour_Contour -> Dirección de imagen que indica donde esta el tumor 1
    Tumour_Contour2 -> Dirección de imagen que indica donde esta el tumor2
    age -> Edad de la paciente
    Density -> nivel de densidad: 3, 2, 1, 4, 0
    """
    # !Tambien hay 3 clases en status de mamografía Benign, Cancer y Normal
    # ? Definir con que clases nos quedaremos
    # toDo: Transformar la clase beningna a normal ya que benigna solamente s un tumor pero no cancerigena
    df_mammogram = pd.read_excel(file_path, sheet_name=0)
    return df_mammogram


# ? path_base es el folder esta las iamgenes crudas
def separate_image_folders(path_base, name_dataset, mode_classification=MODE_DOWNLOAD_DEFAULT):
    dir_binary_classification = path_base + "/" + mode_classification
    if mode_classification == "BinaryNM":
        path_binary_classification_normal = dir_binary_classification + "/normal"
        path_binary_classification_maligno = dir_binary_classification + "/maligno"
        list_sub_folders = [path_binary_classification_normal, path_binary_classification_maligno]

        if create_all_folder(dir_binary_classification, list_sub_folders):
            # --MIAS
            if name_dataset == "MIAS":
                df_mias = create_df_dataset_MIAS()
                read_image_separate_MIAS(df_mias, list_sub_folders,
                                         path_base, mode_classification)
            # --MINI-DDSM
            elif name_dataset == "MINI-DDSM":  # CBIS
                df_MINI_DDSM = create_df_dataset_MINI_DDSM()
                read_image_separate_MINI_DDSM(df_MINI_DDSM, list_sub_folders,
                                              path_base, mode_classification)
    elif mode_classification == "BinaryBM":

        path_binary_classification_benigno = dir_binary_classification + "/benigno"
        path_binary_classification_maligno = dir_binary_classification + "/maligno"
        list_sub_folders = [path_binary_classification_benigno, path_binary_classification_maligno]

        if create_all_folder(dir_binary_classification, list_sub_folders):
            # --MIAS
            if name_dataset == "MIAS":
                df_mias = create_df_dataset_MIAS()
                read_image_separate_MIAS(df_mias, list_sub_folders,
                                         path_base, mode_classification)
            # --MINI-DDSM
            elif name_dataset == "MINI-DDSM":  # CBIS
                df_MINI_DDSM = create_df_dataset_MINI_DDSM()
                read_image_separate_MINI_DDSM(df_MINI_DDSM, list_sub_folders,
                                              path_base, mode_classification)
    elif mode_classification == "BinaryBN":
        path_binary_classification_benigno = dir_binary_classification + "/benigno"
        path_binary_classification_normal = dir_binary_classification + "/normal"
        list_sub_folders = [path_binary_classification_benigno, path_binary_classification_normal]

        if create_all_folder(dir_binary_classification, list_sub_folders):
            # --MIAS
            if name_dataset == "MIAS":
                df_mias = create_df_dataset_MIAS()
                read_image_separate_MIAS(df_mias, list_sub_folders,
                                         path_base, mode_classification)
            # --MINI-DDSM
            elif name_dataset == "MINI-DDSM":  # CBIS
                df_MINI_DDSM = create_df_dataset_MINI_DDSM()
                read_image_separate_MINI_DDSM(df_MINI_DDSM, list_sub_folders,
                                              path_base, mode_classification)
    elif mode_classification == "Binary(BM)N":
        path_binary_classification_tumoral = dir_binary_classification + "/tumoral"
        path_binary_classification_normal = dir_binary_classification + "/normal"
        list_sub_folders = [path_binary_classification_tumoral, path_binary_classification_normal]

        if create_all_folder(dir_binary_classification, list_sub_folders):
            # --MIAS
            if name_dataset == "MIAS":
                df_mias = create_df_dataset_MIAS()
                read_image_separate_MIAS(df_mias, list_sub_folders,
                                         path_base, mode_classification)
            # --MINI-DDSM
            elif name_dataset == "MINI-DDSM":  # CBIS
                df_MINI_DDSM = create_df_dataset_MINI_DDSM()
                read_image_separate_MINI_DDSM(df_MINI_DDSM, list_sub_folders,
                                              path_base, mode_classification)
    elif mode_classification == "ClassBMN":

        path_binary_classification_benigno = dir_binary_classification + "/benigno"
        path_binary_classification_maligno = dir_binary_classification + "/maligno"
        path_binary_classification_normal = dir_binary_classification + "/normal"
        list_sub_folders = [path_binary_classification_benigno, path_binary_classification_maligno,
                            path_binary_classification_normal]

        if create_all_folder(dir_binary_classification, list_sub_folders):
            # --MIAS
            if name_dataset == "MIAS":
                df_mias = create_df_dataset_MIAS()
                read_image_separate_MIAS(df_mias, list_sub_folders,
                                         path_base, mode_classification)
            # --MINI-DDSM
            elif name_dataset == "MINI-DDSM":  # CBIS
                df_MINI_DDSM = create_df_dataset_MINI_DDSM()
                read_image_separate_MINI_DDSM(df_MINI_DDSM, list_sub_folders,
                                              path_base, mode_classification)


# --Utilidad para ver la cantidad de datos en los folders
def amount_images_to_classification(path_to_classification, mode_class):
    path_benigno = path_to_classification + "/benigno"
    path_maligno = path_to_classification + "/maligno"
    path_normal = path_to_classification + "/normal"
    path_tumoral = path_to_classification + "/tumoral"

    # --Impresión de los paths
    if DEBUG_MODE:
        list = [path_benigno, path_maligno, path_normal, path_tumoral]
        print(list)

    if os.path.exists(path_benigno):
        amount_benigno = len(os.listdir(path_benigno))

    if os.path.exists(path_maligno):
        amount_maligno = len(os.listdir(path_maligno))

    if os.path.exists(path_normal):
        amount_normal = len(os.listdir(path_normal))

    if os.path.exists(path_tumoral):
        amount_tumoral = len(os.listdir(path_tumoral))

    # --Impresión de las cantidades
    if DEBUG_MODE:
        dict = {"amount_benigno": amount_benigno, "amount_maligno": amount_maligno, "amount_normal": amount_normal,
                "amount_tumoral": amount_tumoral}
        print(dict)

    if mode_class == "BinaryNM":
        print("Normal:", amount_normal)
        print("Maligno:", amount_maligno)
    elif mode_class == "BinaryBM":
        print("Benigno:", amount_benigno)
        print("Maligno:", amount_maligno)
    elif mode_class == "BinaryBN":
        print("Benigno:", amount_benigno)
        print("Normal:", amount_normal)
    elif mode_class == "Binary(BM)N":
        print("Tumoral:", amount_tumoral)
        print("Normal:", amount_normal)
    else:  # mode_class == MODE_3_CLASS:
        print("Benigno:", amount_benigno)
        print("Maligno:", amount_maligno)
        print("Normal:", amount_normal)


# --Utilidades para guardar el modelo como files
# Summary store in string
# https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
# Guardar json la confiuración y cargar modelo con el json
# https://moonbooks.org/Articles/How-to-save-the-architecture-of-a-tensorflow-model-summary-in-a-json-file-/
# Guardar con MLFlow un txt
# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_text
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#:~:text=get_config(self)%20%3A%20Returns%20a,model%20that%20contains%20this%20layer.
# Guarda la arquitectura sin pesos
# get_config: Devuelve un diccionario que contiene la configuración
# utilizada para inicializar esta capa.
def create_model_json(model):
    # dict_architecture_model = model.get_config()
    json_architecture_model = model.to_json()

    # with open("architecture.json", "w") as outfile:
    #     json.dump(dict_architecture_model, outfile)
    with open("temp/architecture.json", "w") as f:
        json.dump(json.loads(json_architecture_model), f)


def create_summary_model_txt(model):
    summary_as_string = get_model_summary(model)
    summary_as_list = summary_as_string.split(sep='\n')
    with open('temp/summary_model.txt', 'w') as f:
        for item in summary_as_list:
            f.write("%s\n" % item)


def Save_model_summary_txt_architecture_json(model):
    create_model_json(model)
    create_summary_model_txt(model)
    print("Summary en txt creado")
    print("Arquitectura en json creado")

# !MLFlow para qeu registre log de manera rapida
# %%
# Registrar el modelo mlFlow de manera rapida sin configurar nada
# https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html
# import mlflow.tensorflow
# mlflow.tensorflow.autolog()
