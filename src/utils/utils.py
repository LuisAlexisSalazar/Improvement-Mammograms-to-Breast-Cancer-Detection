import zipfile
import tarfile
import os
import pandas as pd
import cv2
from src.settings.config import PATH_DATA_RAW, DEFAULT_MODE_CLASSIFICATION


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


def separate_image_folders(path_base, name_dataset, mode_classification=DEFAULT_MODE_CLASSIFICATION):
    if mode_classification == DEFAULT_MODE_CLASSIFICATION:
        dir_binary_classification = path_base + "/binary_classification"
        # dir_classification = path_data_set + ""
        if not os.path.exists(dir_binary_classification):
            os.mkdir(dir_binary_classification)

        path_binary_classification_benigno = dir_binary_classification + "/benigno"
        path_binary_classification_maligno = dir_binary_classification + "/maligno"
        if not os.path.exists(path_binary_classification_benigno):
            os.mkdir(path_binary_classification_benigno)

        if not os.path.exists(path_binary_classification_maligno):
            os.mkdir(path_binary_classification_maligno)

        if len(os.listdir(path_binary_classification_benigno)) == 0 and len(
                os.listdir(path_binary_classification_maligno)) == 0:
            if name_dataset == "MIAS":
                df_mias = create_df_dataset_MIAS()
                df_mias = df_mias.dropna(subset=['abn_class'])  # Class B and M
                # df_mias["abn_class"] = df_mias['abn_class'].replace(['M'], 'M')
                # df_mias["abn_class"] = df_mias['abn_class'].replace(['B'], 'N')
                # df_mias["abn_class"] = df_mias['abn_class'].replace([None], 'N')

                df_mias = df_mias[["name_file", "abn_class"]]

                # path = "data/dataset_raw/MIAS/"
                # path_data_set = path_data_set + "MIAS/"
                for index, row in df_mias.iterrows():
                    img = cv2.imread(path_base + row['name_file'] + ".pgm", cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (224, 224))

                    if row['abn_class'] == 'B':
                        cv2.imwrite(path_binary_classification_benigno + "/" + row['name_file'] + ".jpg", img)
                    else:  # M
                        cv2.imwrite(path_binary_classification_maligno + "/" + row['name_file'] + ".jpg", img)
            elif name_dataset == "MINI-DDSM":  # CBIS
                df_CBIS = create_df_dataset_MINI_DDSM()
                # df_CBIS["Status"] = df_CBIS['Status'].replace(['Benign', 'Normal'], 'N')
                # df_CBIS["Status"] = df_CBIS['Status'].replace(['Cancer'], 'M')
                # print(len(df_CBIS))
                df_CBIS.drop(df_CBIS.loc[df_CBIS['Status'] == 'Normal'].index, inplace=True)
                # print(len(df_CBIS))

                df_CBIS["Status"] = df_CBIS['Status'].replace(['Benign'], 'B')
                df_CBIS["Status"] = df_CBIS['Status'].replace(['Cancer'], 'M')

                df_CBIS = df_CBIS[["fullPath", "Status", "fileName"]]
                folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
                path_folder_to_read_images = path_base + folder_base_images
                for index, row in df_CBIS.iterrows():
                    img = cv2.imread(path_folder_to_read_images + row['fullPath'], cv2.IMREAD_GRAYSCALE)
                    # cv2.imshow('image', img)
                    # cv2.waitKey(0)
                    img = cv2.resize(img, (224, 224))

                    if row['Status'] == 'B':
                        cv2.imwrite(path_binary_classification_benigno + "/" + row['fileName'], img)
                    else:  # M
                        cv2.imwrite(path_binary_classification_maligno + "/" + row['fileName'], img)
                    # break

    # toDo: Crear directorio donde tendra 3 carpetas separando las 3 Class Benigno, maligno y estado normal
    else:
        pass
