import zipfile
import tarfile
import os
import pandas as pd
from src.settings.config import PATH_DATA_RAW


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
    df_mammogram.dropna(subset=['abn_class'], inplace=True)
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
