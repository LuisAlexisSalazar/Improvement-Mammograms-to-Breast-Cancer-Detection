from src.utils.utils import *
from src.settings import config
import wget
import re

URL_MIAS = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
# !Error la usar el link mejor es descargar de manera manual
URL_MINI_DDSM = "https://www.kaggle.com/datasets/cheddad/miniddsm2/download"


def download_data(dataset=config.DEFAULT_DATASET_USE):
    path_data_set = config.PATH_DATA_RAW  # "data/dataset_raw/"
    URL = config.DEFAULT_DATASET_URL

    # --MIAS
    if dataset == "MIAS":
        URL = URL_MIAS
        path_data_set = path_data_set + "MIAS/"
        name_file = "MIAS.tar.gz"
        files = [f for f in os.listdir(path=path_data_set) if f == name_file]
        if len(files) == 0:
            wget.download(URL, path_data_set + name_file)
        # * descomprimir el archivo .tar.gz
        files_pgm = [f for f in os.listdir(path=path_data_set) if re.match(r'mdb[0-9]*\.pgm', f)]

        if len(files_pgm) == 0:
            unzip_tar(path_data_set, name_file)

    # --MINI-DDSM
    # ?En el caso de kaggle necesita ya estar descargada el comprimido zip, el codigo descomprime
    elif dataset == "MINI-DDSM":
        path_data_set = path_data_set + "MINI-DDSM/"
        name_file = "MINI-DDSM.zip"
        expected_folder = ["Data-MoreThanTwoMask", "MINI-DDSM-Complete-JPEG-8", "MINI-DDSM-Complete-PNG-16"]
        folders = [f for f in os.listdir(path=path_data_set) if f in expected_folder]

        if len(folders) != 3:
            if name_file in os.listdir():
                unzip_zip(path_data_set, name_file)
            else:
                print(
                    "Debes de descargar manualmente desde kaggle " +
                    URL_MINI_DDSM +
                    " y ponerlo en la dirección data/data_raw/MINI-DDSM/"
                )
