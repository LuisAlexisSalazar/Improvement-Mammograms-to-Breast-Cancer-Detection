DEBUG_MODE_MODELS = True
DEBUG_MODE = False
DEFAULT_DATASET_USE = "MIAS"  # MINI-DDSM, CBIS-DDSM
DEFAULT_DATASET_URL = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
PATH_DATA_RAW = "data/dataset_raw/"

# ?Leyenda: N: Normal, B: Benigno y M:Maligno
# *Tipos de descargas: BinaryNM , BinaryBM, , BinaryBN , Binary(BM)N ,ClassBMN
MODE_DOWNLOAD_DEFAULT = "BinaryNM"
MODE_CLASSIFICATION_DEFAULT = "BinaryNM"

ALL_MODE_DOWNLOAD = ["BinaryNM", "BinaryBM", "BinaryBN", "Binary(BM)N", "ClassBMN"]
MODES_BINARY_CLASS = ["BinaryNM", "BinaryBM", "BinaryBN", "Binary(BM)N"]
MODE_3_CLASS = "ClassBMN"
LOG_JSON_TXT_MODEL_MLFlow = True
LOG_MODEL_MLFlow = False
