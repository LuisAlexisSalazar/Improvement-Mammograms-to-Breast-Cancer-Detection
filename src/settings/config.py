DEBUG_MODE_MODELS = True
DEBUG_MODE = False
DEFAULT_DATASET_USE = "MIAS"  # MINI-DDSM, CBIS-DDSM
DEFAULT_DATASET_URL = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
PATH_DATA_RAW = "data/dataset_raw/"
PATH_DATA_PRE_PROCESSING = "data/dataset_preprocessing/"
MODE_DOWNLOAD_DEFAULT = "BinaryNM"
# ?Leyenda: N: Normal, B: Benigno y M:Maligno
# *Tipos de descargas: BinaryNM , BinaryBM, , BinaryBN , Binary(BM)N ,ClassBMN

MODE_CLASSIFICATION_DEFAULT = "BinaryNM"

ALL_MODE_DOWNLOAD = ["BinaryNM", "BinaryBM", "BinaryBN", "Binary(BM)N", "ClassBMN"]
MODES_BINARY_CLASS = ["BinaryNM", "BinaryBM", "BinaryBN", "Binary(BM)N"]
MODE_3_CLASS = "ClassBMN"
LOG_JSON_TXT_MODEL_MLFlow = True
LOG_MODEL_MLFlow = False

# --Module processing
SHOW_STEP_STEP_IMAGE = False
PATH_DATA_PREPROCESSING = "data/dataset_preprocessing/"
# -- Module texture descriptor
USE_DESCRIPTOR = False
APPLY_DESCRIPTOR = True
RADIUS = 2
# Juntos sera 8*2 = 16
# ?Solo se debe modificar el Q_points
N_POINTS = 8
Q_POINTS = 2
