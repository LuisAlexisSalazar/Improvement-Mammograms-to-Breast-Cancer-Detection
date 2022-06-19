DEBUG_MODE_MODELS = True
DEFAULT_DATASET_USE = "MIAS"  # MINI-DDSM, CBIS-DDSM
DEFAULT_DATASET_URL = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
PATH_DATA_RAW = "data/dataset_raw/"

# ?Leyenda: N: Normal, B: Benigno y M:Maligno
# *Tipos de descargas: BinaryNM , BinaryBM, , BinaryBN , Binary(BM)N ,ClassBMN
MODE_DOWNLOAD_DEFAULT = "BinaryNM"

ALL_MODE_DOWNLOAD = ["BinaryNM", "BinaryBM", "BinaryBN", "Binary(BM)N", "ClassBMN"]
