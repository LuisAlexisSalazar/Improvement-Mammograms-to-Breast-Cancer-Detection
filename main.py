import os

import cv2

from src.image_preprocessing.preprocessing_MIAS import Preprocessing
from src.settings.config import PATH_DATA_PREPROCESSING, PATH_DATA_RAW
from src.utils.folders import get_path_folder_to_read, get_paths_sub_folders, get_path_folder_to_result_preprocessing, \
    create_folder_if_not_exist, create_folders_to_mode
from src.utils.utils import get_files_from_folder, create_df_dataset_MINI_DDSM
import numpy as np
import src.image_preprocessing.preprocessing_Ananthan as prorcessingAn
import src.image_preprocessing.preprocessing_DDSM as prorcessingDDSM
from src.image_preprocessing.preprocessing_DDSM_lishen import DMImagePreprocessor
import src.image_preprocessing.preprocessing_MIAS as prepro_MIAS
import src.image_preprocessing.preprocessing_Stack as preproStack

mode_classification = "ClassBMN"
dataset = "MIAS"


def preprocessing_MIAS():
    # *Data Cruda:Obtener los files en base a las clases
    # path_folder = get_path_folder_to_read(mode_classification, dataset)
    # sub_folders = get_paths_sub_folders(path_folder)
    # list_pathFile_per_subfolder = get_files_from_folder(sub_folders)

    # Crear las carpetas donde se guardaran los resultados del procesamiento
    # path_folder_result = get_path_folder_to_result_preprocessing(mode_classification, dataset)
    # create_folder_if_not_exist(path_folder_result)
    # create_folders_to_mode(path_folder_result, mode_classification)
    # sub_folders = os.listdir(path_folder_result)

    # preprocessing = Preprocessing()
    #
    # for subFolder_path_file, subfolder in zip(list_pathFile_per_subfolder, sub_folders):
    #     path_to_save_result = path_folder_result + subfolder + "/"
    #     print(path_to_save_result)
    #     for path_file in subFolder_path_file:
    #         preprocessing.execute(path_file, path_to_save_result)

    # *Data Preprocesada sin etiqueta pero con muculo pectoral con la lcase preprocessing_MIAS
    # *se obtiene en  base a las clases
    # files_per_subfolders = []
    # for sub_folder in sub_folders:
    #     name_files_subfolder = os.listdir(path_folder_result + sub_folder + "/")
    #     for i in range(len(name_files_subfolder)):
    #         name_files_subfolder[i] = path_folder_result + sub_folder + "/" + name_files_subfolder[i]
    #     files_per_subfolders.append(name_files_subfolder)

    # *Data Raw sin separar las clases
    files_data_raw = []
    path_dataset_raw = PATH_DATA_RAW + dataset + "/" + dataset + "-RAW/"

    all_files = os.listdir(path_dataset_raw)
    # print((all_files[3])[-3:])
    files_pgm = list(filter(lambda f: f[-3:] == "pgm", all_files))
    # print(files_pgm)
    # print(len(files_pgm))

    # !Falta procesar las imagenes manuales que estan como indices
    for file_name in files_pgm:
        path_file = path_dataset_raw + file_name
        print(path_file)
        prorcessingAn.execute_ananthan(path_file)

    # Singgle Execution
    # file_name = files_per_subfolders[0][1]
    # file_name = list_pathFile_per_subfolder[0][1]

    # print(file_name)
    # preprocessing.display_image(file_name)
    # print(file_name)
    # prorcessingAn.execute_ananthan(file_name)

    # Mejor sin Gausiana: 115 THRESH_BINARY
    # Mejor con Gausiana: 115 THRESH_BINARY y Gaus 5,5

    # preprocessing = Preprocessing()
    # for subfolder in files_per_subfolders:
    #     for file in subfolder:
    #         print(file.strip("/")[-1])
    #         preprocessing.display_image(file)
    # break
    # break

    # preprocessing = Preprocessing()

    # path_file = "data/dataset_raw/MIAS/ClassBMN/benigno/mdb005.jpg"
    # path_result = "data/dataset_preprocessing/MIAS/ClassBMN/benigno/"
    # preprocessing.execute(path_file, path_result)


def preprocessing_MINI_DDSM():
    # file_name = "data/dataset_raw/MINI-DDSM/MINI-DDSM-Complete-PNG-16/Cancer/0001/C_0001_1.LEFT_MLO.png"
    # file_name = "src/image_preprocessing/test.png"
    file_name = "src/image_preprocessing/test_ddsm.jpg"
    # import os
    # print(os.path.isfile(file_name))
    prorcessingDDSM.execute(file_name)


def try_preprocessing():
    list_processing_manual = []
    df_MINI_DDSM = create_df_dataset_MINI_DDSM()
    # path_file = "data/dataset_raw/MINI-DDSM/MINI-DDSM-Complete-JPEG-8/Cancer/0001/C_0001_1.LEFT_MLO.jpg"
    df_MINI_DDSM["Status"] = df_MINI_DDSM['Status'].replace(['Benign'], 'B')
    df_MINI_DDSM["Status"] = df_MINI_DDSM['Status'].replace(['Cancer'], 'M')
    df_MINI_DDSM["Status"] = df_MINI_DDSM['Status'].replace(['Normal'], 'N')
    name_dataset = "MINI-DDSM"
    df_MINI_DDSM = df_MINI_DDSM[["fullPath", "Status", "fileName"]]
    folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
    path_folder_to_read_images = PATH_DATA_RAW + name_dataset + "/" + folder_base_images

    # width_list = []
    # height_list = []

    path_to_save_image = "data/dataset_preprocessing/MINI-DDSM/ClassBMN2"

    for index, row in df_MINI_DDSM.iterrows():
        # print("Index:", index)
        path_file = path_folder_to_read_images + row['fullPath']

        # print(path_file.split("\\"))
        # name_file = path_file.split("\\")[-1]
        # print(name_file)
        # toDO: Probar con el segundo stack

        image_preprocessing = preproStack.preprocessing_DDSM(path_file)

        # *-------Generar Dataset Procesador con MiniDDSM------------
        if row['Status'] == 'B':
            cv2.imwrite(path_to_save_image + "/benigno/" + row['fileName'], image_preprocessing)
            print("B:   ", row['fileName'])
        elif row['Status'] == 'M':
            cv2.imwrite(path_to_save_image + "/maligno/" + row['fileName'], image_preprocessing)
            print("M:   ", row['fileName'])
        else:
            cv2.imwrite(path_to_save_image + "/normal/" + row['fileName'], image_preprocessing)
            print("N:   ", row['fileName'])

        # -------------------------------------------

        # break
        # path_file = "data/dataset_raw/MINI-DDSM/MINI-DDSM-Complete-JPEG-8/Benign/0029/C_0029_1.LEFT_MLO.jpg"
        # print(path_file)
        # img = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

        # prorcessingDDSM.execute(path_file)
        # prorcessingAn.execute_ananthan(path_file)
        # instance_prepro = prepro_MIAS.Preprocessing()
        # instance_prepro.execute(path_file, None)
        # width_list.append(img.shape[0])
        # height_list.append(img.shape[1])

    # print("Maximo width:", max(width_list))
    # print("Maximo height:", max(height_list))
    # print("Min width:", min(width_list))
    # print("Min height:", min(height_list))
    # cv2.imshow("original", img)
    # cv2.imshow("procesada", img)
    # cv2.waitKey(0)
    # option = input("Preprocessing Manual:")

    # print(list_processing_manual)
    # with open(r'list_MiniDDSM_preprocessing.txt', 'w') as fp:
    #     for item in list_processing_manual:
    #         fp.write("%s\n" % item)


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# my_ROI = np.array([(0, 800), (0, 1023), (800, 1023), (800, 0), (300, 0)])
# im = np.zeros([1024, 1024], dtype=np.uint8)
# im = cv2.fillPoly(im, [np.array(my_ROI)], 255)
# plt.imshow(im)
# plt.show()


# preprocessing_MINI_DDSM()
# file_name = "data/dataset_raw/MINI-DDSM/MINI-DDSM-Complete-PNG-16/Cancer/0001/C_0001_1.LEFT_MLO.png"
# file_name = "data/dataset_raw/MINI-DDSM/MINI-DDSM-Complete-JPEG-8/Cancer/0038/C_0038_1.LEFT_MLO.jpg"
# file_name = "src/image_preprocessing/test_ddsm.jpg"
# file_name = "src/image_preprocessing/test.png"
# dm_img_pproc = DMImagePreprocessor()
# mammo_org = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# mammo_pproc, mammo_col = dm_img_pproc.process(mammo_org, pect_removal=True)
#
# cv2.imshow('mammo_org', mammo_org)
# cv2.imshow('mammo_pproc', mammo_pproc)
# cv2.imshow('mammo_col', mammo_col)
# cv2.waitKey()


# try_preprocessing()
def clahe(img, clip=2.0, tile=(8, 8)):
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    # toDo: Instalar la versión CUDA para ver si aparece el modulo cv2.cuda.createCLAHE
    # !No encuentra el modulo cv2.cuda.createCLAHE
    # *Versión de opencv2 4.5.5
    # https://docs.opencv.org/4.5.5/d8/d0e/group__cudaimgproc__hist.html#ga950d3228b77b368a452553dcf57308c0
    # clahe_create = cv2.cuda.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img


img = cv2.imread(
    "data\\dataset_raw\\MIAS\\ClassBMN\\benigno\\mdb002.jpg",
    cv2.IMREAD_GRAYSCALE)
print(img.shape)
# hh, ww = gray.shape[:2]
# gray = gray[40:hh - 40, 40:ww - 40]

# add 40 pixel black border all around
ret, thresh1 = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)


cv2.imshow("thresh",thresh1)
cv2.waitKey(0)

# cv2.imshow("img",img)
# cv2.waitKey(0)

# img2 = clahe(img)
# cv2.imwrite("E:\\U\\Semestre 10°\\PF3\\Imagenes Diapositivas Final\\mdb002_clahe.jpg", img2)
# cv2.imwrite("mdb002_clahe.jpg", img2)
# imgs = [img, img2]
# import matplotlib.pyplot as plt
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# ax1.imshow(imgs[0], cmap='gray')
# ax1.set_title("Imagen sin etiqueta")
# ax1.axis('off')
#
# ax2.imshow(imgs[1], cmap='gray')
# ax2.set_title("Imagen con CLAHE")
# ax2.axis('off')
#
# plt.show()

# cv2.imshow("img2",img2)
# cv2.imshow("img",img)
# cv2.waitKey(0)
