#!/usr/bin/env python3
# https://stackoverflow.com/questions/48772621/cannot-find-reference-for-opencv-functions-in-pycharm
# https://stackoverflow.com/questions/72583781/im-getting-an-import-error-does-anyone-know-the-solution
import os

import cv2.cv2 as cv2
from brint import Brint, show_img_result, run_gpu_brint_s, run_gpu_brint_m, show_imgs_titles
import numpy as np

import time
from PIL import Image

# ctrl + shift + i show definición
# ctrl + B : travel to implementation
from src.settings.config import RADIUS, Q_POINTS


def read_img_out(file="out2.tif"):
    img = cv2.imread(file)
    b, g, r = cv2.split(img)
    titles = ["Imagen Procesada", "Brint S", "Brint M"]
    imgs = [b, g, r]

    for i in imgs:
        print(i.shape)
    # cv2.imwrite("original.jpg", b)
    show_imgs_titles(imgs, titles)


def execute_single_brint_m_s(
        file_img="E:\\U\Improvement-Mammograms-to-Breast-Cancer-Detection\data\dataset_preprocessing\MINI-DDSM\ClassBMN\\benigno\C_0253_1.LEFT_CC.jpg"
):
    # file_img = "E:\\U\Improvement-Mammograms-to-Breast-Cancer-Detection\data\dataset_raw\MINI-DDSM\MINI-DDSM-Complete-JPEG-8\Benign\\0253\C_0253_1.LEFT_CC.jpg"
    image = cv2.imread(file_img, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread(file_img, -1)
    # print(image.shape)
    image = cv2.resize(image, (224, 224))

    # cv2.imshow("Img", image)
    # cv2.waitKey(0)
    # np.savetxt("pgm.txt", image, fmt='%d')

    # img_brint_s = brint.run_brint_s(image)
    # img_brint_s =run_gpu_brint_s(image,radius=1,q_points=2)
    img_brint_s = run_gpu_brint_s(image, radius=1, q_points=2)
    img_brint_m = run_gpu_brint_m(image, radius=1, q_points=2)
    # t2 = time.time()
    # print(image.shape)
    # print(img_brint_m.shape)
    # *************Gaurdar las 3 imagenes en grises en 1 **************
    # https://stackoverflow.com/questions/71018586/combine-3-grayscale-images-to-1-natural-color-image-in-python
    rgb = np.dstack((image, img_brint_s, img_brint_m)).astype(np.uint8)
    cv2.imwrite("out2.tif", rgb)
    # cv2.imwrite("out3.jpg", rgb)
    # *************Imprimir las 3 imagenes**************
    # titles = ["Imagen Procesada", "Brint S", "Brint M"]
    # imgs = [image, img_brint_s, img_brint_m]
    # show_imgs_titles(imgs, titles)
    # ***************************


def run_brint_m_s(img):
    img_brint_s = run_gpu_brint_s(img, radius=RADIUS, q_points=Q_POINTS)
    img_brint_m = run_gpu_brint_m(img, radius=RADIUS, q_points=Q_POINTS)
    merge_img = np.dstack((img, img_brint_s, img_brint_m)).astype(np.uint8)
    print(merge_img.shape)
    return merge_img


# ? Imagenes en formato .TIF mantiene mejor los formatos en imagenes medicas
# https://stackoverflow.com/questions/26929052/how-to-save-an-array-as-a-grayscale-image-with-matplotlib-numpy
# https://stackoverflow.com/questions/31862958/how-to-merge-images-together-in-python-with-pil-or-something-else-with-each-im
# ? Explicación de la interpoaclión lineal para LBP
# https://medium.com/swlh/local-binary-pattern-algorithm-the-math-behind-it-%EF%B8%8F-edf7b0e1c8b3

def generate_dataset():
    name_data_set = "MINI-DDSM"
    # folder_preprocessing = "ClassBMN_remover_artefactos_mantiene_muscle"
    folder_preprocessing = "ClassBMN2"
    path_folder_root_to_read = "E:\\U\Improvement-Mammograms-to-Breast-Cancer-Detection\data/dataset_preprocessing/" + name_data_set + "/" + folder_preprocessing + "/"
    folder = os.listdir(path_folder_root_to_read)
    path_abs_to_save_imgs = "E:\\U\Improvement-Mammograms-to-Breast-Cancer-Detection\data/dataset_roi/" + name_data_set + "/ClasBMN/"
    for sub_folder in folder:
        path_read_img = path_folder_root_to_read + sub_folder + "/"
        files = os.listdir(path_read_img)
        path_to_save_imgs = path_abs_to_save_imgs + sub_folder + "/"

        for f in files:
            path_img = path_read_img + f
            print(path_img)

            img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            merge_img = run_brint_m_s(img)
            f = f[:-3] + "tif"
            path_img_to_save = path_to_save_imgs + f
            cv2.imwrite(path_img_to_save, merge_img)
            # break


if __name__ == '__main__':
    pass
    # ?-----------Probar funciones del descriptor---------------
    # file = "E:\\U\Improvement-Mammograms-to-Breast-Cancer-Detection\data\dataset_preprocessing\MIAS\ClassBMN_remover_artefactos_mantiene_muscle\\benigno\mdb002.jpg"
    # execute_single_brint_m_s(file)
    # read_img_out("out2.tif")

    # ?--------------------------------------------------------
