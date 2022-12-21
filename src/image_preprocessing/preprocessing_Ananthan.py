import os
import math

#
#
from copy import copy

from skimage.segmentation import flood as flood
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import skew
from scipy.stats import kurtosis
import scipy.ndimage as ndi
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn import svm
from skimage.measure import shannon_entropy
import cv2
from skimage import io
from skimage import color
from skimage.draw import polygon
from skimage.feature import greycomatrix, greycoprops
from sklearn.decomposition import PCA

# from skimage.feature import canny
# from skimage.filters import sobel
# from skimage.transform import hough_line, hough_line_peaks


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from IPython.display import Image

from src.settings.config import SHOW_STEP_STEP_IMAGE

SHOW_STEP_STEP_IMAGE = True
LIST_INDEX_TO_FLIP_MANUAL = [36, 76, 102, 153, 154, 259, 267, 275, 277, 289, 291, 293, 311, 329]
LIST_INDEX_TO_CROPTOP_MANUAL = [1, 10, 308]
LIST_INDEX_TO_REMOVEPECTORALMANUAL = [34, 94, 116, 138, 163, 171, 222, 296, 312, 315, 322]


# new1[:934,:] = topCropped[1][90:,:]
# new2[:1002,:] = topCropped[10][22:,:]
# new3[:1009,:] = topCropped[308][15:,:]

# ----------------------Implementación de ANANTHAN123 ---------------------------------
def right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2):])

    print(left_nonzero, "\t", right_nonzero)
    if (left_nonzero < right_nonzero):
        print("Espejo")
        image = cv2.flip(image, 1)

    return image


def forceFlip(img):
    img = cv2.flip(img, 1)
    return img


def cropLeft(img):
    mini = 0
    pix = 0

    for i in range(1024):
        if img[10][i] > 10:
            mini = i
            break
    newimg = np.ones((1024, 800)) * pix
    #     print(min,img[10][200])
    diff = 1024 - mini
    if diff < 800:
        newimg[:, :diff] = img[:, mini:]
    #         print('in cropleft : shape :', newimg.shape)
    elif diff > 800:
        newimg = img[:, mini:(mini + 800)]
    elif diff == 800:
        newimg = img[:, mini:]
    #     print(newimg.shape)

    #     plt.imshow(newimg)
    return newimg


def removeArt(img):
    thresh = 30
    #     thresh = 10
    minPix = 40

    #     print(start)

    newim = np.zeros((1024, 800))
    newim = img.copy()
    for i in range(1024):
        start = 0
        for k in range(800):
            if img[100][k] != 0:
                start = k
                break
        for j in range(start, 800 - minPix):
            #         for j in range(768-minPix):
            ar = newim[i][j:j + minPix]
            if (all(x < thresh for x in ar)):
                newim[i][j:] = 0
                break
    #     plt.imshow(newim)
    return newim


def cropTop(img):
    newim = np.zeros((1024, 800))
    newim[:1023, :] = img[1:, :]
    return newim


# https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
# cv2.createCLAHE
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


# *Functions to Remove pectoral
# !Usar las imagenes de 1024 x 1024
# https://stackoverflow.com/questions/11270250/what-does-the-python-interface-to-opencv2-fillpoly-want-as-input
def cropRoi(img):  # making triangle
    end = -1
    for i in range(800):
        if img[10][i] < 10:
            end = i
            break
    myROI = [(0, 800), (0, 1023), (800, 1023), (800, 0), (end, 0)]
    img = cv2.fillPoly(img, [np.array(myROI)], 0)
    return img


def newClust(knimage):
    #     print(knimage.shape)
    #     knimage = cv2.cvtColor(knimage,cv2.COLOR_GRAY2RGB)
    vectorized = knimage.reshape((-1, 3))
    pixel_values = vectorized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(knimage.shape)
    return segmented_image


def findSeed(img):
    #     thresh = 162
    pix = 30
    seed = (10, 30)

    maxi = 0
    for i in range(800):
        for j in range(800):
            if img[i][j] > maxi:
                if (all(x > maxi for x in img[i, j:j + pix])) and (all(y > maxi for y in img[i:i + pix, j])):
                    maxi = img[i][j]
                    seed = (i, j)

    return seed


def read_image(path_file):
    img = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    return img


def checkWidth(mimg):
    img = mimg.copy()
    steepDist = 0
    steepLoc = 0
    steepThresh = 30
    for i in range(800):
        for j in range(799, 0, -1):
            localdist = 0
            if img[i][j] == True:
                localdist = j
                if steepDist == 0:
                    steepDist = localdist
                else:
                    if localdist - steepDist > steepThresh:
                        img[steepLoc:, :] = 0
                        return img
                    if localdist < steepDist:
                        steepDist = localdist
                        steepLoc = i
                localdist = 0
                break
    return img


def findDiff(img1, img2):
    for i in range(1024):
        for j in range(800):
            if img1[i][j] > 0:
                img2[i][j] = 0
    #                 img2 =0
    #     plt.imshow(img2)
    return img2


def newRemove(mimg):
    img = mimg.copy()
    crpimg = cropRoi(img)
    fimg = np.float32(crpimg)
    gimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2RGB)
    clustimg = newClust(gimg)
    cvimg = cv2.cvtColor(clustimg, cv2.COLOR_BGR2GRAY)
    seed = findSeed(cvimg)
    floodimg = flood(cvimg, seed)
    widthImg = checkWidth(floodimg)
    cpimg = mimg.copy()
    result = findDiff(widthImg, cpimg)
    # ?UNCOMMENT BELOW LINES TO SEE WHAT'S HAPPENING IN EACH STEP ####

    #     fig, axes = plt.subplots(1, 5, figsize=(15,10))
    #     fig.tight_layout(pad=3.0)
    #     plt.xlim(0,img.shape[1])
    #     plt.ylim(img.shape[0])

    #     axes[0].set_title('original')
    #     axes[0].imshow(mimg, cmap='gray')
    #     axes[0].axis('on')

    #     axes[1].set_title('Segmented')
    #     axes[1].imshow(clustimg, cmap='gray')
    #     axes[1].axis('on')

    #     axes[2].set_title('flooding')
    #     axes[2].imshow(floodimg, cmap='gray')
    #     axes[2].axis('on')

    #     axes[3].set_title('width check')
    #     axes[3].imshow(widthImg, cmap='gray')
    #     axes[3].axis('on')

    #     axes[4].set_title('Pectoral muscle removed')
    #     axes[4].imshow(result, cmap='gray')
    #     axes[4].axis('on')
    #     plt.show()
    return result


# ?Funciones para remover pecho pectoral manual de algunas mamografías
def checkWidthManual(mimg, steep):
    img = mimg.copy()
    steepDist = 0
    steepLoc = 0
    steepThresh = steep
    for i in range(800):
        for j in range(799, 0, -1):
            localdist = 0
            if img[i][j] == True:
                localdist = j
                if steepDist == 0:
                    steepDist = localdist
                else:

                    if localdist - steepDist > steepThresh:
                        img[steepLoc:, :] = 0
                        return img
                    if localdist < steepDist:
                        steepDist = localdist
                        steepLoc = i
                localdist = 0
                break

    return img


def newRemoveManual(mimg, steep):
    img = mimg.copy()
    crpimg = cropRoi(img)
    fimg = np.float32(crpimg)
    gimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2RGB)
    clustimg = newClust(gimg)
    cvimg = cv2.cvtColor(clustimg, cv2.COLOR_BGR2GRAY)
    seed = findSeed(cvimg)
    floodimg = flood(cvimg, seed)
    widthImg = checkWidthManual(floodimg, steep)
    cpimg = mimg.copy()
    result = findDiff(widthImg, cpimg)
    return result


# ?Remoer el ruido residual
def findDiff2(img1, img2_withNoise):
    img3 = img2_withNoise.copy()
    for i in range(1024):
        for j in range(800):
            if img1[i][j] == 0:
                img3[i][j] = 0
    return img3


# https://docs.opencv.org/4.5.2/d5/daf/tutorial_py_histogram_equalization.html
# enhancedImg : La imagen pasada con clahe
# newSliced Es la imagen mejorada es la imagen pasada con la función newRemoveManual o newRemove
def smoothImg(enhancedImg, img_remove_pectoral):
    ite = 15
    ke = 3
    image = img_remove_pectoral
    # image = newSliced[im]
    image = image.astype("uint8")
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Filter using contour area and remove small noise
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ke, ke))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=ite)
    result = findDiff2(close, enhancedImg)
    # result = findDiff2(close, enhancedImg[im])
    # plt.imshow(result)
    return result


# *Función que ejecuta de manera secuencial las demas funciones
def execute_ananthan(path_file):
    img = read_image(path_file)
    img_right = right_orient_mammogram(img)
    # img_crop_left = cropLeft(img_right)
    # img_without_art = removeArt(img_crop_left)
    img_without_art = removeArt(img_right)
    # img_crop_top = cropTop(img_without_art)
    # enhanced_img = clahe(img_crop_top)
    # *Try Mini-DDSM
    # img_crop_top = cropTop(img_without_art)
    enhanced_img = clahe(img_without_art)

    # *Remove pectoral
    img_remove_pectoral = newRemove(enhanced_img)
    # * Remover ruido residual
    img_result = smoothImg(enhanced_img, img_remove_pectoral)

    if SHOW_STEP_STEP_IMAGE:
        img = cv2.resize(img, (224, 224))
        img_right = cv2.resize(img_right, (224, 224))
        enhanced_img = cv2.resize(enhanced_img, (224, 224))
        img_remove_pectoral = cv2.resize(img_remove_pectoral, (224, 224))
        img_result = cv2.resize(img_result, (224, 224))
        cv2.imshow('original', img)
        cv2.imshow('img_right', img_right)
        cv2.imshow('enhanced_img', enhanced_img)
        cv2.imshow('img_remove_pectoral', img_remove_pectoral)
        cv2.imshow('img_result', img_result)
        cv2.waitKey()

    # cv2.imwrite("Resultado-Mini-DDSM.png", img_result)
    # return img_result
