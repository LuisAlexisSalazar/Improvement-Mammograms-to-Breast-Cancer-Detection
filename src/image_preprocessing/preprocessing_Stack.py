import cv2
import numpy as np
from src.settings.config import SHOW_STEP_STEP_IMAGE
from skimage.segmentation import flood as flood
from src.image_preprocessing.preprocessing_DDSM_lishen import DMImagePreprocessor

SHOW_STEP_STEP_IMAGE = False

max_width = 3481
max_height = 2746
min_width = 1088
min_height = 495


def seeOpenCV2(img):
    return cv2.resize(img, (244, 244))


# old MIAS
# max_width = 800
# max_height = 1024
# *segunda fuente ipynb
def keep_right_orient_mammogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2):])

    if left_nonzero < right_nonzero:
        image = cv2.flip(image, 1)

    return image


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


def cropRoi(img):  # making triangle
    end = -1
    # for i in range(800):
    w, h = img.shape

    for i in range(w):
        if img[10][i] < 10:
            end = i
            break
    # myROI = [(0, 800), (0, 1023), (800, 1023), (800, 0), (end, 0)]
    myROI = [(0, 950), (0, h), (w, h), (w, 0), (end, 0)]
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
    w, h = img.shape
    # print(w, h)
    if h < w:
        w = h
    # w, h = w-1, h-1
    maxi = 0
    for i in range(w):
        for j in range(w):
            if img[i][j] > maxi:
                if (all(x > maxi for x in img[i, j:j + pix])) and (all(y > maxi for y in img[i:i + pix, j])):
                    maxi = img[i][j]
                    seed = (i, j)

    return seed


def checkWidth(mimg):
    w, h = mimg.shape
    if h < w:
        w = h
    img = mimg.copy()
    steepDist = 0
    steepLoc = 0
    steepThresh = 30
    for i in range(w):
        for j in range(w - 1, 0, -1):
            localdist = 0
            if img[i][j]:
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
    w, h = img1.shape
    # if h < w:
    #     w = h
    for i in range(w):
        for j in range(h):
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


def findDiff2(img1, img2):
    w, h = img1.shape
    img3 = img2.copy()
    for i in range(w):
        for j in range(h):
            if img1[i][j] == 0:
                img3[i][j] = 0
    #                 img2 =0
    #     plt.imshow(img2)
    return img3


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


def aditiveChangeWhiteToBlack(img):
    th = 255  # defines the value below which a pixel is considered "black"
    #  replace blanco a negro in RGB
    # black_pixels = np.where(
    #     (img[:, :, 0] == th) &
    #     (img[:, :, 1] == th) &
    #     (img[:, :, 2] == th)
    # )
    # img[black_pixels] = [0, 0, 0]

    img = np.where(img == th, 0, img)
    return img


# Primer stack
# https://stackoverflow.com/questions/53829896/opencv-error-215assertion-failed-src-type-cv-8uc1-in-function-cve
def preprocessing_DDSM(path_file):
    # read image
    img = cv2.imread(path_file)
    hh, ww = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # np.savetxt('output.txt', gray, fmt='% 3d')
    # *Eliminar artefactos que son netamente blancos
    gray = aditiveChangeWhiteToBlack(gray)
    # print(gray)
    # print(gray.shape)

    # shave 40 pixels all around
    gray = gray[40:hh - 40, 40:ww - 40]

    # add 40 pixel black border all around
    gray = cv2.copyMakeBorder(gray, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)

    # apply otsu thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw largest contour as white filled on black background as mask
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # dilate mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (305, 305))
    # *El valor fue cambiado porque era mucho tedjio de piel que resaltaba en la mamografia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (157, 157))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # apply mask to image: entender el bitwise_and
    # https://omes-va.com/operadores-bitwise/
    img_without_art = cv2.bitwise_and(img, img, mask=mask)
    # *Manteniendo la orientación unica
    img_out_art_rigth_orient = keep_right_orient_mammogram(img_without_art)

    # + ---------------Usando implementacón del ipynb Ananthan https://www.kaggle.com/code/ananthan123/breast-cancer-prediction-pt-1---------------
    enhanced_img = clahe(img_out_art_rigth_orient)
    # return enhanced_img

    # # *Remove pectoral
    # img_remove_pectoral = newRemove(enhanced_img)
    # # * Remover ruido residual
    # img_result = smoothImg(enhanced_img, img_remove_pectoral)
    # + --------------Usando imeplemntación de lishen https://github.com/lishen/end2end-all-conv/blob/master/EDA.ipynb---
    # dm_img_pproc = DMImagePreprocessor()
    # dm_img_pproc.only_remove_pectoral_muscle(img_out_art_rigth_orient,mask)

    # save results
    # cv2.imwrite('mammogram_thresh.jpg', thresh)
    # cv2.imwrite('mammogram_morph2.jpg', morph)
    # cv2.imwrite('mammogram_mask2.jpg', mask)
    # cv2.imwrite('mammogram_result2.jpg', result)

    # *Solo imeplmentación de ipynb Ananthan
    if SHOW_STEP_STEP_IMAGE:
        # enhanced_img = cv2.resize(enhanced_img, (224, 224))
        # cv2.imshow('Img', seeOpenCV2(img))
        # cv2.imshow('Blanco a Negro', seeOpenCV2(gray))
        # # cv2.imshow('Tresh', seeOpenCV2(thresh))
        # # cv2.imshow('Morfologia', seeOpenCV2(morph))
        # # cv2.imshow('Mask', seeOpenCV2(mask))
        # cv2.imshow('img_result', seeOpenCV2(img_without_art))
        cv2.imshow('img_result_orient', seeOpenCV2(img_out_art_rigth_orient))

        # enhanced_img = seeOpenCV2(img_out_art_rigth_orient)
        # cv2.imshow('img_out_art_rigth_orient', enhanced_img)
        cv2.waitKey(0)
    return seeOpenCV2(img_out_art_rigth_orient)
    #     result = cv2.resize(result, (224, 224))
    #     enhanced_img = cv2.resize(enhanced_img, (224, 224))
    #     img_remove_pectoral = cv2.resize(img_remove_pectoral, (224, 224))
    #     img_result = cv2.resize(img_result, (224, 224))
    #     cv2.imshow('img mantener orientación', result)
    #     cv2.imshow('Mejora de contraste', enhanced_img)
    #     cv2.imshow('img_remove_pectoral', img_remove_pectoral)
    #     cv2.imshow('img_result', img_result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
