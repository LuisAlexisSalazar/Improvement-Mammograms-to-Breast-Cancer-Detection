import cv2
import numpy as np
import os

from skimage.filters.edges import sobel

from src.settings.config import SHOW_STEP_STEP_IMAGE, PATH_DATA_RAW
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage import io
from skimage import color

SHOW_STEP_STEP_IMAGE = False
# SHOW_STEP_STEP_IMAGE = True


def keep_right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2):])

    if left_nonzero < right_nonzero:
        image = cv2.flip(image, 1)

    return image


def read_image_IO(filename):
    image = io.imread(filename)
    # image = color.rgb2gray(image)
    # image = color.gra(image)
    image = keep_right_orient_mammogram(image)
    return image


# Canny: https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html
# Sobel: https://towardsdatascience.com/magic-of-the-sobel-operator-bbbcb15af20d
# https://medium.com/@haidarlina4/sobel-vs-canny-edge-detection-techniques-step-by-step-implementation-11ae6103a56a
def apply_canny(image):
    canny_img = canny(image, 6)
    return sobel(canny_img)
    # return canny_img


# Hough Transform: Detectar lineas rectas
# https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html
# https://www.youtube.com/watch?v=zbyn57jgWNg&ab_channel=IrvingVasquez
def get_hough_lines(canny_img):
    h, theta, d = hough_line(canny_img)
    lines = list()
    print('\nAll hough lines')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
        x1 = 0
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        x2 = canny_img.shape[1]
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        lines.append({
            'dist': dist,
            'angle': np.degrees(angle),
            'point1': [x1, y1],
            'point2': [x2, y2]
        })

    return lines


def short_lines_from_hough_transform(lines):
    MIN_ANGLE = 10
    MAX_ANGLE = 70
    MIN_DIST = 5
    MAX_DIST = 200
    # MAX_DIST = 150

    shortlisted_lines = [x for x in lines if
                         (x['dist'] >= MIN_DIST) &
                         (x['dist'] <= MAX_DIST) &
                         (x['angle'] >= MIN_ANGLE) &
                         (x['angle'] <= MAX_ANGLE)
                         ]
    print('\nShorlisted lines')
    for i in shortlisted_lines:
        print("Angle: {:.2f}, Dist: {:.2f}".format(i['angle'], i['dist']))

    return shortlisted_lines


from skimage.draw import polygon


def remove_pectoral(shortlisted_lines):
    shortlisted_lines.sort(key=lambda x: x['dist'])
    pectoral_line = shortlisted_lines[0]
    d = pectoral_line['dist']
    theta = np.radians(pectoral_line['angle'])

    x_intercept = d / np.cos(theta)
    y_intercept = d / np.sin(theta)

    return polygon([0, 0, y_intercept], [0, x_intercept, 0])


import matplotlib.pyplot as plt
from matplotlib import pylab as pylab


# Read the input
class Preprocessing():
    def __init__(self):
        self.w = 224
        self.h = 224
        self.otsu_threshold = cv2.THRESH_OTSU

        # https://stackoverflow.com/questions/62855718/why-would-cv2-color-rgb2gray-and-cv2-color-bgr2gray-give-different-results
        # cv2.COLOR_RGB2GRAY and cv2.COLOR_BGR2GRAY

    def read_image(self, path_file):
        # Lee con 1 solo canal
        img = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.h, self.w))
        # 224, 224
        # height, width = img.shape
        return img

    def otsu_thresh(self, img):
        thresh = cv2.threshold(src=img, thresh=0, maxval=255, type=self.otsu_threshold)[1]
        # thresh = cv2.threshold(src=img, thresh=15, maxval=255, type=self.otsu_threshold)[1]
        # print(thresh)
        return thresh

    def apply_morphology(self, thresh):
        # ?Tipos de morfologias
        # https://medium.com/analytics-vidhya/morphological-transformations-of-images-using-opencv-image-processing-part-2-f64b14af2a38
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

        # ?Morfologias close: Remover pequeños huecos dentro del objeto del primer plano
        # *objetivo de eliminar regiones pequeñas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        if SHOW_STEP_STEP_IMAGE:
            cv2.imshow('morph1', morph)

        # ?Morfología open: Util para remover ruido
        # *Separa pecho de otras regiones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        if SHOW_STEP_STEP_IMAGE:
            cv2.imshow('morph2', morph)

        # ?Morfología dilate: Incrementar la región del primer plano de la imagen
        # *Compensar lo que el Otsu que no alcanza algunas áreas
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)

        if SHOW_STEP_STEP_IMAGE:
            cv2.imshow('morph3', morph)
        return morph

    def get_mask(self, morph):
        # *Obtener los mas grandes contornos (mama y seno)
        # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b
        # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
        # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        # Describe la jerarquia https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
        # findContours detecta cambios en el color de la imagen y lo marca como contorno.
        # *hierarchy = [Next, Previous, First_Child, Parent]
        # ?Parametros: Fuente de imagen, Modo de recuperar los contornos y metodo de aproximación de contornos
        contours, hierarchy = cv2.findContours(image=morph, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        # print(hierarchy)
        # *Qudarse con el contorno del seno y no de la mamaografía
        # print(len(contours))
        # Calcula el area de control
        big_contour = max(contours, key=cv2.contourArea)
        # big_contour = cv2.contourArea(contours)

        # draw largest contour as white filled on black background as mask
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

        # dilate mask
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gac2db39b56866583a95a5680313c314ad
        # Tipos de Morph: cv.MORPH_CROSS, cv.MORPH_RECT y cv.MORPH_ELLIPSE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga7be549266bad7b2e6a04db49827f9f32
        # Tipos de Morphological: MORPH_DILATE,MORPH_ERODE
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        return mask

    # https://github.com/gsunit/Pectoral-Muscle-Removal-From-Mammograms
    # !Falta: APlciar morfologia y hacer la operación betweesi
    def own_remove_pectoral(self, file_name):
        value_thresh = 110
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # np.savetxt('test.txt', img, fmt='%i')
        # thresh = cv2.threshold(src=img, thresh=value_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # thresh = cv2.threshold(src=img, thresh=value_thresh, maxval=255, type=cv2.THRESH_OTSU)[1]
        thresh = cv2.threshold(src=blur, thresh=value_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
        # thresh = cv2.threshold(src=blur, thresh=value_thresh, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow('original', img)
        cv2.imshow('thresh', thresh)
        cv2.waitKey()

    # -------------Implementatión of gsuint-----------
    # !Errores el canny no separa el musculo pectoral

    def remove_pectoral_by_gsuint(self):
        pass

    def display_image(self, filename):
        # original_image = self.read_image(filename)
        image = read_image_IO(filename)
        # image = keep_right_orient_mammogram(original_image)

        # cv2.imshow('rigth orientation', image)

        canny_image = apply_canny(image)
        cv2.imshow('canny', canny_image)
        cv2.waitKey()
        # return
        lines = get_hough_lines(canny_image)
        # print("Lines:",lines)

        shortlisted_lines = short_lines_from_hough_transform(lines)

        fig, axes = plt.subplots(1, 4, figsize=(15, 10))
        fig.tight_layout(pad=3.0)
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0])

        axes[0].set_title('Right-oriented mammogram')
        axes[0].imshow(image, cmap=pylab.cm.gray)
        axes[0].axis('on')

        axes[1].set_title('Hough Lines on Canny Edge Image')
        axes[1].imshow(canny_image, cmap=pylab.cm.gray)
        axes[1].axis('on')
        axes[1].set_xlim(0, image.shape[1])
        axes[1].set_ylim(image.shape[0])
        for line in lines:
            axes[1].plot((line['point1'][0], line['point2'][0]), (line['point1'][1], line['point2'][1]), '-r')

        axes[2].set_title('Shortlisted Lines')
        axes[2].imshow(canny_image, cmap=pylab.cm.gray)
        axes[2].axis('on')
        axes[2].set_xlim(0, image.shape[1])
        axes[2].set_ylim(image.shape[0])
        for line in shortlisted_lines:
            axes[2].plot((line['point1'][0], line['point2'][0]), (line['point1'][1], line['point2'][1]), '-r')

        rr, cc = remove_pectoral(shortlisted_lines)
        rr = np.array([r for r in rr if r < len(image)])
        cc = cc[:len(rr)]
        # print(len(cc))
        # print(len(rr))
        # print(len(image))
        image[rr, cc] = 0
        axes[3].set_title('Pectoral muscle removed')
        axes[3].imshow(image, cmap=pylab.cm.gray)
        axes[3].axis('on')
        plt.show()

        # cv2.imshow('original', original_image)
        # cv2.imshow('imagen right orient', image)
        # cv2.imshow('thresh', canny_image)
        # rr, cc = remove_pectoral(shortlisted_lines)
        # image[rr, cc] = 0
        # cv2.imshow('result',image )
        # plt.show()

    # ----------------------Fin de Implementación basado en Canny---------------------------------

    def execute(self, path_file, path_to_save_result):
        name_file = path_file.split("/")[-1]
        # if not (name_file in os.listdir(path_to_save_result)):
        # if True:
        #  - Convert to grayscale
        #  - Otsu threshold
        #  - Morphology processing
        #  - Get largest contour from external contours
        #  - Draw all contours as white filled on a black background except the largest as a mask and invert mask
        #  - Apply the mask to the input image
        #  - Save the results
        # path_file = PATH_DATA_RAW + mode
        img = self.read_image(path_file)
        thresh = self.otsu_thresh(img)
        # morph = self.apply_morphology(thresh)
        # mask = self.get_mask(morph)
        # result = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imwrite(path_to_save_result + name_file, result)

        # if SHOW_STEP_STEP_IMAGE:
        if SHOW_STEP_STEP_IMAGE:
            cv2.imshow('original', img)
            cv2.imshow('thresh', thresh)
            # cv2.imshow('morph', morph)
            # cv2.imshow('mask', mask)
            # cv2.imshow('result', result)
            cv2.waitKey()


def pre_processing_folder():
    preprocessing = Preprocessing()

    pass
