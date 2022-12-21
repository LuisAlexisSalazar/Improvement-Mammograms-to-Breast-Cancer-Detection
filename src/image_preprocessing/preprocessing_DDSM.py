import cv2
import numpy as np
from src.settings.config import SHOW_STEP_STEP_IMAGE

SHOW_STEP_STEP_IMAGE = True


# SHOW_STEP_STEP_IMAGE = False
def read_image(path_file):
    img = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    return img


def apply_median_blur(img):
    img_med_blurred = cv2.medianBlur(img, 3)  # <<= para to tune!
    return img_med_blurred


def get_img_binary(img):
    global_threshold = 18
    mammo_binary = cv2.threshold(img, global_threshold,
                                 maxval=255, type=cv2.THRESH_BINARY)[1]
    return mammo_binary


def select_largest_obj(img_bin, lab_val=255, fill_holes=False,
                       smooth_boundary=False, kernel_size=15):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin(2D array): 2D numpy array of binary image.
        lab_val([int]): integer value used for the label of the largest
                        object. Default is 255.
        fill_holes([boolean]): whether fill the holes inside the largest
                               object or not. Default is false.
        smooth_boundary([boolean]): whether smooth the boundary of the
                                    largest object using morphological
                                    opening or not. Default is false.
        kernel_size([int]): the size of the kernel used for morphological
                            operation.
    '''
    n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8, ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)

    return largest_mask


def improvee_contrast(img):
    mammo_breast_equ = cv2.equalizeHist(img)
    return mammo_breast_equ


# Watershed segmentation
def get_img_binary_pectoral(mammo_breast_equ):
    pect_high_inten_thres = 200  # <<= para to tune!
    pect_binary_thres = cv2.threshold(mammo_breast_equ, pect_high_inten_thres,
                                      maxval=255, type=cv2.THRESH_BINARY)[1]
    return pect_binary_thres


def make_watershed(pect_binary_thres, mammo_breast_mask):
    pect_marker_img = np.zeros(pect_binary_thres.shape, dtype=np.int32)
    # Sure foreground.
    pect_mask_init = select_largest_obj(pect_binary_thres, lab_val=255,
                                        fill_holes=True, smooth_boundary=False)
    kernel_ = np.ones((3, 3), dtype=np.uint8)  # <<= para to tune!
    n_erosions = 7  # <<= para to tune!
    pect_mask_eroded = cv2.erode(pect_mask_init, kernel_, iterations=n_erosions)
    pect_marker_img[pect_mask_eroded > 0] = 255
    # Sure background - breast.
    n_dilations = 7  # <<= para to tune!
    pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_, iterations=n_dilations)
    pect_marker_img[pect_mask_dilated == 0] = 128
    # Sure background - background.
    pect_marker_img[mammo_breast_mask == 0] = 64
    return pect_marker_img


def fill_muscle_to_remove(mammo_breast_equ, pect_marker_img):
    mammo_breast_equ_3c = cv2.cvtColor(mammo_breast_equ, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mammo_breast_equ_3c, pect_marker_img)
    pect_mask_watershed = pect_marker_img.copy()
    mammo_breast_equ_3c[pect_mask_watershed == -1] = (0, 0, 255)
    pect_mask_watershed[pect_mask_watershed == -1] = 0
    return pect_mask_watershed, mammo_breast_equ_3c


def get_ROI(pect_mask_watershed, mammo_breast_equ):
    breast_only_mask = pect_mask_watershed.astype(np.uint8)
    breast_only_mask[breast_only_mask != 128] = 0
    breast_only_mask[breast_only_mask == 128] = 255
    kn_size = 25  # <<= para to tune!
    kernel_ = np.ones((kn_size, kn_size), dtype=np.uint8)
    breast_only_mask_smo = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, kernel_)
    mammo_breast_only = cv2.bitwise_and(mammo_breast_equ, breast_only_mask_smo)
    return mammo_breast_only


def execute(path_file):
    img = read_image(path_file)
    # if img == None:
    #     return
    # cv2.imshow("sad", img)
    # cv2.waitKey()
    mammo_med_blurred = apply_median_blur(img)
    mammo_binary = get_img_binary(mammo_med_blurred)

    mammo_breast_mask = select_largest_obj(mammo_binary, lab_val=255,
                                           fill_holes=True,
                                           smooth_boundary=True, kernel_size=15)

    mammo_arti_suppr = cv2.bitwise_and(mammo_med_blurred, mammo_breast_mask)
    mammo_breast_equ = improvee_contrast(mammo_arti_suppr)
    # *Remove pectural muscle
    pect_binary_thres = get_img_binary_pectoral(mammo_breast_equ)
    pect_marker_img = make_watershed(pect_binary_thres, mammo_breast_mask)

    pect_mask_watershed, mammo_breast_equ_3c = fill_muscle_to_remove(mammo_breast_equ, pect_marker_img)
    mammo_breast_only = get_ROI(pect_mask_watershed, mammo_breast_equ)

    if SHOW_STEP_STEP_IMAGE:
        cv2.imshow('original', img)
        cv2.imshow('mammo_breast_mask', mammo_breast_mask)
        cv2.imshow('mammo_arti_suppr', mammo_arti_suppr)
        cv2.imshow('mammo_breast_equ', mammo_breast_equ)
        # cv2.imshow('pect_mask_watershed', pect_mask_watershed)
        # cv2.imshow('mammo_breast_only', mammo_breast_only)
        cv2.waitKey()

    return mammo_breast_only
