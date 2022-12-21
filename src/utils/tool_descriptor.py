#!/usr/bin/env python3
import matplotlib as plt
from numba import jit


def bin_to_decimal(bin):
    res = 0
    bit_num = 0
    for i in bin[::-1]:
        res += i << bit_num
        bit_num += 1
    return res


@jit(nopython=True)
def getPattern(value):
    if value >= 0:
        return 1
    else:
        return 0

from numba import jit
@jit(nopython=True)
def bit_rotate_right(value, length):
    """Cyclic bit shift to the right.
    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer
    """
    return (value >> 1) | ((value & 1) << (length - 1))


def generateTwoImgs(self, nameFile):
    # ! No es necesario pero interesante para otras funciones
    # self.newMatrix = np.float32(self.newMatrix)
    # --------------------
    plt.subplot(121), plt.imshow(self.img.matrixOriginal, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(self.newMatrix, cmap='gray')
    plt.title('LBP and ROI'), plt.xticks([]), plt.yticks([])

    plt.show()
    # -------------------------------
