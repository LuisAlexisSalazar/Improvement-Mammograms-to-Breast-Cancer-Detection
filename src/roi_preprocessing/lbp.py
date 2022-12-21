# ? LBP: Local Binary Pattern
from src.utils.tool_descriptor import *
import numpy as np
import cv2


# https://www.fatalerrors.org/a/implementation-of-lbp-algorithm-in-python.html


def rule(valueNeigboor, valueCentral):
    if valueNeigboor >= valueCentral:
        return 1
    return 0


def cal_basic(img, i, j):
    sum = []
    pixelCenter = img[i][j]
    sum.append(rule(img[i - 1, j], pixelCenter))
    sum.append(rule(img[i - 1, j + 1], pixelCenter))
    sum.append(rule(img[i, j + 1], pixelCenter))
    sum.append(rule(img[i + 1, j + 1], pixelCenter))
    sum.append(rule(img[i + 1, j], pixelCenter))
    sum.append(rule(img[i + 1, j - 1], pixelCenter))
    sum.append(rule(img[i, j - 1], pixelCenter))
    sum.append(rule(img[i - 1, j - 1], pixelCenter))

    return sum


# ? window 3x3
class Lbp:
    # ? singles CPU
    def run(self, img):
        n_row, n_col = img.shape
        imgFilter = np.zeros([n_row, n_col])

        for i in range(n_row):
            for j in range(n_col):
                if i == 0 or j == 0 or i == self.height - 1 or j == self.width - 1:
                    continue
                else:
                    imgFilter[i, j] = bin_to_decimal(cal_basic(img, i, j))
        return imgFilter

    def generateHistogram(self):
        # !Necesario para "calcHist" de Float64 a Float32
        self.newMatrix = np.float32(self.newMatrix)

        hist = cv2.calcHist(self.newMatrix, [0], None, [256], [0, 256])
        plt.plot(hist, color='r')
        plt.title('Histograma en escala a grises')
        plt.show()

_radius = 1
_neighbors = 8

def Compute_LBP(Image):
    # Determine the dimensions of the input image.
    ysize, xsize = Image.shape
    # define circle of symetrical neighbor points
    angles_array = 2 * np.pi / _neighbors
    alpha = np.arange(0, 2 * np.pi, angles_array)
    # Determine the sample points on circle with radius R
    s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
    s_points *= _radius
    # s_points is a 2d array with 2 columns (y,x) coordinates for each cicle neighbor point
    # Determine the boundaries of s_points wich gives us 2 points of coordinates
    # gp1(min_x,min_y) and gp2(max_x,max_y), the coordinate of the outer block
    # that contains the circle points
    min_y = min(s_points[:, 0])
    max_y = max(s_points[:, 0])
    min_x = min(s_points[:, 1])
    max_x = max(s_points[:, 1])
    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    # so if radius = 1 then block size equal to 3*3
    bsizey = np.ceil(max(max_y, 0)) - np.floor(min(min_y, 0)) + 1
    bsizex = np.ceil(max(max_x, 0)) - np.floor(min(min_x, 0)) + 1
    # Coordinates of origin (0,0) in the block
    origy = int(0 - np.floor(min(min_y, 0)))
    origx = int(0 - np.floor(min(min_x, 0)))
    # Minimum allowed size for the input image depends on the radius of the used LBP operator.
    if xsize < bsizex or ysize < bsizey:
        raise Exception('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')
    # Calculate dx and dy: output image size
    # for exemple, if block size is 3*3 then we need to substract the first row and the last row which is 2 rows
    # so we need to substract 2, same analogy applied to columns
    dx = int(xsize - bsizex + 1)
    dy = int(ysize - bsizey + 1)
    # Fill the center pixel matrix C.
    C = Image[origy:origy + dy, origx:origx + dx]
    # Initialize the result matrix with zeros.
    result = np.zeros((dy, dx), dtype=np.float32)
    # print(s_points)
    # return 0
    for i in range(s_points.shape[0]):
        # Get coordinate in the block:
        p = s_points[i][:]

        y, x = p + (origy, origx)
        # Calculate floors, ceils and rounds for the x and ysize
        fx = int(np.floor(x))
        fy = int(np.floor(y))
        cx = int(np.ceil(x))
        cy = int(np.ceil(y))
        rx = int(np.round(x))
        ry = int(np.round(y))
        D = [[]]
        if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
            # Interpolation is not needed, use original datatypes
            N = Image[ry:ry + dy, rx:rx + dx]
            D = (N >= C).astype(np.uint8)
        else:
            # interpolation is needed
            # compute the fractional part.
            ty = y - fy
            tx = x - fx
            # compute the interpolation weight.
            w1 = (1 - tx) * (1 - ty)
            w2 = tx * (1 - ty)
            w3 = (1 - tx) * ty
            w4 = tx * ty
            # compute interpolated image:
            N = w1 * Image[fy:fy + dy, fx:fx + dx]
            N = np.add(N, w2 * Image[fy:fy + dy, cx:cx + dx], casting="unsafe")
            N = np.add(N, w3 * Image[cy:cy + dy, fx:fx + dx], casting="unsafe")
            N = np.add(N, w4 * Image[cy:cy + dy, cx:cx + dx], casting="unsafe")
            D = (N >= C).astype(np.uint8)
        # Update the result matrix.
        v = 2 ** i
        result += D * v
        # cv2.imshow('image2', result)
        # cv2.waitKey(0)
    return result.astype(np.uint8)
