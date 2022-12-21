#!/usr/bin/env python3
# ? BRINT:Binary Rotation Invariant and Noise Tolerant
import matplotlib.pyplot as plt
import numba
import numpy as np
from src.utils.tool_descriptor import bit_rotate_right, bin_to_decimal, getPattern, generateTwoImgs
import math

from numba import jit


# from src.utils.tool_descriptor import _bit_rotate_right

@jit(nopython=True)
def differenceAbsolute(neighbour, center):
    return abs(neighbour - center)


@jit(nopython=True)
def differences_local(neighbours, center):
    arrayDifferece = []
    for neighbour in neighbours:
        arrayDifferece.append(differenceAbsolute(neighbour, center))

    return arrayDifferece


@jit(nopython=True)
def localAverage(array, radius, q):
    newNeigbours = []
    for i in range(0, 8):
        indexStart = i * radius
        indexFinish = indexStart + q

        localAverageGroup = 0
        for j in range(indexStart, indexFinish):
            localAverageGroup += array[j]

        localAverageGroup = localAverageGroup / q
        newNeigbours.append(localAverageGroup)

    return np.array(newNeigbours)


# ?El valor decimal mínimo de la nueva secuencia binaria se puede obtener girando continuamente la secuencia binaria
def get_min_for_revolve(arr):
    # ?Almacene el valor después de cada turno y finalmente seleccione el que tenga el valor más pequeño
    values = []
    # ?Se utiliza para desplazamiento cíclico, y su correspondiente sistema decimal se calcula respectivamente
    circle = arr * 2
    for i in range(0, 8):
        j = 0
        sum = 0
        bit_sum = 0
        while j < 8:
            sum += circle[i + j] << bit_sum
            bit_sum += 1
            j += 1
        values.append(sum)
    return min(values)


@jit(nopython=True)
def getUmbral(neighboursReduce):
    umbral = 0
    for neighbour in neighboursReduce:
        umbral += neighbour
    umbral = umbral / 8
    return umbral


def setNeighbours(neighboursReduce, umbral):
    for i in range(0, len(neighboursReduce)):
        neighboursReduce[i] = neighboursReduce[i] - umbral


@jit(nopython=True)
def ROR(lbp, points):
    rotation_chain = np.zeros(points, dtype=np.uint8)
    rotation_chain[0] = lbp
    for i in range(1, points):
        rotation_chain[i] = bit_rotate_right(rotation_chain[i - 1], points)
    lbp = rotation_chain[0]
    for i in range(1, points):
        lbp = min(lbp, rotation_chain[i])
    return lbp


N_POINT = 8
# ? No usamos histrogramas para la red neuronal
PLOT_IMG = False


def show_img_result(img, title=None):
    # fig, ax = plt.subplots()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    if title:
        plt.title(title)
    else:
        plt.title('A single plot')
    plt.show()


def show_imgs_titles(imgs, titles):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(imgs[0], cmap='gray')
    ax1.set_title(titles[0])
    ax1.axis('off')

    ax2.imshow(imgs[1], cmap='gray')
    ax2.set_title(titles[1])
    ax2.axis('off')

    ax3.imshow(imgs[2], cmap='gray')
    ax3.set_title(titles[2])
    ax3.axis('off')

    plt.show()


def show_all_img_Result(imgs, titles=None):
    f, axs = plt.subplots(1, 3)
    axs[0, 0].plot(imgs[0])
    axs[0, 1].plot(imgs[1])
    axs[0, 2].plot(imgs[2])
    for i, title in enumerate(titles):
        axs[0, i].set_title(title)

    plt.show()


@jit(nopython=True)
def run_gpu_brint_s(img, radius=1, q_points=2):
    n_points_complete = q_points * 8
    neighbours_complete = np.zeros(n_points_complete, dtype=np.uint8)
    lbp_value = np.zeros(N_POINT, dtype=np.uint8)
    n_row, n_col = img.shape
    img_filtered_brint_s = np.zeros(shape=(n_row, n_col))

    # *Interpolación lineal
    for x in range(radius, n_row - radius - 1):
        for y in range(radius, n_col - radius - 1):
            lbp = 0.0
            center = img[y, x]
            # ? Valores muy cercanos al negro no aplicamos el descriptor de textura, porque es fondo
            if center > 10:
                for n in range(n_points_complete):
                    theta = float(2 * np.pi * n) / n_points_complete
                    x_n = x + radius * np.cos(theta)
                    y_n = y - radius * np.sin(theta)

                    x1 = int(math.floor(x_n))
                    y1 = int(math.floor(y_n))

                    x2 = int(math.ceil(x_n))
                    y2 = int(math.ceil(y_n))

                    tx = np.abs(x - x1)
                    ty = np.abs(y - y1)

                    w1 = (1 - tx) * (1 - ty)
                    w2 = tx * (1 - ty)
                    w3 = (1 - tx) * ty
                    w4 = tx * ty

                    neighbour = img[y1, x1] * w1 + img[y2, x1] * w2 + img[y1, x2] * w3 + img[y2, x2] * w4
                    neighbours_complete[n] = neighbour

                # --Reducción de dimensionalidad
                neighbours_reduce = localAverage(neighbours_complete, radius, q_points)

                for n in range(N_POINT):
                    lbp_value[n] = getPattern(neighbours_reduce[n] - center)
                for n in range(N_POINT):
                    lbp += lbp_value[n] * (2 ** n)

                # --ROR
                lbp = ROR(lbp, N_POINT)
                img_filtered_brint_s[y, x] = int(lbp / (2 ** N_POINT - 1) * 255)

    return img_filtered_brint_s


@jit(nopython=True)
def run_gpu_brint_m(img, radius=1, q_points=2):
    n_points_complete = q_points * 8
    n_row, n_col = img.shape
    img_filtered_brint_s = np.zeros(shape=(n_row, n_col))
    neighbours_complete = np.zeros(n_points_complete, dtype=np.uint8)
    # lbp_value = np.zeros((1, n_points_complete), dtype=numba.np.uint8 )
    lbp_value = np.zeros(n_points_complete, dtype=np.uint8)
    for x in range(radius, n_row - radius - 1):
        for y in range(radius, n_col - radius - 1):
            lbp = 0.
            center = img[y, x]
            # ? Valores muy cercanos al negro no aplicamos el descriptor de textura, porque es fondo
            if center > 10:
                for n in range(n_points_complete):
                    theta = float(2 * np.pi * n) / n_points_complete
                    x_n = x + radius * np.cos(theta)
                    y_n = y - radius * np.sin(theta)

                    x1 = int(math.floor(x_n))
                    y1 = int(math.floor(y_n))

                    x2 = int(math.ceil(x_n))
                    y2 = int(math.ceil(y_n))

                    tx = np.abs(x - x1)
                    ty = np.abs(y - y1)

                    w1 = (1 - tx) * (1 - ty)
                    w2 = tx * (1 - ty)
                    w3 = (1 - tx) * ty
                    w4 = tx * ty

                    neighbour = img[y1, x1] * w1 + img[y2, x1] * w2 + img[y1, x2] * w3 + \
                                img[y2, x2] * w4

                    neighbours_complete[n] = neighbour

                center = img[y, x]
                # neighboursDifferenceLocals
                neighboursDifferenceLocals = differences_local(neighbours_complete, center)

                # --Reducción de dimensionalidad
                # z_r_q_i
                neighboursDL_Reduce = localAverage(neighboursDifferenceLocals, radius, q_points)
                # -----------------------------
                umbral = getUmbral(neighboursDL_Reduce)

                for n in range(N_POINT):
                    lbp_value[n] = getPattern(neighboursDL_Reduce[n] - umbral)

                for n in range(N_POINT):
                    lbp += lbp_value[n] * (2 ** n)

                # --ROR
                # !Si se aplica el ROR visualmente vemos que la mama lo osucrece demasiado mejor no ponerlo
                # lbp = ROR(lbp, N_POINT)
                img_filtered_brint_s[y, x] = int(lbp / (2 ** N_POINT - 1) * 255)
                # self.newMatrixBrintM[y, x] = lbp
    return img_filtered_brint_s


class Brint:
    def __init__(self, radius, q_points):
        self.radius = radius
        self.n_points_complete = q_points * N_POINT
        self.q_points = q_points

    def run_brint_s(self, img):
        # neighbours_complete = np.zeros(self.n_points_complete, dtype=numba.np.uint8 )
        # neighbours_complete = np.zeros(self.n_points_complete, dtype=np.uint8)
        neighbours_complete = np.zeros(2 * 8, dtype=np.uint8)
        lbp_value = np.zeros(N_POINT, dtype=np.uint8)
        n_row, n_col = img.shape
        img_filtered_brint_s = np.zeros(shape=(n_row, n_col))
        # conjunto = set()
        # *Interpolación lineal
        for x in range(self.radius, n_row - self.radius - 1):
            for y in range(self.radius, n_col - self.radius - 1):
                lbp = 0.0
                for n in range(self.n_points_complete):
                    theta = float(2 * np.pi * n) / self.n_points_complete
                    x_n = x + self.radius * np.cos(theta)
                    y_n = y - self.radius * np.sin(theta)

                    x1 = int(math.floor(x_n))
                    y1 = int(math.floor(y_n))

                    x2 = int(math.ceil(x_n))
                    y2 = int(math.ceil(y_n))

                    tx = np.abs(x - x1)
                    ty = np.abs(y - y1)

                    w1 = (1 - tx) * (1 - ty)
                    w2 = tx * (1 - ty)
                    w3 = (1 - tx) * ty
                    w4 = tx * ty

                    neighbour = img[y1, x1] * w1 + img[y2, x1] * w2 + img[y1, x2] * w3 + img[y2, x2] * w4
                    neighbours_complete[n] = neighbour

                center = img[y, x]

                # --Reducción de dimensionalidad
                # vecinos reducidos en 8
                neighbours_reduce = localAverage(neighbours_complete, self.radius, self.q_points)
                # -----------------------------
                for n in range(N_POINT):
                    lbp_value[n] = getPattern(neighbours_reduce[n] - center)
                for n in range(N_POINT):
                    lbp += lbp_value[n] * (2 ** n)

                # --ejecución de ROR
                # es necesario sino dara un resultado muy blanqueado
                lbp = ROR(lbp, N_POINT)

                img_filtered_brint_s[y, x] = int(lbp / (2 ** N_POINT - 1) * 255)
                # print("Valores -> ", y, x)
                # print(lbp)
                # conjunto.add(lbp)
                # img_filtered_brint_s[y, x] = lbp
        return img_filtered_brint_s

    def run_brint_m(self, img):
        # self.matrix.astype(dtype=np.float32)
        # self.newMatrixBrintM.astype(dtype=np.float32)
        n_row, n_col = img.shape
        img_filtered_brint_s = np.zeros(shape=(n_row, n_col))
        # neighbours = np.zeros((1, n_points_complete), dtype=numba.np.uint8 )
        neighbours_complete = np.zeros(self.n_points_complete, dtype=np.uint8)
        # lbp_value = np.zeros((1, n_points_complete), dtype=numba.np.uint8 )
        lbp_value = np.zeros(self.n_points_complete, dtype=np.uint8)
        for x in range(self.radius, n_row - self.radius - 1):
            for y in range(self.radius, n_col - self.radius - 1):
                lbp = 0.

                for n in range(self.n_points_complete):
                    theta = float(2 * np.pi * n) / self.n_points_complete
                    x_n = x + self.radius * np.cos(theta)
                    y_n = y - self.radius * np.sin(theta)

                    x1 = int(math.floor(x_n))
                    y1 = int(math.floor(y_n))

                    x2 = int(math.ceil(x_n))
                    y2 = int(math.ceil(y_n))

                    tx = np.abs(x - x1)
                    ty = np.abs(y - y1)

                    w1 = (1 - tx) * (1 - ty)
                    w2 = tx * (1 - ty)
                    w3 = (1 - tx) * ty
                    w4 = tx * ty

                    neighbour = img[y1, x1] * w1 + img[y2, x1] * w2 + img[y1, x2] * w3 + \
                                img[y2, x2] * w4

                    neighbours_complete[n] = neighbour

                center = img[y, x]
                # neighboursDifferenceLocals
                neighboursDifferenceLocals = differences_local(neighbours_complete, center)

                # --Reducción de dimensionalidad
                # z_r_q_i
                neighboursDL_Reduce = localAverage(neighboursDifferenceLocals, self.radius, self.q_points)
                # -----------------------------
                umbral = getUmbral(neighboursDL_Reduce)

                for n in range(N_POINT):
                    lbp_value[n] = getPattern(neighboursDL_Reduce[n] - umbral)

                for n in range(N_POINT):
                    lbp += lbp_value[n] * (2 ** n)

                # --ROR
                # lbp = ROR(lbp, points)
                # --ROR
                # self.newMatrixBrintM[y, x] = lbp
                img_filtered_brint_s[y, x] = int(lbp / (2 ** N_POINT - 1) * 255)

    # !Fallido, Mejorar el subdescriptor s apra que no haga interpolación
    def try_Brint_s(self, radius):
        q = radius
        p = 8 * radius

        # !Si es de radio 1 entonces es usar el lbp ya construido
        # !No funcaionaria con radio de 1 pore eso es un caso especial
        for i in range(0, self.height):
            for j in range(0, self.width):
                neighbors = []

                indexI = i
                indexJ = j

                try:
                    neighbors.append(self.matrix[i][j + radius])

                    # -- desde el primer vecino hacia arriba
                    for ascenso_i in range(1, radius + 1):
                        neighbors.append(self.matrix[i - ascenso_i][j + radius])
                    indexI = i - radius
                    indexJ = j + radius

                    # -- desde la esquina arriba derecha a esquina izquierda
                    displace_left_complete_j = 2 * radius
                    for displace_j in range(1, displace_left_complete_j + 1):
                        neighbors.append(self.matrix[indexI][indexJ - displace_j])
                        indexJ = indexJ - displace_left_complete_j

                    # -- desde esquina izquierda a esquina abajo izquierda
                    displace_left_complete_i = displace_left_complete_j
                    for displace_i in range(1, displace_left_complete_i + 1):
                        neighbors.append(self.matrix[indexI + displace_i][indexJ])

                    indexI = indexI + displace_left_complete_i
                    # -- desde esquina abajo izquierda a esquina derecha abajo
                    displace_left_complete_j = 2 * radius
                    for displace_j in range(1, displace_left_complete_j + 1):
                        neighbors.append(self.matrix[indexI][indexJ + displace_j])
                    indexJ = indexJ + displace_left_complete_j

                    # -- desde esquina abajo derecha a antes de la mitad de punto medio
                    # !Revisar
                    displace_left_complete_i = displace_left_complete_i // 2
                    for displace_i in range(1, displace_left_complete_i):
                        neighbors.append(self.matrix[indexI - displace_i][indexJ])

                    # ?Reducción de dimensionalidad
                    neighbors = localAverage(neighbors, radius, q)
                    print(neighbors)
                    # pattern = calculateLBP(neighbors, self.matrix[i][j])

                    # ? la nueva matrix
                    # self.newMatrix[i, j] = bin_to_decimal(pattern)

                except IndexError:
                    continue
            # revolve_key = get_min_for_revolve(sum)
