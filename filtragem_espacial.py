# Aluno: Joathan Mareto Fontan
# RGA: 201811722008

# Os tempos são exibidos no terminal e no título das imagens

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Função que replica a borda nos quatro cantos e aplica margem
def borda_replicada(img_br, img_aux, k_dim, h, w, nh, nw):
    img_aux[int((k_dim - 1) / 2):nh - int((k_dim - 1) / 2),
            0:int((k_dim - 1) / 2)] = img_br[:, 0:int((k_dim - 1) / 2)]

    img_aux[int((k_dim - 1) / 2):nh - int((k_dim - 1) / 2),
            nw - int((k_dim - 1) / 2):nw] = img_br[:,
                                                   w - int((k_dim - 1) / 2):w]

    img_aux[0:int((k_dim - 1) / 2),
            int((k_dim - 1) / 2):nw -
            int((k_dim - 1) / 2)] = img_br[0:int((k_dim - 1) / 2), :]

    img_aux[nh - int((k_dim - 1) / 2):nh,
            int((k_dim - 1) / 2):nw -
            int((k_dim - 1) / 2)] = img_br[h - int((k_dim - 1) / 2):h, :]

    # Centraliza a imagem
    img_aux[int((k_dim - 1) / 2):h + int((k_dim - 1) / 2),
            int((k_dim - 1) / 2):w + int((k_dim - 1) / 2)] = img_br

    return img_aux


def filter2d(img, kernel):
    hk, wk = kernel.shape

    if hk == wk:
        h, w, c = img.shape

        k_dim = wk

        nh = h + (k_dim - 1)
        nw = w + (k_dim - 1)

        # Imagem auxiliar que contém as bordas
        img_aux = np.zeros((nh, nw, c), dtype=np.int32)
        img_aux = borda_replicada(img, img_aux, k_dim, h, w, nh, nw)

        img_out = np.zeros((h, w, c), dtype=np.int32)

        for cnl in range(3):
            for y in range(nh - (k_dim - 1)):
                for x in range(nw - (k_dim - 1)):
                    k_op = np.sum(img_aux[y:y + hk, x:x + wk, cnl] * kernel)
                    img_out[y, x, cnl] = k_op

        return img_out


def sep_filter2d(img, k1, k2):
    hk = len(k1)
    wk = len(k2)

    if hk == wk:
        h, w, c = img.shape

        k_dim = wk

        nh = h + (k_dim - 1)
        nw = w + (k_dim - 1)

        # Imagem auxiliar que contém as bordas
        img_aux = np.zeros((nh, nw, c), dtype=np.int32)
        img_aux = borda_replicada(img, img_aux, k_dim, h, w, nh, nw)

        # Imagens intermediárias entre a aplicação de cada kernel
        img_interm = np.zeros((nh, nw, c), dtype=np.int32)
        img_interm_2 = np.zeros((nh, nw, c), dtype=np.int32)

        # Imagem de saída sem as bordas extras e com as mesmas dimensões da original
        img_out = np.zeros((h, w, c), dtype=np.int32)

        for cnl in range(3):
            for y in range(nh):
                for x in range(nw):
                    if y < nh - (k_dim - 1):
                        # Cima pra baixo
                        k1_op = np.sum(img_aux[y:y + hk, x, cnl] * k1)
                        img_interm[y + int((k_dim - 1) / 2), x, cnl] = k1_op

                    if x < nw - (k_dim - 1):
                        # Esquerda pra direita
                        k2_op = np.sum(img_interm[y, x:x + wk, cnl] * k2)
                        img_interm_2[y, x + int((k_dim - 1) / 2), cnl] = k2_op

        # Retira as bordas da imagem intermediária
        img_out = img_interm_2[int((k_dim - 1) / 2):h + int((k_dim - 1) / 2),
                               int((k_dim - 1) / 2):w + int((k_dim - 1) / 2)]

        return img_out


if __name__ == '__main__':
    img = cv2.imread('flor.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tamanho do Kernel para 1D e 2D
    kernel_size = 201

    # Kernel 2D
    kernel_2d = np.ones((kernel_size, kernel_size))
    kernel_2d = kernel_2d / (kernel_size * kernel_size)

    # Kernel 1D
    kernel_1d = np.ones(kernel_size)
    kernel_1d = kernel_1d / kernel_size

    _, axs = plt.subplots(1, 3)

    axs[0].set_title('original')
    axs[0].imshow(img)

    # ############################################# #

    # Implementação do filter2d na mão
    inicio = time.time()
    img_out = filter2d(img, kernel_2d)
    fim = time.time()

    msg = 'filter2d: {:.3f} s'.format(fim - inicio)
    print('\n' + msg)

    axs[1].set_title('tamanho do kernel: ' + str(kernel_size) + '\n\n\n' + msg)
    axs[1].imshow(img_out)

    # ############################################# #

    # Implementação do sep_filter2d na mão
    inicio = time.time()
    img_out_sep = sep_filter2d(img, kernel_1d, kernel_1d)
    fim = time.time()

    msg = 'sep_filter2d: {:.3f} s'.format(fim - inicio)
    print(msg + '\n')

    axs[2].set_title(msg)
    axs[2].imshow(img_out_sep)

    # ############################################# #

    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Joathan Mareto Fontan')

    plt.show()
