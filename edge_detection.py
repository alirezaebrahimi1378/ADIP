from itertools import count

import numpy as np

import cv2


def robert(img ,thre,use_thr):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    roberts_cross_v = np.array([[1, 0],[0, -1]])
    roberts_cross_h = np.array([[0, 1],[-1, 0]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, roberts_cross_v)
    img_prewitty = cv2.filter2D(img_gaussian, -1, roberts_cross_h)
    edge = np.sqrt(img_prewittx ** 2 + img_prewitty ** 2)
    if use_thr == 'on':
        edge = np.where(edge < thre , 0 ,255)
    return edge

def prewit(img , thre,use_thr):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    edge = np.sqrt(img_prewittx**2 + img_prewitty**2)
    if use_thr == 'on':
        edge = np.where(edge < thre , 0 ,255)
    return edge

def sobel(img , thre,use_thr):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape
    edge = np.zeros(shape)
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            edge[i][j] = int(np.sqrt((gray[i-1][j+1] + 2*gray[i][j+1] + gray[i+1][j+1] - gray[i-1][j-1] - 2*gray[i][j-1] - gray[i+1][j-1]) ** 2 + (gray[i+1][j-1] + 2*gray[i+1][j] + gray[i+1][j+1] -gray[i-1][j-1] - 2*gray[i-1][j] - gray[i-1][j+1]) ** 2))
    if use_thr == 'on':
        edge = np.where(edge < thre , 0 ,255)
    return edge

def canny(img , upper_ther , lower_ther):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray ,lower_ther, upper_ther)
    return edge