import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = '/home/alireza/Desktop/seg/rectangle3.tif'
def UINT8(Data) :
    shape = Data.shape
    for i in range(shape[0]):
        data = Data[i , : , :]
        data = data / data.max()
        data = 255 * data
        Data[i] = data.astype(np.uint8)
    return Data
def apply_kernel(img , kernel_size):

    img = UINT8(img)
    img = img[14:, : , :]
    img= img.transpose(1 , 2 , 0)
    kernel = (kernel_size , kernel_size)
    blur = cv2.blur(img , kernel)
    g_blur = cv2.GaussianBlur(img,kernel,0)
    sobel = cv2.Sobel(img , ddepth = 2,ksize = kernel_size , dx = 1 , dy = 0)
    return blur , g_blur , sobel
