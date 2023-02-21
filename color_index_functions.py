import numpy as np

def UINT8(Data) :
    shape = Data.shape
    for i in range(shape[0]):
        data = Data[i , : , :]
        data = data / data.max()
        data = 255 * data
        data = data.astype(np.uint8)
        Data[i] = data
    return Data

def Uint8(img):
    img = img/img.max()
    img = img * 255
    img = img.astype(np.uint8)
    return img

def NDVI(img):
    b8 = img[7 , : , :]
    b4 = img[3 , : , :]
    ndvi = (b8 - b4)/(b8 + b4)
    return ndvi

def MSAVI(img):
    b8 = img[7 , : , :]
    b4 = img[3 , : , :]
    msavi = (2*b8+1-np.sqrt(np.square(2*b8+1)-8*(b8-b4)))/2
    return msavi

def IPVI(img):
    b8 = img[7 , : , :]
    b4 = img[3 , : , :]
    ipvi = (b8*(NDVI(img)+1))/2*(b8+b4)
    return ipvi

def fc(r,g,b):
    rgb = np.dstack((r , g , b))
    return rgb