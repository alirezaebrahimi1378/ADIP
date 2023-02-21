import numpy as np
import matplotlib.pyplot as plt

def calc_2dft(input):
    ft = np.fft.fft2(input)
    return np.fft.fftshift(ft)

def UINT8(Data) :
    shape = Data.shape
    for i in range(shape[0]):
        data = Data[i , : , :]
        data = data / data.max()
        data = 255 * data
        Data[i] = data.astype(np.uint8)
    return Data

def fourier(img,d0, filter):
    img = img[4, :, :]
    m , n = img.shape
    plt.set_cmap("gray")
    ft = calc_2dft(img)
    FT = np.log(abs(ft))
    # defining ideal low_pass filter
    L = np.zeros((m,n) , dtype = np.float32)
    for u in range(m):
        for v in range(n):
            d = np.sqrt((u - m/2)**2 + (v - n/2)**2)
            if d <= d0 :
                L[u,v] = 1
            else:
                L[u,v] = 0
    H = 1 - L
    if filter == 1:
        gaussian_filter = ft * L
    else:
        gaussian_filter = ft * H

    #inversing low_pass filtered frequency image to spatial image
    inv = np.fft.ifftshift(gaussian_filter)

    ginv = np.abs(np.fft.ifft2(inv))

    return FT , np.log(abs(gaussian_filter)) , ginv
