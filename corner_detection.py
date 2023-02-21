import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def harris(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2,21, 0.1)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img


def moravec(img):
    thresh = 100
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray.copy() , cv2.COLOR_GRAY2RGB)
    shifts = [(1 , 0) , (1 , 1) , (0 , 1) , (-1 , 1)]
    for y in range(1 , rgb.shape[1] - 1):
        for x in range(1 , rgb.shape[0] - 1):
            e = 100000
            for shift in shifts:
                diff = gray[x + shift[0] , y + shift[1]]
                diff = diff - gray[x , y]
                diff = diff * diff
                if diff < e :
                    e = diff
            if e > thresh :
                rgb[x , y] = (0 , 0 ,255)
    return(rgb)


def Sift(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return(img)


def shi_tomasi(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    return corners

