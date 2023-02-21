import cv2
import numpy as np
import os

def features(img,direction):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / gray.max()
    gray = gray * 5
    gray = gray.astype(np.uint8)
    shape = gray.shape
    co = np.zeros((6 , 6))
    if direction == '0':
        for i in range(1 , shape[0] - 1):
            for j in range(1 , shape[1] - 1):
                a = gray[i , j]
                b = gray[i , j + 1]
                co[a , b] += 1
    elif direction == '45':
        for i in range(1 , shape[0] - 1):
            for j in range(1 , shape[1] - 1):
                a = gray[i , j]
                b = gray[i + 1 , j + 1]
                co[a , b] += 1
    elif direction == '90':
        for i in range(1 , shape[0] - 1):
            for j in range(1 , shape[1] - 1):
                a = gray[i , j]
                b = gray[i + 1 , j + 1]
                co[a , b] += 1
    pro = co / np.sum(co)
    Energy = 0
    Entropy = 0
    Max_pro = pro.max()
    Contrast = 0
    Inv_diff_mom = 0
    Corr = 0
    sigmax = 0
    sigmay = 0
    mx = 0
    my = 0
    sumx = 0
    sumy = 0
    for i in range(0 , 6):
        for j in range(0 , 6):
            sumx = sumx + pro[i , j]
        mx = mx + i * sumx

    for j in range(0 , 6):
        for i in range(0 , 6):
            sumy = sumy + pro[i , j]
        my = my + j * sumy

    sumx = 0
    sumy = 0
    for i in range(0 , 6):
        for j in range(0 , 6):
            sumx = sumx + pro[i , j]
        sigmax = sigmax + (((i-mx)**2)*sumx)

    for j in range(0 , 6):
        for i in range(0 , 6):
            sumy = sumy + pro[i , j]
        sigmay = sigmay + (((j-my)**2)*sumy)

    for i in range(0 , 6):
        for j in range(0 , 6):
            Energy = Energy + pro[i , j]**2
            Entropy = Entropy + (pro[i , j] *np.log2(pro[i,j]))
            Contrast = Contrast + ((abs(i - j)**2) * pro[i , j])
            Corr = Corr + ((i*j*pro[i , j]) - mx*my)
            if i != j :
                Inv_diff_mom = Inv_diff_mom + (pro[i,j]/(abs(i - j)**2))

    Corr = Corr/(sigmax * sigmay)

    return Energy , Entropy , Max_pro , Contrast , Inv_diff_mom , Corr

def NineD(img , dim):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    l5 = np.array([[1 , 4 , 6 , 4 , 1]])
    r5 = np.array([[1 , -4 , 6 , -4 , 1]])
    e5 = np.array([[-1 , -2 , 0 , 2 , 1]])
    s5 = np.array([[-1 , 0 , 2 , 0 , -1]])
    w5 = np.array([[-1 , 2 , 0 , -2 , 1]])

    l5e5 = np.multiply(np.transpose(l5) ,e5)
    e5l5 = np.multiply(np.transpose(e5) ,l5)
    l5r5 = np.multiply(np.transpose(l5) ,r5)
    r5l5 = np.multiply(np.transpose(r5) ,l5)
    e5s5 = np.multiply(np.transpose(e5) ,s5)
    s5e5 = np.multiply(np.transpose(s5) ,e5)
    s5s5 = np.multiply(np.transpose(s5) ,s5)
    r5r5 = np.multiply(np.transpose(r5), r5)
    l5s5 = np.multiply(np.transpose(l5), s5)
    s5l5 = np.multiply(np.transpose(s5), l5)
    e5e5 = np.multiply(np.transpose(e5), e5)
    e5r5 = np.multiply(np.transpose(e5), r5)
    r5e5 = np.multiply(np.transpose(r5), e5)
    s5r5 = np.multiply(np.transpose(s5), r5)
    r5s5 = np.multiply(np.transpose(r5), s5)

    L5E5 = cv2.filter2D(src=img, ddepth=-1, kernel=l5e5)
    E5L5 = cv2.filter2D(src=img, ddepth=-1, kernel=e5l5)
    L5R5 = cv2.filter2D(src=img, ddepth=-1, kernel=l5r5)
    R5L5 = cv2.filter2D(src=img, ddepth=-1, kernel=r5l5)
    E5S5 = cv2.filter2D(src=img, ddepth=-1, kernel=e5s5)
    S5E5 = cv2.filter2D(src=img, ddepth=-1, kernel=s5e5)
    S5S5 = cv2.filter2D(src=img, ddepth=-1, kernel=s5s5)
    R5R5 = cv2.filter2D(src=img, ddepth=-1, kernel=r5r5)
    L5S5 = cv2.filter2D(src=img, ddepth=-1, kernel=l5s5)
    S5L5 = cv2.filter2D(src=img, ddepth=-1, kernel=s5l5)
    E5E5 = cv2.filter2D(src=img, ddepth=-1, kernel=e5e5)
    E5R5 = cv2.filter2D(src=img, ddepth=-1, kernel=e5r5)
    R5E5 = cv2.filter2D(src=img, ddepth=-1, kernel=r5e5)
    S5R5 = cv2.filter2D(src=img, ddepth=-1, kernel=s5r5)
    R5S5 = cv2.filter2D(src=img, ddepth=-1, kernel=r5s5)

    dict = {'L5E5/E5L5':(L5E5 + E5L5)/2 , 'L5R5/R5L5':(L5R5 + R5L5)/2 , 'E5S5/S5E5':(E5S5+S5E5)/2 , 'S5S5':S5S5 , 'R5R5':R5R5 , 'L5S5/S5L5':(L5S5+S5L5)/2 , 'E5E5':E5E5 , 'E5R5/R5E5':(E5R5+R5E5)/2 , 'S5R5/R5S5' : (S5R5 + R5S5)/2}

    return dict[dim]


root = os.getcwd()
path8 = root + '/DATA/rock.jpeg'
img = cv2.imread(path8)
NineD(img , 'L5E5/E5L5')