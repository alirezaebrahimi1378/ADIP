import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import rasterio as rio

def rmse(res):
    dif_vector =np.sqrt(res[: , 0] **2 + res[: , 1] **2)
    rmse = np.mean(dif_vector)
    return rmse

def match(train_e , train_t , test_e , test_t , method = 'quadric'):
    x = train_e
    y = train_t
    X = test_e
    Y = test_t
    y_label = np.hstack((y[: , 0] , y[: , 1]))
    Y_label = np.hstack((Y[: , 0] , Y[: , 1]))

    if method == 'affine' :
        # X = a0 + a1*x + a2*y
        B1 = np.column_stack((np.ones([len(x[: , 0]), 1]) , np.array([x[: , 0],x[: , 1]]).T , np.zeros([len(x), 3])))
        B2 = np.column_stack((np.zeros([len(x), 3]), np.ones([len(x[: , 0]), 1]) ,np.array([x[: , 0], x[: , 1]]).T))
        A = np.vstack((B1, B2))
        Params = np.linalg.inv(A.T @ A) @ (A.T @ y_label)

        xnew_train = Params[0] + Params[1] * x[:, 0] + Params[2] * x[:, 1]
        ynew_train = Params[3] + Params[4] * x[:, 0] + Params[5] * x[:, 1]

        xnew_test = Params[0] + Params[1] * X[: , 0] + Params[2] * X[: , 1]
        ynew_test = Params[3] + Params[4] * X[: , 0] + Params[5] * X[: , 1]

        res_image_train = np.column_stack((xnew_train, ynew_train)) - y
        res_image_test = np.column_stack((xnew_test, ynew_test)) - Y
        rmse_train = rmse(res_image_train)
        rmse_test = rmse(res_image_test)

    if method == 'quadratic' :
        # X = a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y
        B1 = np.column_stack((np.ones([len(x[: , 0]), 1]) , np.array([x[: , 0],x[: , 1],x[: , 0]**2 , x[: , 1]**2 , x[: , 0]*x[: , 1]]).T , np.zeros([len(x), 6])))
        B2 = np.column_stack((np.zeros([len(x), 6]), np.ones([len(x[: , 0]), 1]) ,np.array([x[: , 0], x[: , 1],x[: , 0]**2 , x[: , 1]**2 ,x[: , 0] * x[: , 1]]).T))
        A = np.vstack((B1, B2))
        Params = np.linalg.inv(A.T @ A) @ (A.T @ y_label)

        xnew_train = Params[0] + Params[1] * x[:, 0] + Params[2] * x[:, 1] + Params[3] * x[:, 0] ** 2 + Params[4] * x[:,1] ** 2 + Params[5] * x[:, 0] * x[:, 1]
        ynew_train = Params[6] + Params[7] * x[:, 0] + Params[8] * x[:, 1] + Params[9] * x[:, 0] ** 2 + Params[10] * x[:,1] ** 2 + Params[11] * x[:, 0] * x[:, 1]

        xnew_test = Params[0] + Params[1] * X[: , 0] + Params[2] * X[: , 1] + Params[3] * X[: , 0] ** 2 + Params[4] * X[: , 1]**2 + Params[5] * X[: , 0] * X[: , 1]
        ynew_test = Params[6] + Params[7] * X[: , 0] + Params[8] * X[: , 1] + Params[9] * X[: , 0] ** 2 + Params[10] * X[: , 1] ** 2 + Params[11] * X[:, 0] * X[:, 1]

        res_image_train = np.column_stack((xnew_train, ynew_train)) - y
        res_image_test = np.column_stack((xnew_test, ynew_test)) - Y
        rmse_train = rmse(res_image_train)
        rmse_test = rmse(res_image_test)

    if method == 'cubic' :
        # X = a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y + a6*x**2*y + a7*x*y**2 + a8*x**3 + a9*y**3
        B1 = np.column_stack((np.ones([len(x[:, 0]), 1]),np.array([x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1] , x[: , 0]**2 * x[: , 1] , x[: , 0] * x[: , 1]**2 ,x[: , 0]**3 , x[: , 1]**3  ]).T,np.zeros([len(x), 10])))
        B2 = np.column_stack((np.zeros([len(x), 10]), np.ones([len(x[:, 0]), 1]),np.array([x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1] , x[: , 0]**2 * x[: , 1] , x[: , 0] * x[: , 1]**2 ,x[: , 0]**3 , x[: , 1]**3]).T))
        A = np.vstack((B1, B2))
        Params = np.linalg.inv(A.T @ A) @ (A.T @ y_label)

        xnew_train = Params[0] + Params[1] * x[:, 0] + Params[2] * x[:, 1] + Params[3] * x[:, 0] ** 2 + Params[4] * x[:,1] ** 2 + Params[5] * x[:, 0] * x[:, 1] + Params[6] * x[: , 0]**2 * x[: , 1] + Params[7] * x[: , 0] * x[: , 1]**2 + Params[8] * x[: , 0]**3 + Params[9] * x[: , 1]**3
        ynew_train = Params[10] + Params[11] * x[:, 0] + Params[12] * x[:, 1] + Params[13] * x[:, 0] ** 2 + Params[14] * x[:,1] ** 2 + Params[15] * x[:, 0] * x[:, 1] + Params[16] * x[: , 0]**2 * x[: , 1] + Params[17] * x[: , 0] * x[: , 1]**2 + Params[18] * x[: , 0]**3 + Params[19] * x[: , 1]**3

        xnew_test = Params[0] + Params[1] * X[:, 0] + Params[2] * X[:, 1] + Params[3] * X[:, 0] ** 2 + Params[4] * X[:,1] ** 2 + Params[5] * X[:, 0] * X[:, 1] + Params[6] * X[: , 0]**2 * X[: , 1] + Params[7] * X[: , 0] * X[: , 1]**2 + Params[8] * X[: , 0]**3 + Params[9] * X[: , 1]**3
        ynew_test = Params[10] + Params[11] * X[:, 0] + Params[12] * X[:, 1] + Params[13] * X[:, 0] ** 2 + Params[14] * X[:,1] ** 2 + Params[15] * X[:, 0] * X[:, 1] + Params[16] * X[: , 0]**2 * X[: , 1] + Params[17] * X[: , 0] * X[: , 1]**2 + Params[18] * X[: , 0]**3 + Params[19] * X[: , 1]**3


        res_image_train = np.column_stack((xnew_train, ynew_train)) - y
        res_image_test = np.column_stack((xnew_test, ynew_test)) - Y
        rmse_train = rmse(res_image_train)
        rmse_test = rmse(res_image_test)

    return res_image_train, res_image_test, rmse_train, rmse_test

def coord_transform(coords , geopath) :
    img = rio.open(geopath)
    left = img.bounds[0]
    bottom = img.bounds[1]
    right = img.bounds[2]
    top = img.bounds[3]
    dx = (right - left) / img.shape[0]
    dy = (top - bottom) / img.shape[1]
    coords[: , 0] = left + coords[: , 0] * dx
    coords[: , 1] = top - coords[: , 1] * dy
    return(coords)