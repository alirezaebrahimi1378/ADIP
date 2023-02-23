import tkinter
import random
from customtkinter import *
import customtkinter as ctk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import imutils
import numpy as np
import rasterio as rio

from matching_functions import match , coord_transform
from color_index_functions import UINT8 , NDVI , MSAVI ,Uint8 , IPVI , fc
from fourier_functions import fourier
from kernel import apply_kernel
from corner_detection import harris, moravec , Sift , shi_tomasi
from edge_detection import robert , prewit , sobel , canny
from texture import features , NineD
from segmentation import ther_based , watershed , k_means , mean_shift , fcm




ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('dark-blue')





class Window(CTk):
    def __init__(self):
        super().__init__()
        self.geometry('2000x1000')
        self.root = os.getcwd()
        self.path1 = self.root + '/DATA/download.jpeg'
        self.path2 = self.root + '/DATA/photo_1.jpg'
        self.path3 = self.root + '/DATA/photo_2.jpg'
        self.path4 = self.root + '/DATA/QuickBird_1.tif'
        self.path5 = self.root + '/DATA/rectangle6.tif'
        self.path6 = self.root + '/DATA/rectangle3.tif'
        self.path7 = self.root + '/DATA/image.jpg'
        self.path8 = self.root + '/DATA/rock.jpeg'
        self.path9 = self.root + '/DATA/flower.jpg'
        self.path10 = self.root + '/DATA/nature.jpg'
        self.show_plots()

        self.frame1 = frame1(master= self)
        self.frame1.grid(row=0 , column = 4 , columnspan = 2 , pady =5)
        self.frame1.configure(border_width=3,border_color='#639DC1',corner_radius=10 , fg_color='#343840')

        self.frame2 = frame2(master=self)
        self.frame2.grid(row=1, column=4, columnspan = 2 , pady = 10)
        self.frame2.configure(border_width=3 , border_color='#639DC1' ,corner_radius=10 ,fg_color='#343840')

        self.frame3 = frame3(master=self)
        self.frame3.grid(row=2, column=4 ,pady = 19)
        self.frame3.configure(border_width=3, border_color='#639DC1',corner_radius=10 ,fg_color='#343840')

        self.frame4 = frame4(master=self)
        self.frame4.grid(row=3, column=4 ,pady = 19)
        self.frame4.configure(border_width=3, border_color='#639DC1',corner_radius=10 ,fg_color='#343840')

        self.frame5 = frame5(master=self , width=470 , height = 400)
        self.frame5.grid(row=2, column=2, pady=20 , rowspan = 2,padx = 30 , sticky = 'nw')
        self.frame5.configure(border_width=3, border_color='#639DC1',corner_radius=10 ,fg_color='#343840')

        self.text_box = CTkTextbox(self ,border_color='#639DC1',scrollbar_button_color='#639DC1',fg_color='#343840',corner_radius=10,border_width=3 , width = 600 , height=420)
        self.text_box.grid(row = 2 , column=0 , columnspan = 2 , rowspan = 2  ,sticky = 'n', pady = 20 , padx= 50)

    def show_plots(self):
        figure = plt.figure(figsize=(11, 5), dpi=100)
        # figure.patch.set_facecolor('#1a1a1a')
        img = cv2.imread(self.path1)
        figure.add_subplot(121)
        plt.imshow(img)
        plt.axis('off')
        figure.add_subplot(122)
        plt.imshow(img)
        plt.axis('off')
        chart = FigureCanvasTkAgg(figure, self)
        chart.get_tk_widget().grid(row=0, column=0 , rowspan=2 , columnspan=4 , pady = 10 , padx = 100)
        return None


########################################## defining frames ##########################################

class frame1(CTkFrame):
    def __init__(self , master):
        super().__init__(master)
        self.root = master
        self.frame_name = CTkLabel(self, text='transforms')
        self.frame_name.grid(row=0, column=0, columnspan=4, pady=10)
        ############################## rotation ##############################
        #functions
        def bc1(self):
            value = self.sld1.get()
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#1a1a1a')
            img = cv2.imread(self.root.path1)
            res = imutils.rotate(img, value)
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(res)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            return None

        def sl1(value):
            self.lbl2.configure(text=str(int(self.sld1.get())))

        #widgets
        self.sld1 = CTkSlider(self,width = 200 ,from_ = 0 , to = 90,number_of_steps=90,command =sl1)
        self.sld1.grid(row=1 , column=1 , pady = 10)
        self.sld1.set(0)
        self.lbl1 = CTkLabel(self , text = 'rotation angle')
        self.lbl1.grid(row=1 , column=0 , padx = 10)
        self.lbl2 = CTkLabel(self , text = str(int(0)))
        self.lbl2.grid(row = 1 , column = 2 , sticky='w')
        self.btm1 = CTkButton(self ,width=70 ,text='rotate' , command=lambda : bc1(self))
        self.btm1.grid(row = 1 , column = 3,sticky='e' , pady = 10, padx=10)

        ############################## resize ##############################
        #functions
        def sl2(value):
            self.lbl4.configure(text=str(int(self.sld2.get())))

        def bc2(self):
            value = self.sld2.get()
            method = self.om1.get()
            if method =='nearest':
                Method = cv2.INTER_NEAREST
            elif method == 'linear':
                Method = cv2.INTER_LINEAR
            elif method == 'cubic':
                Method = cv2.INTER_CUBIC
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            img = cv2.imread(self.root.path1)
            (rows, cols) = img.shape[:2]
            res = cv2.resize(img, (int(cols / value), int(rows / value)), interpolation=Method)
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(res)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            return None

        #widgets
        self.lbl3 = CTkLabel(self , text='resampling scale')
        self.lbl3.grid(row=3 , column=0 , padx = 10)
        self.sld2 = CTkSlider(self , width = 200 , from_=1 , to=20 , number_of_steps=10,command=sl2)
        self.sld2.grid(row = 3 , column=1 ,pady = 10)
        self.sld2.set(0)
        self.lbl4 = CTkLabel(self,text = str(int(self.sld2.get())))
        self.lbl4.grid(row = 3 , column=2 ,sticky='w' )
        self.lbl5 = CTkLabel(self,text= 'resampling method')
        self.lbl5.grid(row=4 , column = 0 , padx = 10)
        self.om1_value = StringVar(value = 'nearest')
        self.om1 = CTkOptionMenu(self, values=['nearest' , 'linear' ,'cubic'] , variable=self.om1_value)
        self.om1.grid(row=4 , column = 1,pady = 10)
        self.btm2 = CTkButton(self,text='rescale' , command=lambda:bc2(self),width=70)
        self.btm2.grid(row=3 , column = 2,columnspan = 2,rowspan=2,sticky='e' ,padx = 10)

        ############################## shift ##############################
        #functions
        def sl3(value):
            self.lbl8.configure(text=str(int(self.sld3.get())))

        def sl4(value):
            self.lbl9.configure(text=str(int(self.sld4.get())))

        def bc3(self):
            x_shift = self.sld3.get()
            y_shift = self.sld4.get()
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            img = cv2.imread(self.root.path1)
            (rows, cols) = img.shape[:2]
            M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            res = cv2.warpAffine(img, M, (cols, rows))
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(res)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            return None

        #widgets
        self.lbl6 = CTkLabel(self , text='horizental shift')
        self.lbl6.grid(row = 5 , column=0 , padx = 10)
        self.lbl7 = CTkLabel(self, text='vertical shift')
        self.lbl7.grid(row=6, column=0, padx=10)
        self.sld3 = CTkSlider(self , width = 200 , from_=-100, to=100, number_of_steps=200 , command=sl3)
        self.sld3.grid(row=5 , column = 1)
        self.lbl8 = CTkLabel(self ,text = str(int(self.sld3.get())))
        self.lbl8.grid(row =5 , column=2 , sticky='w')
        self.sld4 = CTkSlider(self, width=200, from_=-100, to=100, number_of_steps=200, command=sl4)
        self.sld4.grid(row=6, column=1 )
        self.lbl9 = CTkLabel(self, text=str(int(self.sld3.get())))
        self.lbl9.grid(row=6, column=2 , sticky='w')
        self.btm3 = CTkButton(self,width = 70, text='shift', command=lambda: bc3(self))
        self.btm3.grid(row=5, column=3 , rowspan = 2 , padx = 10,pady = 10 , sticky='e')

        ############################## shear ##############################
        # functions
        def sl5(value):
            self.lbl11.configure(text=f'{self.sld5.get():.2f}')

        def bc4(self):
            value = self.sld5.get()
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            img = cv2.imread(self.root.path1)
            (rows, cols) = img.shape[:2]
            M = np.float32([[1, value, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
            res = cv2.warpPerspective(img, M, (int(cols *(1+value)), int(rows *(1+value))))
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(res)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            return None
        # widgets
        self.lbl10 = CTkLabel(self , text='shear parameter').grid(row=7 , column=0)
        self.sld5 = CTkSlider(self , from_ = 0 , to=1 , number_of_steps=100 , command = sl5)
        self.sld5.grid(row=7 , column=1 , pady = 10)
        self.lbl11 = CTkLabel(self, text=f'{self.sld5.get():.2f}')
        self.lbl11.grid(row= 7 , column=2 , sticky = 'w')
        self.btm4 = CTkButton(self , width=70,text='shear' , command= lambda:bc4(self))
        self.btm4.grid(row= 7 , column=3 ,pady = 10,padx = 10 ,sticky='e')

class frame2(CTkFrame):
    def __init__(self , master):
        super().__init__(master)
        self.root = master
        self.frame_name = CTkLabel(self , text='image matching')
        self.frame_name.grid(row=0 , column = 0 , columnspan = 2 , pady = 10 , padx = 178)
        #functions
        def bc1(self):
            self.root.text_box.configure(state='normal')
            self.root.text_box.delete('0.0', 'end')
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            img1 = cv2.imread(self.root.path2)
            img2 = cv2.imread(self.root.path3)
            figure.add_subplot(121)
            plt.imshow(img1)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(img2)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            self.coordsx = []
            self.coordsy = []
            def onclick(event , master = self):
                X_coordinate = int(event.xdata)
                Y_coordinate = int(event.ydata)
                self.coordsx.append(X_coordinate)
                self.coordsy.append(Y_coordinate)
                counter = len(self.coordsx)
                self.coords = np.vstack((np.array(self.coordsx) ,np.array(self.coordsy)))
                if counter % 2 == 0:
                    self.root.text_box.insert(index = 'end' ,text=f'coordinates of point in image one : [{self.coordsx[-2]},{self.coordsy[-2]}] / coordinates of point in image two : [{self.coordsx[-1]},{self.coordsy[-1]}]\n')
                else:
                    point_number = int(counter / 2)
                    self.root.text_box.insert(index='end' , text = f'**************  point number {point_number}  **************\n')
                    self.root.text_box.insert(index='end',text='point of image one have been selected\n')
            cid = figure.canvas.mpl_connect('button_press_event', onclick)

        def bc2(self):
            method = self.radiovar.get()
            if method == 1 :
                coords1 = np.empty((round(self.coords.shape[1] / 2), 2))
                coords2 = np.empty((round(self.coords.shape[1] / 2), 2))
                m = 0
                n = 0

                for i in range(self.coords.shape[1]):
                    if i % 2 == 0:
                        coords1[m, :] = self.coords[:, i]
                        m = m + 1
                    else:
                        coords2[n, :] = self.coords[:, i]
                        n = n + 1

                coords1 = np.array(coords1)
                coords2 = np.array(coords2)

                # np.savetxt('file1.txt', coords1)
                # np.savetxt('file2.txt', coords2)

            elif method == 2:
                coords1 = np.loadtxt('file1.txt')
                coords2 = np.loadtxt('file2.txt')


            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            img1 = cv2.imread(self.root.path2)
            img2 = cv2.imread(self.root.path3)
            figure.add_subplot(121)
            plt.imshow(img1)
            plt.axis('off')
            plt.plot(coords1[:, 0], coords1[:, 1], 'r*')
            figure.add_subplot(122)
            plt.imshow(img2)
            plt.axis('off')
            plt.plot(coords2[:, 0], coords2[:, 1], 'b*')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

            #creating train and test dataset
            rate = 0.2
            random.seed(1234)
            k = round(rate * coords1.shape[0])
            test_ind = random.sample(range(0, coords1.shape[0]), k)

            # transforming coordinates from pixel to earth

            coords2 = coord_transform(coords2, self.root.path4)
            # creating test and train dataset     e = entry / t = target
            test_e = coords1[test_ind, :]
            test_t = coords2[test_ind, :]
            train_e = np.delete(coords1, test_ind, axis=0)
            train_t = np.delete(coords2, test_ind, axis=0)
            func = self.om1.get()

            res_train ,res_test , rmse_train , rmse_test = match(train_e, train_t, test_e, test_t, method=func)
            self.root.text_box.configure(state = 'normal')
            self.root.text_box.delete('0.0','end')
            self.root.text_box.insert('end','********************** train_res **********************\n')
            self.root.text_box.insert('end',f'{res_train}\n')
            self.root.text_box.insert('end', '********************** test_res **********************\n')
            self.root.text_box.insert('end', f'{res_test}\n')
            self.root.text_box.insert('end', '********************** train_rmse **********************\n')
            self.root.text_box.insert('end', f'{rmse_train}\n')
            self.root.text_box.insert('end', '********************** test_rmse **********************\n')
            self.root.text_box.insert('end', f'{rmse_test}\n')
            self.root.text_box.configure(state='disable')
        #widgets
        self.btm1 = CTkButton(self , text='start matching',command = lambda :bc1(self))
        self.btm1.grid(row = 1 ,column= 1 , rowspan = 2 , padx = 20)
        self.btm2 = CTkButton(self , text = 'calculate' , command = lambda: bc2(self))
        self.btm2.grid(row = 3 ,column= 1 , padx = 20)
        self.radiovar = tkinter.IntVar(master = self ,value = 0)
        self.rb1 = CTkRadioButton(master = self , text='determine new points',variable=self.radiovar , value=1 )
        self.rb1.grid(row = 1 , column = 0 , padx = 10 ,pady = 4, sticky = 'w')
        self.om1_value = StringVar(value='affine')
        self.om1 = CTkOptionMenu(self, values=['affine', 'quadratic', 'cubic'], variable=self.om1_value)
        self.om1.grid(row = 3 , column = 0 , pady = 10)
        self.rb2 = CTkRadioButton(master = self, text='use predetermined points',variable=self.radiovar, value=2)
        self.rb2.grid(row = 2 , column= 0 , padx = 10 , sticky = 'w')

class frame3(CTkFrame):
    def __init__(self , master):
        super().__init__(master)
        self.root = master
        self.frame_name = CTkLabel(self , text='true color - false color')
        self.frame_name.grid(row=0 , column = 0 , columnspan = 6,padx = 161 ,pady = 10)

        #functions
        def bc1(self):
            image = rio.open(self.root.path5)
            img = image.read()
            img = UINT8(img)
            img1 = img[1:4 , : , :]
            img_vis1 = img1.transpose(1, 2, 0)
            ndvi_img = NDVI(img)
            msavi_img = MSAVI(img)
            ipvi_img = IPVI(img)
            dict = {'b1' :img[0] ,'b2' :img[1],'b3' :img[2],'b4' :img[3],'b5' :img[4],'b6' :img[5],'b7' :img[6],'b8' :img[7],'b9' :img[8],'b10' :img[9],'b11' :img[10],'b12' :img[11] ,'ndvi' :ndvi_img,'msavi' :msavi_img,'ipvi' :ipvi_img}
            R = self.om1.get()
            G = self.om2.get()
            B = self.om3.get()
            r = dict[R]
            g = dict[G]
            b = dict[B]
            false = fc(Uint8(r) , Uint8(g) ,Uint8(b))
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            figure.add_subplot(121)
            plt.imshow(img_vis1.astype(np.uint8))
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(false)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

        #widgets
        self.om1_value = StringVar(value='b1')
        self.om2_value = StringVar(value='b2')
        self.om3_value = StringVar(value='b3')
        self.lbl1 = CTkLabel(self,text='band1 : ')
        self.lbl1.grid(row=1, column=0 , padx = 5)
        self.om1 = CTkOptionMenu(self, width=10,values=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'ndvi', 'msavi', 'ipvi'], variable=self.om1_value)
        self.om1.grid(row=1, column=1,padx=10, pady=10)
        self.lbl2 = CTkLabel(self, text='band2 : ')
        self.lbl2.grid(row=1, column=2, padx = 5)
        self.om2 = CTkOptionMenu(self, width=10 ,values=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'ndvi', 'msavi', 'ipvi'], variable=self.om2_value)
        self.om2.grid(row=1, column=3,padx=10, pady=10)
        self.lbl3 = CTkLabel(self, text='band3 : ')
        self.lbl3.grid(row=1, column=4, padx = 5)
        self.om3 = CTkOptionMenu(self,width = 10 ,values=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'ndvi', 'msavi', 'ipvi'], variable=self.om3_value)
        self.om3.grid(row=1, column=5,padx = 10, pady=10)
        self.btm1 = CTkButton(self , text='create image' , command = lambda:bc1(self))
        self.btm1.grid(row = 2 , column = 0 , columnspan = 6 , pady = 10)

        # self.sld1 = CTkSlider(self, width=200, from_=0, to=90, number_of_steps=90, command=sl3)


class frame4(CTkFrame):
    def __init__(self , master):
        super().__init__(master)
        self.root = master
        self.frame_name = CTkLabel(self , text='fourier')
        self.frame_name.grid(row=0 , column = 0 , columnspan = 4 , pady = 10)
        #functions
        def sl1(value):
            self.lbl2.configure(text=str(int(self.sld1.get())))
        def bc1(self):
            image = rio.open(self.root.path6)
            img = image.read()
            img_vis = img[1:4, :, :]
            img_vis = UINT8(img_vis)
            img_vis = img_vis.transpose(1, 2, 0).astype(np.uint8)
            filter = self.radiovar.get()
            ft , inv , ginv = fourier(img ,self.sld1.get() ,filter)
            dict = {'shift_fft_image' : ft,'after gaussian' :inv ,'after_inverse' : ginv}
            var = self.om1.get()
            img2 = dict[var]
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            figure.add_subplot(121)
            plt.imshow(img_vis)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(img2)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

            return None
        def bc2(self):
            type = self.om2.get()
            size = int(self.om3.get())
            image = rio.open(self.root.path6)
            img = image.read()
            img_vis = img[14:, :, :]
            img_vis = UINT8(img_vis)
            img_vis = img_vis.transpose(1, 2, 0).astype(np.uint8)
            blur , gaussian_blur , sobel = apply_kernel(img ,size)
            dict = {'blur':blur, 'gaussian blur':gaussian_blur, 'sobel':sobel}
            img2 = dict[type]
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            figure.add_subplot(121)
            plt.imshow(img_vis)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(img2.astype(np.uint8))
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

        #widgets
        self.radiovar = tkinter.IntVar(master=self, value=0)
        self.om1_value = StringVar(value='shift_fft_image')
        self.om2_value = StringVar(value='blur')
        self.om3_value = StringVar(value='3')

        self.lbl1 = CTkLabel(self, text='kernel diameter')
        self.lbl1.grid(row=1, column=0)
        self.sld1 = CTkSlider(self ,width = 100 ,from_ = 10 ,to = 300 , number_of_steps=290 , command = sl1)
        self.sld1.grid(row = 1 , column=1)
        self.lbl2 = CTkLabel(self, text=f'{self.sld1.get()}')
        self.lbl2.grid(row=1, column=2, sticky='w')
        self.rb1 = CTkRadioButton(master=self, text='low_pass', variable=self.radiovar, value=1)
        self.rb1.grid(row=2, column=0, padx=10, pady=4, sticky='w')
        self.rb2 = CTkRadioButton(master=self, text='high_pass', variable=self.radiovar, value=2)
        self.rb2.grid(row=3, column=0, padx=10, pady=4, sticky='w')
        self.om1 = CTkOptionMenu(self, values=['shift_fft_image','after gaussian','after_inverse'], variable=self.om1_value)
        self.om1.grid(row = 2 , column = 1 , rowspan = 2)
        self.btm1 = CTkButton(self ,width = 100 ,text = 'show image' , command = lambda:bc1(self))
        self.btm1.grid(row=2 , column = 2,padx = 10 , rowspan = 2)
        self.lbl3 = CTkLabel(self,width = 60,text = 'kernel type: ')
        self.lbl3.grid(row = 4 ,column = 0 , pady = 10)
        self.om2 = CTkOptionMenu(self,values=['blur', 'gaussian blur', 'sobel'],variable=self.om2_value)
        self.om2.grid(row=4, column=1)
        self.lbl4 = CTkLabel(self,width = 70 , text = 'kernel size: ')
        self.lbl4.grid(row = 4 , column = 2)
        self.om3 = CTkOptionMenu(self,width = 30 ,values=['3','5','7','9'],variable=self.om3_value)
        self.om3.grid(row = 4 , column = 3 , padx = 10)
        self.btm2 = CTkButton(self , text = 'apply kernel',width=50 , command = lambda:bc2(self))
        self.btm2.grid(row = 5 , column = 0 ,columnspan = 4 , pady = 10)


class frame5(CTkScrollableFrame):
    def __init__(self , master , **kwargs):
        super().__init__(master , **kwargs)
        self.root = master
        #functions
        def bc1(self):
            method = self.om1.get()
            img1 = cv2.imread(self.root.path7)
            img2 = img1[60: 260, 100: 370, :]
            (rows, cols) = img2.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -45, 1)
            img2 = cv2.warpAffine(img2, M, (cols, rows))
            if method == 'harris':
                img1_harris = harris(img1)
                img2_harris = harris(img2)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img1_harris)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(img2_harris)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            elif method == 'moravec':
                coord1 = moravec(img1)
                coord2 = moravec(img2)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img1)
                plt.axis('off')
                plt.plot(coord1[:, 1], coord1[:, 0], 'b*')
                figure.add_subplot(122)
                plt.imshow(img2)
                plt.axis('off')
                plt.plot(coord2[:, 1], coord2[:, 0], 'b*')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            elif method == 'shi-tomasi':
                coord1 = shi_tomasi(img1)
                coord2 = shi_tomasi(img2)
                img1_vis = img1.copy()
                img2_vis = img2.copy()
                for i in coord1:
                    x, y = i.ravel()
                    cv2.circle(img1_vis, (x, y), 3, 255, -1)
                for i in coord2:
                    x, y = i.ravel()
                    cv2.circle(img2_vis, (x, y), 3, 255, -1)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img1_vis)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(img2_vis)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

            elif method == 'sift':
                img1_sift = Sift(img1)
                img2_sift = Sift(img2)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img1_sift)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(img2_sift)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
        def sl1(value):
            self.lbl4.configure(text=str(int(self.sld1.get())))
        def sl2(value):
            self.lbl6.configure(text=str(int(self.sld2.get())))
        def sl3(value):
            self.lbl8.configure(text=str(int(self.sld3.get())))
        def sl4(value):
            self.lbl10.configure(text=str(int(self.sld4.get())))
        def sl5(value):
            self.lbl12.configure(text=str(int(self.sld5.get())))
        def sl6(value):
            self.lbl18.configure(text=str(int(self.sld6.get())))
        def bc2(self):
            method = self.radiovar.get()
            use_thr = self.checkvar.get()
            img = cv2.imread(self.root.path1)
            if method == 1 :
                thre = self.sld1.get()
                edge = robert(img , thre , use_thr)
            elif method == 2 :
                thre = self.sld2.get()
                edge = prewit(img , thre , use_thr)
            elif method == 3 :
                thre = self.sld3.get()
                edge = sobel(img , thre , use_thr)
            elif method == 4:
                lower_ther = self.sld4.get()
                upper_ther = self.sld5.get()
                edge = canny(img , lower_ther , upper_ther)
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(edge)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)


        def bc3(self):
            dialog = CTkInputDialog(text="please type full path to your image:", title="input image")
            path = dialog.get_input()
            img = cv2.imread(path)
            direction = self.om2.get()
            dim = self.om3.get()
            Energy , Entropy , Max_pro , Contrast , Inv_diff_mom , Corr = features(img , direction)
            self.root.text_box.configure(state='normal')
            self.root.text_box.delete('0.0', 'end')
            self.root.text_box.insert('end', '********************** energy **********************\n')
            self.root.text_box.insert('end', f'{Energy:.4f}\n')
            self.root.text_box.insert('end', '********************** entropy **********************\n')
            self.root.text_box.insert('end', f'{Entropy:.4f}\n')
            self.root.text_box.insert('end', '********************** maximum probability **********************\n')
            self.root.text_box.insert('end', f'{Max_pro:.4f}\n')
            self.root.text_box.insert('end', '********************** contrast **********************\n')
            self.root.text_box.insert('end', f'{Contrast:.4f}\n')
            self.root.text_box.insert('end',
                                      '********************** inverse difference moment **********************\n')
            self.root.text_box.insert('end', f'{Inv_diff_mom:.4f}\n')
            self.root.text_box.insert('end', '********************** correlation **********************\n')
            self.root.text_box.insert('end', f'{Corr}\n')
            self.root.text_box.configure(state='disable')
            Dim = NineD(img , dim)
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(Dim)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
        def bc4(slef):
            img = cv2.imread(self.root.path8)
            direction = self.om2.get()
            dim = self.om3.get()
            Energy , Entropy , Max_pro , Contrast , Inv_diff_mom , Corr = features(img, direction)
            self.root.text_box.configure(state='normal')
            self.root.text_box.delete('0.0', 'end')
            self.root.text_box.insert('end', '********************** energy **********************\n')
            self.root.text_box.insert('end', f'{Energy:.4f}\n')
            self.root.text_box.insert('end', '********************** entropy **********************\n')
            self.root.text_box.insert('end', f'{Entropy:.4f}\n')
            self.root.text_box.insert('end', '********************** maximum probability **********************\n')
            self.root.text_box.insert('end', f'{Max_pro:.4f}\n')
            self.root.text_box.insert('end', '********************** contrast **********************\n')
            self.root.text_box.insert('end', f'{Contrast:.4f}\n')
            self.root.text_box.insert('end', '********************** inverse difference moment **********************\n')
            self.root.text_box.insert('end', f'{Inv_diff_mom:.4f}\n')
            self.root.text_box.insert('end', '********************** correlation **********************\n')
            self.root.text_box.insert('end', f'{Corr:.4f}\n')
            self.root.text_box.configure(state='disable')
            Dim = NineD(img, dim)
            figure = plt.figure(figsize=(11, 5), dpi=100)
            # figure.patch.set_facecolor('#639DC1')
            figure.add_subplot(121)
            plt.imshow(img)
            plt.axis('off')
            figure.add_subplot(122)
            plt.imshow(Dim)
            plt.axis('off')
            chart = FigureCanvasTkAgg(figure, self.root)
            chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)




        def bc5(self):
            method = self.om5.get()
            numc = int(self.sld6.get())
            if method == 'threshold based':
                img = cv2.imread(self.root.path9)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = ther_based(img)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(mask)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

            elif method == 'watershed' :
                img = cv2.imread(self.root.path9)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = img.copy()
                mask = watershed(image)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(mask)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            elif method == 'k-means' :
                img = cv2.imread(self.root.path10)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = img.copy()
                mask = k_means(image , numc)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(mask)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

            elif method == 'mean-shift':
                img = cv2.imread(self.root.path10)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = img.copy()
                mask = mean_shift(image)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(mask)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)
            elif method == 'FCM' :
                img = cv2.imread(self.root.path10)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                image = img.copy()
                image = cv2.cvtColor(image ,cv2.COLOR_RGB2GRAY)
                mask = fcm(image , numc)
                figure = plt.figure(figsize=(11, 5), dpi=100)
                # figure.patch.set_facecolor('#639DC1')
                figure.add_subplot(121)
                plt.imshow(img)
                plt.axis('off')
                figure.add_subplot(122)
                plt.imshow(mask)
                plt.axis('off')
                chart = FigureCanvasTkAgg(figure, self.root)
                chart.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=4, pady=10, padx=100)

        #widgets
        ################################################# corner detection #################################################
        self.radiovar = tkinter.IntVar(master=self, value=0)
        self.frame_name = CTkLabel(self, text='corner detection')
        self.frame_name.grid(row=0, column=0, columnspan=5 , pady = 10)
        self.om1_value = StringVar(value='harris')
        self.om2_value = StringVar(value='0')
        self.om3_value = StringVar(value='L5E5/E5L5')
        self.om4_value = StringVar(value='7')
        self.om5_value = StringVar(value='watershed')
        self.checkvar = StringVar(value = 'on')
        self.lbl1 = CTkLabel(self , text = 'method:')
        self.lbl1.grid(row = 1 , column = 0)
        self.om1 = CTkOptionMenu(self,width = 100 ,values=['harris','moravec','sift' , 'shi-tomasi'], variable=self.om1_value)
        self.om1.grid(row=1, column = 1, padx = 10 , pady = 10)
        self.btm1 = CTkButton(self , text='corner detection' , command = lambda:bc1(self))
        self.btm1.grid(row = 1 , column = 3 , columnspan = 2 , padx = 40)
        ################################################# edge detection #################################################
        self.lbl2 = CTkLabel(self , text = 'edge detection')
        self.lbl2.grid(row = 2 , column = 0 , columnspan = 5 , pady = 10)
        self.rb1 = CTkRadioButton(master=self,width = 70 ,text='robert', variable=self.radiovar, value=1)
        self.rb1.grid(row=3, column=0 , padx=10, pady=4, sticky='w')
        self.lbl3 = CTkLabel(self ,width = 80 ,text = 'threshold:')
        self.lbl3.grid(row = 3 , column = 1 , sticky = 'e')
        self.sld1 = CTkSlider(self ,width = 200,from_=1, to=10, number_of_steps=10, command=sl1)
        self.sld1.grid(row=3, column=2 , columnspan = 2)
        self.lbl4 = CTkLabel(self ,text=str(int(self.sld1.get())))
        self.lbl4.grid(row=3, column=4, sticky='w')
        self.rb2 = CTkRadioButton(master=self ,width = 70 ,text='prewit', variable=self.radiovar, value=2)
        self.rb2.grid(row=4, column=0, padx=10, pady=4, sticky='w')
        self.lbl5 = CTkLabel(self, width=80, text='threshold:')
        self.lbl5.grid(row=4, column=1, sticky='e')
        self.sld2 = CTkSlider(self,width = 200, from_=1, to=10, number_of_steps=10, command=sl2)
        self.sld2.grid(row=4, column=2 , columnspan = 2)
        self.lbl6 = CTkLabel(self, text=str(int(self.sld2.get())))
        self.lbl6.grid(row=4, column=4, sticky='w')
        self.rb3 = CTkRadioButton(master=self,width = 70 , text='sobel', variable=self.radiovar, value=3)
        self.rb3.grid(row=5, column=0, padx=10, pady=4, sticky='w')
        self.lbl7 = CTkLabel(self, width=80, text='threshold:')
        self.lbl7.grid(row=5, column=1, sticky='e')
        self.sld3 = CTkSlider(self,width = 200, from_=50, to=200, number_of_steps=10, command=sl3)
        self.sld3.grid(row=5, column=2 , columnspan =2)
        self.lbl8 = CTkLabel(self, text=str(int(self.sld3.get())))
        self.lbl8.grid(row=5, column=4, sticky='w')
        self.rb4 = CTkRadioButton(master=self,width = 70 , text='canny', variable=self.radiovar, value=4)
        self.rb4.grid(row=6,rowspan = 2 ,column=0, padx=10, pady=4, sticky='w')
        self.lbl9 = CTkLabel(self, width=80, text='lower thereshold:')
        self.lbl9.grid(row=6, column=1, sticky='e')
        self.sld4 = CTkSlider(self,width = 200, from_=0, to=190, number_of_steps=100, command=sl4)
        self.sld4.grid(row=6, column=2 , columnspan = 2)
        self.lbl10 = CTkLabel(self, text=str(int(self.sld4.get())))
        self.lbl10.grid(row=6, column=4, sticky='w')
        self.lbl11 = CTkLabel(self, width=80, text='upper thereshold:')
        self.lbl11.grid(row=7, column=1, sticky='e')
        self.sld5 = CTkSlider(self,width = 200, from_=200, to=400, number_of_steps=200, command=sl5)
        self.sld5.grid(row=7, column=2,columnspan = 2)
        self.lbl12 = CTkLabel(self, text=str(int(self.sld5.get())))
        self.lbl12.grid(row=7, column=4, sticky='w')
        self.cb1 = CTkCheckBox(self , text = 'use threshold to create mask' ,variable=self.checkvar , onvalue='on' ,offvalue='off')
        self.cb1.grid(row = 8 , column = 0 , columnspan = 2)
        self.btm2 = CTkButton(self , text = 'edge detection' , command = lambda:bc2(self))
        self.btm2.grid(row = 8 , column = 2 , columnspan = 3 , pady = 10)
        ################################################# texture #################################################
        self.lbl13 = CTkLabel(self , text = 'texture')
        self.lbl13.grid(row = 9 , column = 0 , columnspan = 5 , pady = 10)
        self.lbl14 = CTkLabel(self ,width = 60 ,text = 'direction:')
        self.lbl14.grid(row = 10 , column = 0)
        self.om2 = CTkOptionMenu(self, width=50, values=['0', '45', '90'],variable=self.om2_value)
        self.om2.grid(row=10, column=1, padx=10, pady=10, sticky = 'w')
        self.lbl14 = CTkLabel(self, text='feature:')
        self.lbl14.grid(row=10, column=2)
        self.om3 = CTkOptionMenu(self, width=100, values=['L5E5/E5L5' , 'L5R5/R5L5' , 'E5S5/S5E5' , 'S5S5' , 'R5R5' , 'L5S5/S5L5' , 'E5E5' , 'E5R5/R5E5' , 'S5R5/R5S5'],variable=self.om3_value)
        self.om3.grid(row=10, column=3 ,columnspan = 2 , padx=10, pady=10)
        self.btm3 = CTkButton(self , text = 'input a new picture' ,command = lambda : bc3(self))
        self.btm3.grid(row = 11 , column = 0 , columnspan = 2)
        self.btm4 = CTkButton(self , text = 'predefined image' , command = lambda: bc4(self))
        self.btm4.grid(row = 11 , column = 3 , columnspan = 2)
        ################################################# image segmentation #################################################
        self.lbl15 = CTkLabel(self , text = 'image segmentation')
        self.lbl15.grid(row = 12 , column = 0 , columnspan = 5 , pady = 10)
        self.lbl16 = CTkLabel(self , text = 'method:')
        self.lbl16.grid(row = 13 , column = 0)
        self.om5 = CTkOptionMenu(self , values = ['threshold based' , 'watershed' , 'k-means' , 'mean-shift' , 'FCM'] ,variable=self.om5_value)
        self.om5.grid(row = 13 , column = 1, sticky = 'w')
        self.btm5 = CTkButton(self , text = 'show result' , command = lambda:bc5(self))
        self.btm5.grid(row = 13 , column = 3 , columnspan = 2)
        self.lbl17 = CTkLabel(self , text = 'number of clusters:')
        self.lbl17.grid(row = 14 , column = 0 , columnspan = 2)
        self.sld6 = CTkSlider(self, width=200, from_=3, to=8, number_of_steps=6, command=sl6)
        self.sld6.grid(row=14, column=2 , columnspan = 2)
        self.lbl18 = CTkLabel(self , text = str(int(self.sld6.get())))
        self.lbl18.grid(row = 14 , column = 4)
app = Window()
app.mainloop()

