# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:38:55 2018

@author: Rahul Verma
"""
import pandas as pd
import numpy as np
from skimage.io import imsave
from PIL import Image 
import PIL
import os

from random import randint
from sklearn.preprocessing import scale
#data=pd.read_csv("C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\fer2013\\fer2013.csv")
#train_data = data[data.Usage == "Training"]
#public_data = data[data.Usage == "PublicTest"]
#private_data = data[data.Usage == "PrivateTest"]


#train_pixels_values = train_data.pixels.str.split(" ").tolist()
#train_pixels_values = pd.DataFrame(train_pixels_values, dtype=int)
#train_images = train_pixels_values.values
#train_images = train_images.astype(np.uint8)
#train_labels_flat = train_data["emotion"].values.ravel()


#public_pixels_values = public_data.pixels.str.split(" ").tolist()
#public_pixels_values = pd.DataFrame(public_pixels_values, dtype=int)
#public_images = public_pixels_values.values
#public_images = public_images.astype(np.float)
#public_labels_flat = public_data["emotion"].values.ravel()


#private_pixels_values = private_data.pixels.str.split(" ").tolist()
#private_pixels_values = pd.DataFrame(private_pixels_values, dtype=int)
#private_images = private_pixels_values.values
#private_images = private_images.astype(np.float)
#private_labels_flat = private_data["emotion"].values.ravel()


def create_image(data,labels,path):

    
    for row in range(data.shape[0]):
        k=data[row]
        image=k.reshape((48,48)).astype(np.uint8)
        label=labels[row]
        imsave(path+str(row)+"_"+str(label)+".png",image)
        #img = np.concatenate((image,image,image),axis=2)
        #print(image.shape)
        #img = Image.fromarray(img)
        #img.save(path+str(row)+".png",img)
    return 


#create_image(train_images,train_labels_flat,"C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Train_img\\")
#create_image(public_images,public_labels_flat,"C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Test_img\\public\\")
#create_image(private_images,private_labels_flat,"C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Test_img\\private\\")

#train_path="C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Train_img\\"
#public_test_path="C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Test_img\\public\\"
#private_test_path="C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Test_img\\private\\"

def data_augmentation(path):
    files=os.listdir(path)
    for i in range(len(files)):
        img=Image.open(path+files[i])
        mirror=img.transpose(method=Image.FLIP_LEFT_RIGHT)
        rotate=img.rotate(angle=randint(-45, 45))
        resize_int=randint(42,54)
        resized=rotate.resize((resize_int,resize_int))
        w,h=resized.size
        #print(w,h)
        dx,dy=42,42
        x=randint(0,w-dx)
        y=randint(0,h-dy)
        cropped=resized.crop((x,y,x+dx,y+dy))
        cropped.save(path+"aug_"+files[i])
    return

#data_augmentation(train_path)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



def create_preprocess_data(path):
    files=os.listdir(path)
    data_list=[]
    y_list=[]
    for i in range(len(files)):
        img=Image.open(path+files[i])
        Image_name=files[i]
        if "aug" in Image_name:
            continue
            #m,n=Image_name.split(".")
            #h,k,l=m.split("_")
            
            #l=int(l)
            #y_list.append(l)  
        else:
            m,n=Image_name.split(".")
            a,b=m.split("_")
            b=int(b)
            #print(a,b)
            y_list.append(b)
        if(img.size==(48,48)):
            img=img.resize((42,42))
        I = np.asarray(img).astype(np.float32).ravel()       
        data_list.append(I)
        
    data=np.asarray(data_list)
    data=data-np.mean(data,axis=1).reshape(-1,1)
    data=np.divide(data,3.125)
    data=data-np.mean(data,axis=0)
    data=np.divide(data,np.std(data,axis=0))
    
    #Create X,Y
    data_l=[]
    for i in range(data.shape[0]):
        I=data[i]
        I=I.reshape((42,42,1))
        data_l.append(I)

    X=np.asarray(data_l)
    Y=np.asarray(y_list).reshape((len(y_list),1))
    Y=dense_to_one_hot(Y,7)
    return X,Y


#X,Y=create_preprocess_data(train_path)


#from sklearn.model_selection import train_test_split

#X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.1,random_state=57)
         
#def load_public_test(path):
    #files=os.listdir(path)
    #test_files=[]
    #Y_list=[]
    #for i in range(len(files)):
        #img=Image.open(path+files[i])
        #Image_name=files[i]
        #if(img.size==(48,48)):
         #   img=img.resize((42,42))
       # m,n=Image_name.split(".")
        #a,b=m.split("_")
        #b=int(b)
            #print(a,b)
        #Y_list.append(b)    
        #I=np.asarray(img).resize((42,42,1))
        #test_files.append(I)
    #X=np.asarray(test_files)
    #Y=np.asarray(Y_list).reshape((len(Y_list),1))
    #Y=dense_to_one_hot(Y,7)
    #return X,Y
     
    






        