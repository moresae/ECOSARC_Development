
import os
from reprlib import aRepr
# for reading and processing images
import imageio
import matplotlib.pyplot as plt
import numpy as np # for using np arrays
from numpy import asarray
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import os

Xx = np.load("X.npy")
yy = np.load("Y.npy")




def scale_dataset(i_h,i_w):
  m,a,b=Xx.shape
  X = np.zeros((m,i_h,i_w), dtype=np.float32)
  y = np.zeros((m,i_h,i_w), dtype=np.float32)

  i=0
  for img in Xx:
    scaled = cv2.resize(img,dsize=(i_w,i_h), interpolation=cv2.INTER_LINEAR) 
    X[i] =scaled
    i=i+1
  i=0
  for mask in yy:
    scaled = cv2.resize(mask,dsize=(i_w,i_h), interpolation=cv2.INTER_LINEAR) 
    y[i] =scaled
    i=i+1

  return X,y

X,y = scale_dataset(400,256)

mask1= y[0]
mask2= y[1]
np.save('mask1.npy',mask1)
np.save('mask2.npy',mask2)



print('X and y ready scaled')

## ------------------------------------------------------------------------------------------------


"""SPLIT DATA INTO TRAIN-VALIDATION-TEST"""
from sklearn.model_selection import train_test_split

#Divide dataset in train-val-test
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print(x_train.shape)
#print(x_val.shape)
#print(x_test.shape)

"""#   DATA AUGMENTATION ON THE TRAINING DATASET"""
def augment_image(img):
  im1 =scale_horizontal(img,10)
  im2 =scale_horizontal(img,-10)
  im3 =scale_vertical(img,15)
  im4 =scale_vertical(img,-15)
  im5 =horizontal_mirror(img)
  im6 = change_brightness(img,0.5)
  im7 = change_brightness(img,0.8)
  return im1,im2,im3,im4,im5,im6,im7

def scale_vertical(img,factor):
  new_height = int(img.shape[0]*(1+factor/100))
  dim_size = (img.shape[1],new_height)
  vertical_img = cv2.resize(img, dim_size, interpolation=cv2.INTER_AREA)
  return vertical_img

# horizontal scaling
def scale_horizontal(img,factor):
  new_width = int(img.shape[1]*(1+factor/100))
  dim_size = (new_width,img.shape[0])
  horizon_img = cv2.resize(img, dim_size, interpolation=cv2.INTER_AREA)
  return horizon_img

def horizontal_mirror(img):
  return np.fliplr(img)

def change_brightness(image,gain):
   new_image = tf.image.adjust_brightness(image, gain)
   return new_image.numpy()


def image_augment(na,i_h,i_w):
  # Pull the relevant dimensions for image and mask
  m,a,b = x_train.shape  # pull height, width, and channels of image

  # Define X and Y as number of images along with shape of one image
  Xa = np.zeros((m*na,i_h,i_w), dtype=np.float32)
  ya = np.zeros((m*na,i_h,i_w), dtype=np.float32)

  i = 0 
  for single_img in x_train:
    im1,im2,im3,im4,im5,im6,im7 = augment_image(single_img)

    Xa[i]= cv2.resize(im1, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1
    
    Xa[i]=cv2.resize(im2,dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1 
    
    Xa[i]=cv2.resize(im3,dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1
    
    Xa[i]=cv2.resize(im4, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1 
    
    Xa[i]=cv2.resize(im5,dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1

    Xa[i]=cv2.resize(im6, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1

    Xa[i]=cv2.resize(im7, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1

  i = 0    
  for single_mask in y_train:
    
    im1,im2,im3,im4,im5,im6,im7 = augment_image(single_mask)

    ya[i]=cv2.resize(im1, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1
    
    ya[i]=cv2.resize(im2,dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1 
    
    ya[i]=cv2.resize(im3, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC)  
    i+=1
    
    ya[i]=cv2.resize(im4,dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC)  
    i+=1 
    
    ya[i]=cv2.resize(im5, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1

    ya[i]=cv2.resize(im6, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1

    ya[i]=cv2.resize(im7, dsize=(i_w,i_h), interpolation=cv2.INTER_CUBIC) 
    i+=1   

       
  return Xa,ya


na = 7
Xa,ya = image_augment(na,400, 256)


#np.save('Xa.npy', Xa)
#np.save('ya .npy', ya )

#Train dataset: add the augmentnted data
X_train_augm = np.concatenate((x_train,Xa))
y_train_augm = np.concatenate((y_train,ya))



X_train_augm = X_train_augm/255

i1 = X_train_augm[1]
print(np.max(i1))
