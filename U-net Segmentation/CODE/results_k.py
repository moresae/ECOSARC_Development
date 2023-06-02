# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np # for using np arrays
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import jaccard_score

import pandas as pd
# """  LOAD AND SCALE DATA  """

X = np.load("X.npy")
y = np.load("Y.npy")



"""SPLIT DATA INTO TRAIN-VALIDATION-TEST"""
valid_files=pd.read_csv('valid_files.csv')  
x_train_val, x_test, y_train_val, y_test,files_train,files_test  = train_test_split(X,y,valid_files,test_size=0.1,random_state=42,shuffle=False)

test_k1=np.load("y_pred_xtest_k_1.npy")
test_k2=np.load("y_pred_xtest_k_2.npy")
test_k3=np.load("y_pred_xtest_k_3.npy")
test_k4=np.load("y_pred_xtest_k_4.npy")
test_k5=np.load("y_pred_xtest_k_5.npy")


#SAVE FINAL MASKS WITH NAMES RELATIVE TO ORIGINAL NAMES
for i in range(len(test_k1)):
  im1 = test_k1[i]/255
  im1=cv2.resize(im1,dsize=(880, 1000), interpolation=cv2.INTER_LINEAR) 

  im2 = test_k2[i]/255
  im2=cv2.resize(im2,dsize=(880, 1000), interpolation=cv2.INTER_LINEAR) 

  im3 = test_k3[i]/255
  im3=cv2.resize(im3,dsize=(880, 1000), interpolation=cv2.INTER_LINEAR) 

  im4 = test_k4[i]/255
  im4=cv2.resize(im4,dsize=(880, 1000), interpolation=cv2.INTER_LINEAR) 

  im5 = test_k5[i]/255
  im5=cv2.resize(im5,dsize=(880, 1000), interpolation=cv2.INTER_LINEAR) 
  
  # #---------------------------------
  # # Save  the final masks as png
  # #---------------------------------
  filei=files_test.iloc[i]['filename']
  plt.imshow(im1.reshape(im1.shape[0],im1.shape[1]))
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig('./mask_k1/'+filei+'_k1.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  plt.close()


  plt.imshow(im2.reshape(im1.shape[0],im1.shape[1]))
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig('./mask_k2/'+filei+'_k2.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  plt.close()

  plt.imshow(im3.reshape(im1.shape[0],im2.shape[1]))
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig('./mask_k3/'+filei+'_k3.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  plt.close()


  plt.imshow(im4.reshape(im1.shape[0],im1.shape[1]))
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig('./mask_k4/'+filei+'_k4.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  plt.close()



  plt.imshow(im5.reshape(im5.shape[0],im1.shape[1]))
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig('./mask_k5/'+filei+'_k5.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  plt.close()

