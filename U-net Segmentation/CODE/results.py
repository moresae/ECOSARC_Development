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
  im1 = test_k1[i]
  im2 = test_k2[i]
  im3 = test_k3[i]
  im4 = test_k4[i]
  im5 = test_k5[i]


  img_out_k = (im1+im2+im3+im4+im5)/5



  #scale image to original resoultion
  img_out=cv2.resize(img_out_k,dsize=(880, 1000), interpolation=cv2.INTER_LINEAR) 
  
  y_true=y_test[i]/255
  y_pred=img_out
  

  filei=files_test.iloc[i]['filename']
  #save figures comparing inference with ground truth
  plt.figure(figsize=(20,10))
  plt.subplot(1,3,1)
  plt.imshow(x_test[i])
  plt.title('Original image')
  plt.subplot(1,3,2)
  plt.imshow(y_test[i].reshape(y_test[i].shape[0],y_test[i].shape[1]))
  plt.title('Original mask')
  plt.subplot(1,3,3)
  plt.imshow(img_out.reshape(img_out.shape[0],img_out.shape[1]))
  plt.title('Predicted mask averaged over k=5 models')
  
  plt.savefig('./test_ave/'+filei+'.png',format='png')
  plt.close()



  # #---------------------------------
  # # Save  the test mask for reference
  # #---------------------------------
  # filei=files_test.iloc[i]['filename']
  # plt.imshow(y_test[i].reshape(y_test[i].shape[0],y_test[i].shape[1]))
  # plt.axis('off')
  # plt.gca().set_axis_off()
  # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
  #             hspace = 0, wspace = 0)
  # plt.margins(0,0)
  # plt.gca().xaxis.set_major_locator(plt.NullLocator())
  # plt.gca().yaxis.set_major_locator(plt.NullLocator())

      
  # plt.savefig('./test_ave/'+filei+'_mask.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  # plt.close()


  
  #---------------------------------
  # Save  the final masks as png
  #---------------------------------
  # filei=files_test.iloc[i]['filename']
  # plt.imshow(img_out.reshape(img_out.shape[0],img_out.shape[1]))
  # plt.axis('off')
  # plt.gca().set_axis_off()
  # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
  #             hspace = 0, wspace = 0)
  # plt.margins(0,0)
  # plt.gca().xaxis.set_major_locator(plt.NullLocator())
  # plt.gca().yaxis.set_major_locator(plt.NullLocator())

      
  # plt.savefig('./test_ave/'+filei+'_out.png',format='png', bbox_inches = 'tight',      pad_inches = 0)
  # plt.close()




