# settings for data labelling, import, and preprocessing

import sys

# path to liver_dl_pipeline folder
path_to_main = ''

# add path to custom libraries
sys.path.insert(1, path_to_main + 'libraries')

import _1_imports
from _1_imports import *
#%run -i libraries/_1_imports.py

import _2_PD_setup
from _2_PD_setup import *

import _3_data_select
from _3_data_select import *

import _4_data_pipe_df
from _4_data_pipe_df import *

import _4_data_pipe_sequential
from _4_data_pipe_sequential import *

import _5_data_modify
from _5_data_modify import *

import _6_model_eval
from _6_model_eval import *

# show available devices (make sure GPU is there)
#print(device_lib.list_local_devices())

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# provide number of cores available for use
num_cores = multiprocessing.cpu_count()
#num_cores = 16

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TO LOAD AN EXISTING DATASET (ie. completed labelling and preprocessing) that was created and saved using 0_main_dataprep.py
# skip if making a new dataset

load_existing_dataset = False
if load_existing_dataset:
    dataset_name = ''
    sys.exit("Existing dataset will be loaded. Bypassing standard preprocessing settings")

'''
## Sergio 10.12.2020 - large dataset annotated with labels 3
load_existing_dataset = True
if load_existing_dataset:
    dataset_name = 'rf_dataset_2020_12_10_17_17'
    print("Existing dataset will be loaded. Bypassing standard preprocessing settings")

## Sergio 09.12.2020 - small dataset

load_existing_dataset = True
if load_existing_dataset:
    dataset_name = 'rf_dataset_2020_12_9_16_33'
    print("Existing dataset will be loaded. Bypassing standard preprocessing settings")
'''


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TO USE THE SAME TRAIN/VAL/TEST SPLIT AS AN EXISTING DATASET (note: below, you NEED to select the same labelling scheme as the chosen dataset, if you use this option)
# this uses the existing split in terms of CASES in train and validation, so it will not necessarily give you the frames in the same order

use_existing_split = False
if use_existing_split:
    dataset_name = ''

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# USE THE EXACT SAME TRAIN/VAL SPLIT (when using a different data type than the saved dataset you want to copy the split from)
# use at your own risk - THIS WILL ONLY WORK IF instance_rate == 1 for the dataset you are taking the split from
# it will give you the exact same split, with the same frame order

use_exact_existing_split = False
if use_exact_existing_split:
    dataset_name = ''


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# select labelling type, then provide appropriate labels in the relevant section
# use '-1' for cases you want to omit, use '0','1', and onward for cases you want to use

# 4 is introduced for steatosis labelling
labelling = 3

if labelling == 1:
    
    # SIMPLE BINARY LABELLING: (healthy and F0) vs (anything above min_kpa)
    #min_kpa = 7.0
    
    # healthy,F0-F2 vs F3,F4,C
    min_kpa = 9.0
    
    # healthy,F0-F3 vs F4,C
    #min_kpa = 17.0

if labelling == 2:
    
    # more detailed LABELLING

    h_c = '0'
    f0_c = '0'
    f2_c = '-1'
    f3_c = '1'
    f4_c = '-1'
    c_c = '-1'

if labelling == 3:
    
    # even more detailed LABELLING

    h_c = '-1'
    f0_c = '0'
    f2_c = '1'
    f3_c = '1'
    f4_c = '1'
    f_noKPA = '-1'
    c_c = '-1'
    
    # use true if you want to give weird cases a specific label
    # use '-1' t omit unusual cases
    other_c = '-1'
    other_cases = False

if labelling == 4: ### Steatosis labelling
    s1_cut = 238
    s2_cut = 260
    s3_cut = 290
    s0_c = '0'
    s1_c = '1'
    s2_c = '2'
    s3_c = '3'
    f_c = '-1'
    h_c = '-1'



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# choose to remove shadowy frames

remove_shadowing = True

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# select data type and shape ( 'rf' , 'quant' , 'bmode' , 'nps' , 'ps' , 'qus' )
# instance_rate referres to the number of data instances acquired from each frame. Typically the frames themselves are used as the instances( meaning instance_rate = 1 )

#data_type = 'nps'
#data_type = 'ps'
data_type = 'rf'
use_default_shape = False
if use_default_shape:
    data_shape , data_dim , data_channels = define_data_shape( data_type )
    instance_rate = 1
    
    # insert modification function definition
    # the default is no modification
    def no_mod(input):
        return output
    
    modify_function = no_mod
    
else:
    # insert custom data shape
    #data_shape = (16,12,91,1)
    data_shape = (2928,1)
    ### Sergio: For 2D -> data_shape = (2928, 192, 1)
      

    #data_shape = ( , , , )
    data_dim = data_shape[:len(data_shape)-1]
    data_channels = data_shape[len(data_shape)-1]
    
    # if using custom data shape, a function is required to modify the original data
    # examples of custom functions can be found in the library _5_data_modify.py, which can be referenced here
    # if your instance rate is > 1, your modification function should return a list of the produced data instances that were retrieved from the single frame
    
    #instance_rate = 1
    instance_rate = 192
    # Sergio: For 2D -> instance_rate = 1

    #instance_rate = 1 #test Sergio 09.12.2020

    # insert modification function definition
    # the default is no modification
    def no_mod(input):
        return input
    
    modify_function = no_mod
    # example of another function
    #modify_function = expand_dims_1
    modify_function = rf_1D
    

    # Sergio: For 2D > modify_function = expand_dims_1 (add dummy channel with dim = 1)
custom_save_label = ""  ## Sergio, allows adding details to model save name

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# choose train/val/test split settings

# shuffle before splitting and/or balancing if you want a unique set every time
# note that automatic shuffling is done for the train and validation sets after they have been determined
trainval_shuffle = False

# choose sizes
# if you want to use the full size, set numTotal_train_val to -1
#numTotal_train_val = -1
#numTotal_train_val = 20000
numTotal_train_val = 2500 # Test Sergio 09.12.2020
# fraction of data in train
split_frac = 0.7
# choose to balance sets or leave them unbalanced (ie. have equal proportions of all classes/labels within a set)
train_even = False
if train_even:
    # if you want to balance the train set, you have the option of duplicating data from labels lacking data or to remove data from labels with most data
    # True to duplicate data that is lacking, False to remove data from most common labels
    train_duplicate = True
val_even = False#True
test_even = False#True

# only valid when instance_rate = 1
# choose seed with which to shuffle the train dataframe after the train/val split has been created (regardless of whether or not use_existing_split == True)
# if you want a random seed then use that option
chosen_seed = 0
use_random_seed = True

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# select data loading method (loading all data into RAM or loading sequentially - note that sequential loading is slow for training, but fine for testing)
# for sequential loading, data needs to be copied locally onto the instance - this can be done easily with data2instance.ipynb
# for loading into RAM, data can be fetched locally or from the GCP buckets
# there is no need to load the explicit test set into RAM, so that option is withheld

train_val_RAM = True

if train_val_RAM:
    # True if data has been imported to the instance
    local = True # True -> False by Sergio201106

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# decide if you would like to save your train/val/test split for reuse later
# if train/val has been loaded into RAM, this will save the train and val data as well so they can easily be reloaded later

save_datasets = True

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------