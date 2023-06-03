# ------------------------------------------------------------------------------------------------------------------------------------
## Load Omar's RF data
## Peform transformations as necessary
## Store
## Perform DL training with each transformation
## Compute McNemar tests one to one.
#------------------------------------------------------------
# Sergio J Sanabria, 2021.
exec(open("0_settings_preprocessing_rfvsbmode_OMAR.py").read())
exec(open("0_settings_sergiorfvsbmodeliver_OMAR.py").read())

import matplotlib.pyplot as plt


#path_to_raw = "../QUS_48x61_newfeatsQUS_RaulSeg_251cases_MasksAndArrays/"
path_to_raw = '../'

### Prepare train and test sets
DATA_NAME = 'n35_rf_data.npy'
LABELS_NAME = 'n35_rf_labels.npy'
MASK_NAME = 'n35_rf_mask.npy'

prepare_train_test_split = False

test_set_list = [[12, 13, 16, 19, 20, 27, 33], [2, 3, 9, 24, 28, 32], [6, 22, 23, 29, 30, 35], [4, 8, 15, 17, 26, 34]]  #test_set = [1, 7, 10, 18, 25, 31]


if prepare_train_test_split == True:
    id_test_set = 0
    for test_set in test_set_list:
        bmode_data = np.load(path_to_raw + DATA_NAME)
        bmode_labels = np.load(path_to_raw + LABELS_NAME)
        bmode_mask = np.load(path_to_raw + MASK_NAME)

        test_set_ids = np.zeros(bmode_labels.shape[0])
        train_set_ids = np.array(test_set_ids)
        for id in range(train_set_ids.shape[0]):
            if int(bmode_labels[id,3]) in test_set:
                test_set_ids[id] = 1
        train_set_ids = 1- test_set_ids
        bmode_labels_train = bmode_labels[np.nonzero(train_set_ids)[0],:]
        bmode_labels_test = bmode_labels[np.nonzero(test_set_ids)[0],:]
        bmode_mask_train = bmode_mask[np.nonzero(train_set_ids)[0],:]
        bmode_mask_test = bmode_mask[np.nonzero(test_set_ids)[0],:]
        bmode_data_train = bmode_data[np.nonzero(train_set_ids)[0],:]
        bmode_data_test = bmode_data[np.nonzero(test_set_ids)[0],:]

        print(path_to_raw + 'train' + str(id_test_set) + '_' + DATA_NAME)
        np.save(path_to_raw + 'train' + str(id_test_set) + '_' + DATA_NAME, bmode_data_train)
        np.save(path_to_raw + 'test' + str(id_test_set) + '_' + DATA_NAME, bmode_data_test)
        np.save(path_to_raw + 'train' + str(id_test_set) + '_' + LABELS_NAME, bmode_labels_train)
        np.save(path_to_raw + 'test' + str(id_test_set) + '_' + LABELS_NAME, bmode_labels_test)
        np.save(path_to_raw + 'train' + str(id_test_set) + '_' + MASK_NAME, bmode_mask_train)
        np.save(path_to_raw + 'test' + str(id_test_set) + '_' + MASK_NAME, bmode_mask_test)
        id_test_set = id_test_set + 1


'''
#Open label information
#path_to_raw = "../QUS_48x61_newfeatsQUS_RaulSeq_datasetsmall/"
traindataframetovid = np.load( (path_to_raw + 'traindataframetovid.npy') ) #340 - 340 videos, 2 per patient (first video 8 frames, second 2 frames)
trainvidtopat = np.load( (path_to_raw + 'trainvidtopat.npy') ) # 170 patients - each with two videos - 10 frames/patient successively
trainlabframe = np.load( (path_to_raw + 'trainlabframe.npy') ) # 170 patients - each with two videos - 10 frames/patient successively
testdataframetovid = np.load( (path_to_raw + 'testdataframetovid.npy') ) #340 - 340 videos, 2 per patient (first video 8 frames, second 2 frames)
testvidtopat = np.load( (path_to_raw + 'testvidtopat.npy') ) # 170 patients - each with two videos - 10 frames/patient successively
testlabframe = np.load( (path_to_raw + 'testlabframe.npy') ) # 77 patients - each with two videos - 10 frames/patient successively
#Debugging - understanding patient population - generating labels
#plt.plot(indices, trainlabframe[:,2])
#classes = ntrainlabframe[:,0]
#num_class = len(np.nonzero(classes==0)[0])
#np.average(trainlabframe[0:680, 0])
'''

# ------------------------------------------------------------------------------------------------------------------------------------
load_existing_dataset =True
# preprocessing for generation of data in instance

if load_existing_dataset:
    pass
else:
    ############## 0 - General definitions ##################################
    #path_to_raw = "../QUS_48x61_newfeatsQUS_RaulSeq_datasetsmall/"
    path_to_raw = "../"
    inputnn = "../inputnn_PHILIPSBMODE/"

    ########### 1-Perform RF vs B-mode conversion + spectrogram calculation #####################################
    skipRFvsBmodeConversion = True
    if (skipRFvsBmodeConversion == False):
        #Variables
        whichdata = ['train', 'test']  #'train_',
        #list_labels_modify_functions = [ ]
        #list_labels_modify_functions = [  ["_iq", stages_rf2D_rftobmode_iq], ["_env", stages_rf2D_rftobmode_env], ["_log",stages_rf2D_rftobmode_log], ["_den",stages_rf2D_rftobmode_den],["_ps", stages_rf2D_ps], ["_psden", stages_rf2D_ps_denoising]]
        #list_labels_modify_functions = [ ["_pslogfirst", stages_rf2D_ps_normfirst], ["_phasefun", stages_rf2D_phasefun], ["_iq", stages_rf2D_rftobmode_iq], ["_env", stages_rf2D_rftobmode_env], ["_log",stages_rf2D_rftobmode_log], ["_den",stages_rf2D_rftobmode_den], ["_ps", stages_rf2D_ps_wihoutlog_v02], ["_pslog", stages_rf2D_ps_v02], ["_pslognorm", stages_rf2D_ps_normalized_v02], ["_pslognormden", stages_rf2D_ps_denoising_v02]]
        list_labels_modify_functions = [ ["_phasefun", stages_rf2D_phasefun],  ["_pslogfirst", stages_rf2D_ps_normfirst]]

        cross_validation_folds = [4]

        # Loop on validation folds
        for kfold in cross_validation_folds:
            #Data evaluation loop
            for xdata in whichdata:
                #Open data
                RF_array = np.load( (path_to_raw + xdata + str(kfold) + '_' + DATA_NAME )) # same size as trainRF_array, contains segmentation masks    
                Nframes = RF_array.shape[0]
                for xfun in list_labels_modify_functions:
                    custom_save_label = xfun[0]
                    modify_function = xfun[1]
                    # Apply transformation on data 
                    # Input data = (25,500, 27, 392,1) <-> Transformation occurs in dimensions 2,3
                    if custom_save_label[1:3] == 'ps':
                        RF_array_transform = np.zeros([Nframes, 191, 250, 33])
                    else:
                        RF_array_transform = np.zeros(RF_array.shape)
                    #Nframes = 3 # For debugging purposes
                    for iframe in range(Nframes): # We have purposely avoided vectorization to avoid potential memory issues
                        print(custom_save_label[1:] + ': ' + str(iframe) + ' of ', str(RF_array.shape[0]))
                        RF_array_transform[iframe, :, :] = modify_function(RF_array[iframe,:,:])
                    # Store data with right label
                    # Plot for debugging purposes
                    #fig = plt.figure()
                    #ax = fig.add_subplot(111)
                    #ax.imshow(RF_array_transform[0,:,:], vmin=0, vmax=1)   
                    #ax.set_aspect('auto')        
                    #plt.show()
                    #plot_image(RF_array_transform[iframe, :,:, 15], xdata + custom_save_label + '.png',  00, 1)
                    np.save(inputnn + xdata + str(kfold)+ custom_save_label + '.npy', RF_array_transform)
                del RF_array
                del RF_array_transform

    ###***MODIFIED FOR PHILIPSDATA
    ########### 2- Generate 2D Patches for raw B-mode data, and get labels and receipt for further patch extraction ################################################
    #Data extraction variables
    skipGenerate2DPatches = True
    if (skipGenerate2DPatches == False):
        Npatches = 300
        Naxial = 392
        Nlines = 27
        #Variables
        whichdata = ['train', 'test']

        cross_validation_folds = [0, 1, 2, 3]  #, 4

        # Loop on validation folds
        for kfold in cross_validation_folds:        #Data evaluation loop

            for xdata in whichdata:
                #Open train data and segmentation mask
                labframe = np.load( (path_to_raw + xdata + str(kfold) + '_' + LABELS_NAME) ) 
                RF_mask = np.load( (path_to_raw + xdata + str(kfold) + '_' + MASK_NAME)) # same size as trainRF_array, contains segmentation masks
                RF_array = np.load( (path_to_raw + xdata + str(kfold) + '_' + DATA_NAME)) # same size as trainRF_array, contains segmentation masks
                '''  # For debugging purposes
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(RF_array[85,:,:], vmin=-200, vmax=200)
                ax.set_aspect('auto')
                plt.show()
                '''
                Nframes = RF_mask.shape[0]
                maskArea = np.zeros(Nframes)
                coverage = np.array(maskArea)
                patches2D = np.zeros([Nframes*Npatches, Nlines, Naxial, 1]) # Big array where data is stored
                labels2D = np.zeros([Nframes*Npatches, 4]) # Big array where labels for 2D patches are stored
                labels1D = np.zeros([Nframes*Npatches*Nlines, 4]) # Big array where labels for 1D patches are stored
                where_patches_receipt = np.zeros([Nframes*Npatches, 7]) # [frameID, startid0, endid0, startid1, endid1, coverage, maskArea]
                ## Loop and generate both patches2D and labels2D
                idwrite = 0   # For generating patches2d and labels2d
                idwrite1D = 0 # For generating patches1d and labels1d
                #Nframes = 10 # For debugging purposes
                for index in range(Nframes):
                    patientId = int(labframe[index,3])
                    print('Frame: ', str(index) + ' of ' + str(Nframes))
                    largest_connected_region =  find_largest_connected_region(RF_mask[index,:,:])
                    mask_patch, coverage[index], where_patches = mask_patch_extraction(RF_array[index,:,:], largest_connected_region, Naxial+1, Nlines, Npatches) # mask_patch (393, 27, 15) <> +1 in axial coordinate to have odd number
                    if coverage[index] == 0: # Bug correction for some wrongly segmented masks
                        RF_mask[index, 100:4900, 0] = 1
                        RF_mask[index, 100:4900, -1] = 1
                        largest_connected_region =  find_largest_connected_region(RF_mask[index,:,:])
                        from scipy.ndimage.morphology import binary_fill_holes
                        largest_connected_region = binary_fill_holes(largest_connected_region)
                        mask_patch, coverage[index], where_patches = mask_patch_extraction(RF_array[index,:,:], largest_connected_region, Naxial+1, Nlines, Npatches) # mask_patch (393, 27, 15) <> +1 in axial coordinate to have odd number
                    if coverage[index] > 0:
                        maskArea[index] = np.sum(largest_connected_region)
                        mask_patch2 = np.transpose(mask_patch[0:Naxial,:,:], [2,1,0])
                        patches2D[idwrite:idwrite+Npatches, :, :, 0] = mask_patch2[:,:,:]
                        labels2D[idwrite:idwrite+Npatches, 0] = labframe[index,1]
                        labels2D[idwrite:idwrite+Npatches, 1] = labframe[index,0]
                        labels2D[idwrite:idwrite+Npatches, 2] = labframe[index,2]
                        labels2D[idwrite:idwrite+Npatches, 3] = patientId
                        labels1D[idwrite1D:idwrite1D+Npatches*Nlines, 0] = labframe[index,1]
                        labels1D[idwrite1D:idwrite1D+Npatches*Nlines, 1] = labframe[index,0]
                        labels1D[idwrite1D:idwrite1D+Npatches*Nlines, 2] = labframe[index,2]
                        labels1D[idwrite1D:idwrite1D+Npatches*Nlines, 3] = patientId
                        where_patches_receipt[idwrite:idwrite+Npatches, 0] = index
                        where_patches_receipt[idwrite:idwrite+Npatches, 1:5] = where_patches[:,:]
                        where_patches_receipt[idwrite:idwrite+Npatches, 5] = coverage[index]
                        where_patches_receipt[idwrite:idwrite+Npatches, 6] = maskArea[index]
                        idwrite = idwrite + Npatches
                        idwrite1D = idwrite1D + Npatches*Nlines
                    else:
                        print("Insufficient coverage")
                Nusableframes = idwrite/Npatches
                ## Storage functions
                print('Usable frames: ' + str(Nusableframes) + ' of ' + str(Nframes))
                print('Storing 2D patch data...')
                np.save(inputnn + xdata + str(kfold) + 'RF_patches_' +   'array.npy', patches2D[0:idwrite,:,:,:])
                np.save(inputnn + xdata + str(kfold) +  '_patches_' +  '_labels2D.npy', labels2D[0:idwrite,:])
                np.save(inputnn + xdata + str(kfold) +  '_patches_' +  '_labels1D.npy', labels1D[0:idwrite1D,:])
                np.save(inputnn + xdata + str(kfold) +  '_patches_' +  '_where_patches_receipt.npy', where_patches_receipt[0:idwrite,:])
                ## Clear unused large variables RF_mask, RF_array
                del RF_mask
                del RF_array
                del patches2D
                del labels2D
                del labels1D
        # For debugging
        '''
        mpatches2D = patches2D[0:idwrite, :,:,:]
        mpatches2D2 = np.transpose(mpatches2D, [0, 2, 1, 3]) # (2490x 392x 27x1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mpatches2D2[:,:,13,0], vmin=-30, vmax=30)
        ax.set_aspect('auto')
        plt.show()
        '''
    ########### 3 - Repeat patch generation for stages RF vs Bmode ################################################
    # we loop for datasets we want to extract (train, test)
    # we open patch coordinates
    # we loop for datasets we want to edit (_den, _env, _ps, ...)
    # we add modifiers of dimensions when necessary
    # we extract patches and store.
    # DONE
    # Data extraction 
    skipGenerate2DPatchesFromRFvsBmode = False
    if (skipGenerate2DPatchesFromRFvsBmode == False):
        Npatches = 300
        Nlines = 27
        Naxial = 392
        Naxialps = 14
        Nfreqps = 28
        Noverlaps = 26
        RFLineSize = 5000
        #Variables
        whichdata = ['train', 'test']
        #list_labels_modify = ["_env", "_log", "_den",   "_iq",  "_ps", "_pslog", "_pslognorm", "_pslognormden", "_pslogfirst", "_phasefun"]
        list_labels_modify = ["_phasefun", "_pslogfirst"]
        #Data evaluation loop

        cross_validation_folds = [4]  #, 4

        # Loop on validation folds
        for kfold in cross_validation_folds:        #Data evaluation loop
            for xdata in whichdata:
                #Open data and patch extraction receipt
                where_patches_receipt = np.load((inputnn + xdata + str(kfold) + '_patches__where_patches_receipt.npy'))
                Nframes = int(where_patches_receipt.shape[0]/Npatches) # How many frames we need to deal with
                for xmodifier in list_labels_modify:
                    if (xmodifier[1:3] == 'ps'):
                        patches2D = np.zeros([Nframes*Npatches, Nlines, Naxialps, Nfreqps]) # Big array where data is stored
                    else:
                        patches2D = np.zeros([Nframes*Npatches, Nlines, Naxial, 1]) # Big array where data is stored
                    # Open data
                    data = np.load( (inputnn + xdata +  str(kfold)  + xmodifier + '.npy')) # same size as trainRF_array, contains segmentation masks    
                    idwrite = 0   # For generating patches2d and labels2d
                    for indexF in range(Nframes):
                        whichFrame = int(where_patches_receipt[idwrite, 0])
                        print(xmodifier + ' -> Frame: ', str(indexF) + ' of ' + str(Nframes) + ': ' + str(whichFrame))
                        ### Start extracting patches from dataset
                        if xmodifier[1:3] == 'ps': # For _ps and _psden
                            where_coords = np.array(where_patches_receipt[idwrite:idwrite+Npatches,1:5])
                            where_coords[:,0] = np.floor(where_coords[:,0]/Noverlaps) 
                            where_coords[:,1] = where_coords[:,0] + Naxialps
                            mask_patch = mask_patch_extraction_from_random_patches_ids_multichannel(data[whichFrame,:,:,:], where_coords, Npatches)
                            mask_patch2 = np.transpose(mask_patch[:,:,0:Nfreqps,:], [3, 1,0,2])
                            patches2D[idwrite:idwrite+Npatches, :,:, :] = mask_patch2[:,:,:,:]
                            #data.shape = (170,111, 192, 33)  <-> (2928, 192) with Noverlap= 26
                            #patches2D.shape = (2490, 27, 14, 28)
                        elif  xmodifier[1:3] == 'iq': # For_iq
                            where_coords = np.array(where_patches_receipt[idwrite:idwrite+Npatches,1:5])
                            where_coords_iq1 = np.array(where_coords)
                            where_coords_iq2 = np.array(where_coords)
                            where_coords_iq1[:,0] = np.floor(where_coords[:, 0]/2)
                            where_coords_iq1[:,1] = where_coords_iq1[:,0] + Naxial/2 
                            where_coords_iq2[:,0] = np.floor(np.floor(where_coords[:, 0]/2) + RFLineSize/2)
                            where_coords_iq2[:,1] = where_coords_iq2[:,0] + Naxial/2 
                            mask_patch_iq1 = mask_patch_extraction_from_random_patches_ids(data[whichFrame,:,:], where_coords_iq1, Npatches)
                            mask_patch_iq2 = mask_patch_extraction_from_random_patches_ids(data[whichFrame,:,:], where_coords_iq2, Npatches)
                            mask_patch = np.concatenate((mask_patch_iq1, mask_patch_iq2), axis=0)
                            mask_patch2 = np.transpose(mask_patch[0:Naxial,:,:], [2,1,0])
                            patches2D[idwrite:idwrite+Npatches, :,:, 0] = mask_patch2[:,:,:]
                        else: # For _env, _log, _den
                            mask_patch = mask_patch_extraction_from_random_patches_ids(data[whichFrame,:,:], where_patches_receipt[idwrite:idwrite+Npatches,1:5], Npatches)
                            mask_patch2 = np.transpose(mask_patch[0:Naxial,:,:], [2,1,0])
                            patches2D[idwrite:idwrite+Npatches, :,:, 0] = mask_patch2[:,:,:]
                        idwrite = idwrite + Npatches
                    ##### Plots for debugging purposes
                    #fig = plt.figure()
                    #ax = fig.add_subplot(111)
                    #ax.imshow(patches2D[1245,:,:,11], vmin=0, vmax=1)
                    #plt.show()
                    print('Storing 2D patch data...')
                    np.save(inputnn + xdata + str(kfold) + 'RF_patches_' + 'array' + xmodifier + '.npy', patches2D[0:idwrite,:,:,:])
    SystemExit()
    # Sergio: This part is to create new data at different points of the RF to B-mode processing chain
    # and store in the local instance. The function does not have continuity and a reboot of the code with 
    # load_existing_dataset = True is necessary to arrive to training and DL results.


# ----------------------------------------------------------------------------------------------------------------
########### 4 - Initialize and train neural network ##############################################
# We loop on NN models and data we want to train with
# We train network
# We generate and store validation data

# loading data for training , you will only reach here if load_existing_dataset = True
if load_existing_dataset == False:
    SystemExit()

# provide number of cores available for use
#print("Available CPU cores: " + str(num_cores))

MODIFIER = '_PHILIPSBMODE'
MODIFIER_STORE = '_PHILIPSBMODE_RF'

inputnn = "../inputnn" + MODIFIER + "/"
outputnnmodel = "../outputnnmodel" + MODIFIER_STORE + "/"
outputnnresults = "../outputnnresults" + MODIFIER_STORE + "/"
numClasses = 4



# We open model library
# ------------------------------------------------------------------------------------------------------------------------------------
exec(open("0_main_modellibrary_rfvsbmode_OMAR.py").read())

### List of datasets/models to test systematically
### 1D neural network architecture
### patch1D->rf   (Ninstances, 392, 1) (**needs reshape)
### patch1D->iq (**needs reshape)
### patch1D->env (**needs reshape)
### patch1D->den (**needs reshape)
### patch1D->ps  (**needs reshape)
### patch1D->ps_den (** needs reshape)
### 2D neural network architecture
### patch2D->rf   (Ninstances, 27, 392, 1)
### patch2D->iq
### patch2D->env
### patch2D->den
### patch2D->ps  (**needs reshape)
### patch2D->ps_den (** needs reshape)
### 3D neural network architecture (Ninstances, 27,14,28)
### patch3D->ps
### patch3D->ps_den
### 1D neural network architecture

### Optimization of hyperparameters
### patch1D->rf   (Ninstances, 392, 1) (**needs reshape)
### patch2D->rf   (Ninstances, 27, 392, 1)
### patch3D->ps   (Ninstances, 27,14,28)

#mymodel1D = model_1D_210411_01()
#mymodel2D = model_2D_210411_02()
#mymodel3D = model_3D_210411_03()
# We pass here the data that should be tested and the corresponding models to evaluate, check folders for exact names.



dict_train_settings1D = {
    "numEpochs": 100,
    "batchSize": 30,
    "learning_rate": 0.01,
}
dict_train_settings2D = {
    "numEpochs": 100,
    "batchSize": 30,
    "learning_rate": 0.01,
}
'''
dict_train_settings2Dbis = {
    "numEpochs": 100,
    "batchSize": 30,
    "learning_rate": 0.01,
}
'''

dict_train_settings3D = {
    "numEpochs": 100,
    "batchSize": 30,
    "learning_rate": 0.01,
}

label_multiclass = '_4L'

list_labels_training = [
                     #   ["array", "_labels2D", stages_dont_do_anything, model_2D_210411_06_4classes, "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],
                     #   ["array_pslog", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim_4classes, "Run210425_model_3D_210411_07bis" + label_multiclass, dict_train_settings3D],
                     #   ["array_pslog", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06_4classes, "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],
                     #   ["array_iq", "_labels2D", stages_dont_do_anything, model_2D_210411_06_4classes, "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],
                     #   ["array_env", "_labels2D", stages_dont_do_anything, model_2D_210411_06b_4classes, "Run210425_model_2D_210411_06bis" + label_multiclass, dict_train_settings2D],
                    #    ["array_log", "_labels2D", stages_dont_do_anything, model_2D_210411_06b_4classes, "Run210425_model_2D_210411_06bis" + label_multiclass, dict_train_settings2D],
                     #   ["array_den", "_labels2D", stages_dont_do_anything, model_2D_210411_06b_4classes, "Run210425_model_2D_210411_06bis" + label_multiclass, dict_train_settings2D],
                        ["array_phasefun", "_labels2D", stages_dont_do_anything, model_2D_210411_06b_4classes, "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],
                     #   ["array_ps", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim_4classes, "Run210425_model_3D_210411_07bis" + label_multiclass, dict_train_settings3D],
                     #   ["array_ps", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06_4classes, "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],
                     #   ["array_pslognorm", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim_4classes, "Run210425_model_3D_210411_07bis" + label_multiclass, dict_train_settings3D],
                     #   ["array_pslognorm", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06_4classes,  "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],                        
                     #   ["array_pslognormden", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim_4classes, "Run210425_model_3D_210411_07bis" + label_multiclass, dict_train_settings3D],
                     #   ["array_pslognormden", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06_4classes,  "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],   
                        ["array_pslogfirst", "_labels2D",  stages_reshape3D_2Dpatch, model_2D_210411_06_4classes,  "Run210425_model_2D_210411_06" + label_multiclass, dict_train_settings2D],                     
                        ["array_pslogfirst", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim_4classes, "Run210425_model_3D_210411_07bis" + label_multiclass, dict_train_settings3D],
                      #  ["array", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08_4classes, "Run210425_model_1D_210411_08" + label_multiclass, dict_train_settings1D], 
                      #  ["array_iq", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08_4classes, "Run210425_model_1D_210411_08" + label_multiclass, dict_train_settings1D],
                      #  ["array_env", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08b_4classes, "Run210425_model_1D_210411_08b" + label_multiclass, dict_train_settings1D],
                      #  ["array_log", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08b_4classes, "Run210425_model_1D_210411_08b" + label_multiclass, dict_train_settings1D],
                      #  ["array_den", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08b_4classes, "Run210425_model_1D_210411_08b" + label_multiclass, dict_train_settings1D],
                      #  ["array_phasefun", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08_4classes, "Run210425_model_1D_210411_08" + label_multiclass, dict_train_settings1D],
                        ]
'''
list_labels_training = [
                  #     ["array", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08, "Run210425_model_1D_210411_08", dict_train_settings1D], 
                        ["array", "_labels2D", stages_dont_do_anything, model_2D_210411_06_4classes, "Run210425_model_2D_210411_06_4class", dict_train_settings2D],
                  #     ["array_iq", "_labels2D", stages_dont_do_anything, model_2D_210411_06, "Run210425_model_2D_210411_06", dict_train_settings2D],
                  #      ["array_env", "_labels2D", stages_dont_do_anything, model_2D_210411_06b, "Run210425_model_2D_210411_06bis", dict_train_settings2D],
                  #      ["array_log", "_labels2D", stages_dont_do_anything, model_2D_210411_06b, "Run210425_model_2D_210411_06bis", dict_train_settings2D],
                  #     ["array_den", "_labels2D", stages_dont_do_anything, model_2D_210411_06b, "Run210425_model_2D_210411_06bis", dict_train_settings2D],
                #       ["array_ps", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim, "Run210425_model_3D_210411_07bis", dict_train_settings3D],
                #         ["array_ps", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06, "Run210425_model_2D_210411_06", dict_train_settings2D],
                #        ["array_psden", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim, "Run210425_model_3D_210411_07bis", dict_train_settings3D],
                #       ["array_psden", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06,  "Run210425_model_2D_210411_06", dict_train_settings2D],                        

                  #      ["array_ps", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim, "Run210425_model_3D_210411_07bis", dict_train_settings3D],
                  #      ["array_ps", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06, "Run210425_model_2D_210411_06", dict_train_settings2D],
                       ["array_pslog", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim_4classes, "Run210425_model_3D_210411_07bis_4class", dict_train_settings3D],
                        ["array_pslog", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06_4classes, "Run210425_model_2D_210411_06_4class", dict_train_settings2D],
                    #    ["array_pslognorm", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim, "Run210425_model_3D_210411_07bis", dict_train_settings3D],
                    ##   ["array_pslognorm", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_06,  "Run210425_model_2D_210411_06", dict_train_settings2D],                        
                    #    ["array_pslognormden", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim, "Run210425_model_3D_210411_07bis", dict_train_settings3D],
                ]
'''


# ------------------------------------------------------------------------------------------------------------------------------------
# hide pandas warning
pd.set_option('mode.chained_assignment', None)
# the default is:
# pd.set_option('mode.chained_assignment', 'warn')
# ------------------------------------------------------------------------------------------------------------------------------------
# testing settings
# %run -i 0_settings_analysis.py
# ONLY RELEVANT FOR BINARY CLASSIFICATION: threshold for the average label of the training instances for a single case in order for the case to be labelled '1' rather than '0'
# 0.5 is a reasonable value for general use
threshold = 0.5
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# choose which sets to use for making predictions and then analyzing
# it's good to use both train and val - this will allow you to make comparisons, understand existing overfitting, etc.
do_pred_train = True
do_pred_val = True
# the reserved test set should be left alone for the most part, and only be used to test promising models
do_pred_test = False
# ------------------------------------------------------------------------------------------------------------------------------------

skipTraining = False

if skipTraining == False:


    cross_validation_folds = [4]  #[0,1,2]  #, 4



    # As a result we should have a folder for each data/model containing trained models, and accuracy/loss plots
    for x in list_labels_training:

        # Loop on validation folds
        for kfold in cross_validation_folds:        #Data evaluation loop

            print("*********************")
            print("KFOLD " + str(kfold))

            # We open data
            dataset_name = x[0]
            labels_name = x[1]
            modify_fun = x[2]
            model_fun = x[3]
            model_label = x[4]
            print(dataset_name + ':' + model_label)
            dict_train_settings = x[5]
            destination_model =  outputnnmodel + model_label + '__' + dataset_name + "_kfold" + str(kfold)
            destination_results =  outputnnresults + model_label + '__' + dataset_name  + "_kfold" + str(kfold)
            #################################### LOAD DATA FOR TRAINING #############################################################
            # Get training dataset, get validation dataset  : train_data, val_data
            train_val_RAM = True
            train_data = np.load(inputnn + "train" + str(kfold) + "RF_patches_" + dataset_name + '.npy') # Already prepared for training
            train_data = modify_fun(train_data)  # Perform reshaping if necessary - check memory concerns
            val_data = np.load(inputnn + "test" + str(kfold) + "RF_patches_" + dataset_name + '.npy')
            val_data = modify_fun(val_data)  # Perform reshaping if necessary - check memory concerns

            '''
            # Reshape for pre-trained neural network - We are going to blow out memory - We should have batch loading here
            NN_SIZE = 299  
            NN_CHANNELS = 3
            train_data_nn = np.zeros((train_data.shape[0], NN_SIZE, NN_SIZE, NN_CHANNELS))
            for indexnn in range((int)(NN_SIZE/train_data.shape[1])):
                for indexc in range(NN_CHANNELS):
                    train_data_nn[:, indexnn*train_data.shape[1]:indexnn*train_data.shape[1]+train_data.shape[1], :, indexc] = train_data[:, :, 0:NN_SIZE, 0]
            val_data_nn = np.zeros((val_data.shape[0], NN_SIZE, NN_SIZE, NN_CHANNELS))
            for indexnn in range((int)(NN_SIZE/val_data.shape[1])):
                for indexc in range(NN_CHANNELS):
                    val_data_nn[:, indexnn*val_data.shape[1]:indexnn*val_data.shape[1]+val_data.shape[1], :, indexc] = val_data[:, :, 0:NN_SIZE, 0]
            train_data = train_data_nn
            val_data = val_data_nn
            '''

            # Get labels
            labels_train = np.load(inputnn + "train" + str(kfold) + "_patches_" + labels_name + '.npy')
            labels_val = np.load(inputnn + "test" + str(kfold) + "_patches_" + labels_name + '.npy')
            ## Here we could include modifier functions for regression or other label classification...
            if (numClasses == 2):

                ## Configuration 1: S0 vs S1-S2-S3
                labels_train2 = labels_train[:,0]
                labels_train2[labels_train2 > 0] = 1 ## Here we introduce the two-class classificator
                labels_train[:,0] = labels_train2[:]
                labels_train_1hot = one_hot_encode(numClasses, labels_train2)
                labels_val2 = labels_val[:,0]
                labels_val2[labels_val2 > 0] = 1
                labels_val[:,0] = labels_val2[:]
                labels_val_1hot = one_hot_encode(numClasses, labels_val2)

                '''
                ## Configuration 1: F0-F1 vs F2-F3-F4
                labels_train2 = labels_train[:,0]
                labels_train2[labels_train2 > 1] = 1 ## Here we introduce the two-class classificator
                labels_train[:,0] = labels_train2[:]
                labels_train_1hot = one_hot_encode(numClasses, labels_train2)
                labels_val2 = labels_val[:,0]
                labels_val2[labels_val2 > 1] = 1
                labels_val[:,0] = labels_val2[:]
                labels_val_1hot = one_hot_encode(numClasses, labels_val2)
                '''
            
                '''
                ## Configuration 2: F0-F1 vs F4
                labels_train2 = labels_train[:,0]
                where_labels0 = np.nonzero(labels_train2 == 0)[0]
                where_labels1 = np.nonzero(labels_train2 == 3)[0]
                where_usable_labels = np.concatenate((where_labels0, where_labels1), axis=0)
                labels_train2 = labels_train2[where_usable_labels]
                labels_train2[labels_train2 > 1] = 1 
                labels_train = labels_train[where_usable_labels, :]
                labels_train[:,0] = labels_train2[:]
                train_data = train_data[where_usable_labels]
                labels_train_1hot = one_hot_encode(numClasses, labels_train2)

                labels_val2 = labels_val[:,0]
                where_labels0 = np.nonzero(labels_val2 == 0)[0]
                where_labels1 = np.nonzero(labels_val2 == 3)[0]
                where_usable_labels = np.concatenate((where_labels0, where_labels1), axis=0)
                labels_val2 = labels_val2[where_usable_labels]
                labels_val2[labels_val2 > 1] = 1 
                labels_val = labels_val[where_usable_labels, :]
                labels_val[:,0] = labels_val2[:]
                val_data = val_data[where_usable_labels]
                labels_val_1hot = one_hot_encode(numClasses, labels_val2)
                
                '''
                '''
                ### Configuration 3: Steatosis classification
                ### Steatosis thresholds
                ### S0 < 238
                ### S1 238-260
                ### S2 260-290
                ### S3 >290
                CUTOFF = 238
                labels_train2 = labels_train[:, 2]
                ### Cut-offs
                steatosis_labels = np.array(labels_train2)
                steatosis_labels[:] = -1
                steatosis_labels[labels_train2 >= CUTOFF] = 1
                steatosis_labels[labels_train2 <= CUTOFF] = 0
                labels_train2 = steatosis_labels
                where_usable_labels = np.nonzero(labels_train2 != -1)[0]
                labels_train2 = labels_train2[where_usable_labels]
                labels_train = labels_train[where_usable_labels, :]
                labels_train[:,0] = labels_train2[:]
                train_data = train_data[where_usable_labels]
                labels_train_1hot = one_hot_encode(numClasses, labels_train2)

                labels_val2 = labels_val[:,2]
                steatosis_labels = np.array(labels_val2)
                steatosis_labels[:] = -1
                steatosis_labels[labels_val2 >= CUTOFF] = 1
                steatosis_labels[labels_val2 <= CUTOFF] = 0
                labels_val2 = steatosis_labels
                where_usable_labels = np.nonzero(labels_val2 != -1)[0]
                labels_val2 = labels_val2[where_usable_labels]
                labels_val = labels_val[where_usable_labels, :]
                labels_val[:,0] = labels_val2[:]
                val_data = val_data[where_usable_labels]
                labels_val_1hot = one_hot_encode(numClasses, labels_val2)    
                '''



            else: # 4 classes
                labels_train2 = labels_train[:,0]
                labels_val2 = labels_val[:,0]
                labels_train_1hot = one_hot_encode(numClasses, labels_train[:,0])
                labels_val_1hot = one_hot_encode(numClasses, labels_val[:,0])
            ########## HERE WE COULD ADD DATA SHUFFLING ####### Optionally, we could do shuffling of both labels and training set
            val_tuple = ( val_data , labels_val_1hot )
            gc.collect()
            print('-Load data for training done------------------')
            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Relevant variables are x=train_data, y=labels_train_1hot, val_tuple
            model = model_fun()

            ######## Settings for training ###############################################################################################################
            # settings for training
            # We move this variables outside
            numEpochs = dict_train_settings['numEpochs']        
            #numEpochs = 5
            # number of instances per batch
            batchSize = dict_train_settings['batchSize']
            # set the initial learning rate. If no learning rate scheduler is made then this will be the constant learning rate
            learning_rate = dict_train_settings['learning_rate']
            # use weights to balance the penalty for misclassifications according to the relative amounts of data for the different classes
            use_class_weights = True
            # learning rate and learning rate scheduler
            use_scheduler = True
            if use_scheduler:
                # choose (or insert) your scheduler here. you can use tfk schedulers or custom ones
                initial_learning_rate = learning_rate            
                # exponential
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate , decay_steps=100000 , decay_rate=0.96 , staircase=True )
                # inverse time
                #lr_schedule = keras.optimizers.schedules.InverseTimeDecay( initial_learning_rate , decay_steps=1.0 , decay_rate=0.5 )
                # piecewise constant
                #boundaries = [100000, 110000]; values = [1.0, 0.5, 0.1]
                #lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay( boundaries , values )
                # polynomial
                #end_learning_rate = 0.00001
                #decay_steps = 10000
                #lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay( initial_learning_rate , decay_steps , end_learning_rate , power=0.5 )
                # custom schedulers    
                #def exp_decay(epoch):
                #    initial_lrate = 0.1
                #    k = 0.1
                #    lrate = initial_lrate * exp(-k*t)
                #    return lrate
                #   lr_schedule = LearningRateScheduler(exp_decay)
                #def scheduler(epoch):
                #    if epoch < 10:
                #        return 0.001
                #    else:
                #        return 0.001 * tf.math.exp(0.1 * (10 - epoch))
                #lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
            else:
                # constant lr
                lr_schedule = learning_rate

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # select an optimizer
            #if (what_is_my_aim == 1) || (what_is_my_aim == 4):
            my_optimizer = tf.keras.optimizers.Adadelta( learning_rate=lr_schedule, rho=0.95 , epsilon=1e-07 )
            #else:
            #my_optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.01, rho = 0.9)


            #my_optimizer = tf.keras.optimizers.Adagrad( learning_rate=lr_schedule
            #                                           , initial_accumulator_value=0.1 , epsilon=1e-07 )

            #my_optimizer = tf.keras.optimizers.Adam( learning_rate=lr_schedule
            #                                        , beta_1=0.9 , beta_2=0.999 , epsilon=1e-07 , amsgrad=False )

            #my_optimizer = tf.keras.optimizers.SGD( learning_rate=lr_schedule
            #                                       , momentum=0.0 , nesterov=False )

            #my_optimizer = tf.keras.optimizers.Adamax( learning_rate=lr_schedule
            #                                          , beta_1=0.9 , beta_2=0.999 , epsilon=1e-07 )

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # additional callbacks
            # a garbage collection callback, and a model checkpoint (based on lowest loss) callback are already implemented, so don't recreate these

            # by default there are no additional callbacks. If you make more callbacks, insert them into this list
            additional_callbacks = []

            # early stopping
            #newCallback1 = tf.keras.callbacks.EarlyStopping(
            #                                    monitor='val_loss' , min_delta=0 , patience=0 , verbose=1 , mode='auto' ,
            #                                    baseline=None , restore_best_weights=False
            #                                )
            #additional_callbacks.append( newCallback1 )

            # tensorboard callback
            #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            #additional_callbacks.append( tensorboard_callback )

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            kernel_init = tfk.initializers.glorot_uniform() # Xavier uniform initiliazer
            bias_init = tfk.initializers.Constant(value=0.2)
            #model = rf_best_()
            '''
            ############## DATA AUGMENTATION - INTRODUCE GENERATORS ##################################
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            data_gen_train = ImageDataGenerator(
                rescale=1,
                rotation_range=0,
                width_shift_range=0,
                height_shift_range=0,
                horizontal_flip=True,
                #zoom_range=0.2,
                #brightness_range = [0.8,1],
                samplewise_std_normalization=False,
                #zca_whitening=True,
                #zca_epsilon=1e-02,
            )
            it_train = data_gen_train.flow(train_data, labels_train_1hot, batch_size = batchSize, shuffle = True)
        
            #if (what_is_my_aim == 1) || (what_is_my_aim == 4):
                # compile with the optimizer (and therefore learning rate schedule) chosen in settings
            
            else:
                # regression problems
                model.compile(
                    optimizer=my_optimizer, 
                    loss= tfk.losses.mse,
                    metrics=[ 'mae' ], 
                    loss_weights=None, 
                    sample_weight_mode=None, 
                    weighted_metrics=None, 
                    target_tensors=None)       
            '''
    ########################################## TRAINING LOOP ######################################
            if (numClasses == 2):
                model.compile(
                    optimizer=my_optimizer, 
                    loss= tfk.losses.binary_crossentropy,
                    metrics=[ 'accuracy' ], 
                    loss_weights=None, 
                    sample_weight_mode=None, 
                    weighted_metrics=None,
                    target_tensors=None)
            else:
                model.compile(
                    optimizer=my_optimizer, 
                    loss= tfk.losses.categorical_crossentropy,
                    metrics=[ 'accuracy' ], 
                    loss_weights=None, 
                    sample_weight_mode=None, 
                    weighted_metrics=None,
                    target_tensors=None)

            # main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
            # this script handles training a model on the prepared data

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # set callbacks

            # create a callback to save checkpoints
            filepath = outputnnmodel + '/model_weights_checkpoint.hdf5'
            # accuracy or loss option for checkpoints
            #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            ## to retrieve -> model.load_weights(checkpoint_filepath)

            # create a callback to clear memory every epoch, since model.fit(...) uses an insane amount of RAM
            class GarbageCollector(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    gc.collect()

            # early stopping
            from keras.callbacks import EarlyStopping
            es = EarlyStopping(monitor='val_loss', mode='auto', verbose = 1, patience = 5)

            callbacks_list = [ checkpoint , GarbageCollector() , es] + additional_callbacks

            # class weights
            weights = None
            if use_class_weights:
                weights = get_class_weights( labels_train_1hot )




            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # if data has been loaded into RAM
            
            history = model.fit(
                                x=train_data,
                                y=labels_train_1hot,
                                epochs=numEpochs,
                                batch_size = batchSize,
                                verbose=1,
                
                                callbacks=callbacks_list, 
                
                                validation_data=val_tuple,
                                shuffle=True,
                                #workers=1,
                                #use_multiprocessing=False,
                                class_weight = weights
                            )
            '''
            history = model.fit_generator(
                                it_train,
                                steps_per_epoch = int(np.ceil(it_train.n/float(batchSize))),
                                epochs = numEpochs,
                                verbose=1,
                                callbacks=callbacks_list, 
                                validation_data=val_tuple,
                                class_weight = weights
                            )
            '''

            # Save the entire model as a SavedModel.

            if os.path.exists(outputnnmodel + model_label + '__' + dataset_name + '_kfold' + str(kfold)):
                shutil.rmtree(outputnnmodel + model_label + '__' + dataset_name + '_kfold' + str(kfold))

            os.mkdir( outputnnmodel+ model_label + '__' + dataset_name+ '_kfold' + str(kfold))
            model.save( (outputnnmodel +model_label  + '__' + dataset_name + '/model') )
            gc.collect()
            if os.path.exists(outputnnresults  + model_label + '__' + dataset_name+ '_kfold' + str(kfold)):
                shutil.rmtree(outputnnresults  + model_label + '__' + dataset_name+ '_kfold' + str(kfold))
            os.mkdir(outputnnresults  + model_label + '__' + dataset_name+ '_kfold' + str(kfold))

            ## code to retrieve model once more
            ## model = tf.keras.models.load_model(outputnnmodel+ model_label + '__' + dataset_name + '/model')
        

    ##### DATA ANALYSIS AND EVALUATION LOOP ##################################################################################################
            # main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
            # this script handles testing a model
            # datetime(year, month, day, hour, minute, second, microsecond)
            #right_now = datetime.datetime.now()
            #date_s = ( '{}_{}_{}_{}_{}'.format(right_now.year,right_now.month,right_now.day,right_now.hour,right_now.minute) )
            #date_s
            # name model
            # model_name = data_type + '_model_' + date_s
            # where to save model is available in destination_results

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # LEARNING CURVES from training
            is_regression = 0
            if is_regression == 0:
                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                with open(destination_results + "/" "Chistory.npy", 'wb') as f:
                    np.save(f, acc)
                    np.save(f, val_acc)
                    np.save(f, loss)
                    np.save(f, val_loss)
                plot_learning( acc , val_acc , loss , val_loss ,  destination_results)
            else:
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                mae = history.history['mae']
                val_mae= history.history['val_mae']
                r2 = history.history['r2']
                val_r2 = history.history['r2']
                with open(destination_results + "_Rhistory.npy", 'wb') as f:
                    np.save(f, loss)
                    np.save(f, val_losss)
                    np.save(f, mae)
                    np.save(f, val_mae)
                    np.save(f, r2)
                    np.save(f, val_r2)
                plot_learning_regression( mae , val_mae , loss , val_loss , r2, val_r2, destination_results)
            gc.collect()
            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # set generator for predictions using sequential loading
            # prediction parameters
            # batch sizes for testing do not need tuning, they are only chosen for convenience and speed
            if instance_rate == 1:
                test_batch = num_cores*15
            else:
                test_batch = instance_rate * num_cores * 5
            pred_params = { 
                            'batch_size': test_batch,
                            'dim': data_dim,
                            'n_channels': data_channels,
                            'n_classes': numClasses,
                            'shuffle': False,
                            'n_cores': num_cores,
                            'modify_function': modify_function,
                            'n_instances': instance_rate
                            }

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # train set

            if do_pred_train:
                
                print('\n\nTRAIN SET ANALYSIS----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                if train_val_RAM:
                    # in RAM
                    pred_train = model.predict(
                                            train_data,
                                            batch_size = batchSize,
                                            verbose=1
                                            )
                else:
                    # sequential loading
                    training_pred_generator = DataGenerator(IDs_train, labelDict_train, **pred_params)
                    pred_train = model.predict( training_pred_generator , verbose=1 , workers=num_cores , use_multiprocessing=True )
                
                gc.collect()
                
                # analyze
                
                # for debugging
                #labels_train_1hot[0] = [0, 1]; labels_train[0,0] = 1

                # instance level
                print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
                train_real_labels_i, train_test_maxfrac_i, train_test_labels_i, train_metrics_i = results_instance( pred_train , labels_train_1hot , destination_results, '_train_')
                np.save(destination_results + "/train_real_labels_i.npy", np.array(train_real_labels_i))
                np.save(destination_results + "/train_test_labels_i.npy", np.array(train_test_labels_i))
                # We should also save train_metrics_i (pickle?)

                # case level
                print('\n\\\\\\\\\\ CASE LEVEL /////')
                train_real_labels_c, train_test_case_df, train_test_labels_c, train_metrics_c = results_case( pred_train , labels_train ,destination_results, '_train_', threshold )
                np.save(destination_results + "/train_real_labels_c.npy", np.array(train_real_labels_c))
                np.save(destination_results + "/train_test_labels_c.npy", np.array(train_test_labels_c))
                # We should also save train_metrics_c, train_test_case_df (pickle?)

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # validation set

            if do_pred_val:
                
                print('\n\nVALIDATION SET ANALYSIS----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                if train_val_RAM:
                    # in RAM
                    pred_val = model.predict(
                                            val_tuple[0],
                                            batch_size = batchSize,
                                            verbose=1
                                            )
                else:
                    # sequential loading
                    validation_pred_generator = DataGenerator(IDs_val, labelDict_val, **pred_params)
                    pred_val = model.predict( validation_pred_generator , verbose=1 , workers=numCores , use_multiprocessing=True )
                
                gc.collect()
                
                # analyze
                
                # for debugging
                #labels_val_1hot[0] = [0, 1]; labels_val[0,0] = 1

                # instance level
                print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
                val_real_labels_i, val_test_maxfrac_i, val_test_labels_i, val_metrics_i = results_instance( pred_val ,labels_val_1hot , destination_results, '_val_')
                np.save(destination_results + "/val_real_labels_i.npy", np.array(val_real_labels_i))
                np.save(destination_results + "/val_test_labels_i.npy", np.array(val_test_labels_i))
                # We should also save val_metrics_i

                # case level
                print('\n\\\\\\\\\\ CASE LEVEL /////')
                val_real_labels_c, val_test_case_df, val_test_labels_c, val_metrics_c = results_case( pred_val , labels_val,destination_results, '_val_', threshold )
                np.save(destination_results + "/val_real_labels_c.npy", np.array(val_real_labels_c))
                np.save(destination_results + "/val_test_labels_c.npy", np.array(val_test_labels_c))
                # we should also ssave val_metrics_c, and val_test_case_df

            ### Until here - To be completed
            #exec(open("0_main_modelsave_rfvsbmode.py").read())
    ########################################################## SAVE MODEL ###################################################################################
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

                '''
                # datetime(year, month, day, hour, minute, second, microsecond)
                right_now = datetime.datetime.now()
                date_s = ( '{}_{}_{}_{}_{}'.format(right_now.year,right_now.month,right_now.day,right_now.hour,right_now.minute) )
                date_s

                # name model
                model_name = data_type + '_model_' + date_s
                model_name
                '''

                # define dataframe
                r_df = pd.DataFrame()

                # assign columns

                # basic data
                r_df['model name'] = [model_label + '__' + dataset_name]
                r_df['data type'] = [data_type]
                r_df['data shape'] = [data_shape]

                # labelling data
                r_df['numClasses'] = numClasses

                # All this is irrelevant here
                '''
                r_df['labelling'] = labelling
                if labelling == 1:
                    r_df['min_kpa'] = min_kpa
                elif labelling == 2:
                    r_df['h_c'] = h_c
                    r_df['f0_c'] = f0_c
                    r_df['f2_c'] = f2_c
                    r_df['f3_c'] = f3_c
                    r_df['f4_c'] = f4_c
                    r_df['c_c'] = c_c
                elif labelling == 3:
                    r_df['h_c'] = h_c
                    r_df['f0_c'] = f0_c
                    r_df['f2_c'] = f2_c
                    r_df['f3_c'] = f3_c
                    r_df['f4_c'] = f4_c
                    r_df['f_noKPA'] = f_noKPA
                    r_df['c_c'] = c_c
                    r_df['other_c'] = other_c
                '''

                # training and train/val/test split data
                #r_df['training_type'] = [train_type]
                r_df['epochs'] = [numEpochs]
                r_df['batch size'] = [batchSize]

                r_df['train size'] = [labels_train.shape[0]]
                r_df['train split'] = [np.unique(labels_train[:,3]).shape[0] ] # Number of patient cases in trainining set
                r_df['val size'] = [labels_val.shape[0]]
                r_df['val split'] = [np.unique(labels_val[:,3]).shape[0] ]

                r_df['acc'] = [acc]
                r_df['val_acc'] = [val_acc]
                r_df['loss'] = [loss]
                r_df['val_loss'] = [val_loss]

                # analysis data
                # try except is in case you didn't run analysis on those cases

                # train analysis
                try:
                    r_df['train num cases'] = [len(train_case_df)]
                    r_df['train case split'] = [np.asarray( train_case_df['Label'].value_counts() )]
                    r_df['f1_train'] = [f1_train]
                    r_df['accuracy_train'] = [accuracy_train]
                    r_df['CfMx_train'] = [CfMx_train]
                    r_df['fpr_train'] = [fpr_train]
                    r_df['tpr_train'] = [tpr_train]
                    r_df['roc_auc_train'] = [roc_auc_train]
                    r_df['f1_train_case'] = [f1_train_case]
                    r_df['accuracy_train_case'] = [accuracy_train_case]
                    r_df['CfMx_train_case'] = [CfMx_train_case]
                    r_df['fpr_train_case'] = [fpr_train_case]
                    r_df['tpr_train_case'] = [tpr_train_case]
                    r_df['roc_auc_train_case'] = [roc_auc_train_case]
                    r_df['train_case_df'] = [train_case_df]
                except:
                    r_df['train num cases'] = [np.nan]
                    r_df['train case split'] = [np.nan]
                    r_df['f1_train'] = [np.nan]
                    r_df['accuracy_train'] = [np.nan]
                    r_df['CfMx_train'] = [np.nan]
                    r_df['fpr_train'] = [np.nan]
                    r_df['tpr_train'] = [np.nan]
                    r_df['roc_auc_train'] = [np.nan]
                    r_df['f1_train_case'] = [np.nan]
                    r_df['accuracy_train_case'] = [np.nan]
                    r_df['CfMx_train_case'] = [np.nan]
                    r_df['fpr_train_case'] = [np.nan]
                    r_df['tpr_train_case'] = [np.nan]
                    r_df['roc_auc_train_case'] = [np.nan]
                    r_df['train_case_df'] = [np.nan]

                # validation analysis
                try:
                    r_df['val num cases'] = [len(val_case_df)]
                    r_df['val case split'] = [np.asarray( val_case_df['Label'].value_counts() )]
                    r_df['f1_val'] = [f1_val]
                    r_df['accuracy_val'] = [accuracy_val]
                    r_df['CfMx_val'] = [CfMx_val]
                    r_df['fpr_val'] = [fpr_val]
                    r_df['tpr_val'] = [tpr_val]
                    r_df['roc_auc_val'] = [roc_auc_val]
                    r_df['f1_val_case'] = [f1_val_case]
                    r_df['accuracy_val_case'] = [accuracy_val_case]
                    r_df['CfMx_val_case'] = [CfMx_val_case]
                    r_df['fpr_val_case'] = [fpr_val_case]
                    r_df['tpr_val_case'] = [tpr_val_case]
                    r_df['roc_auc_val_case'] = [roc_auc_val_case]
                    r_df['val_case_df'] = [val_case_df]
                except:
                    r_df['val num cases'] = [np.nan]
                    r_df['val case split'] = [np.nan]
                    r_df['f1_val'] = [np.nan]
                    r_df['accuracy_val'] = [np.nan]
                    r_df['CfMx_val'] = [np.nan]
                    r_df['fpr_val'] = [np.nan]
                    r_df['tpr_val'] = [np.nan]
                    r_df['roc_auc_val'] = [np.nan]
                    r_df['f1_val_case'] = [np.nan]
                    r_df['accuracy_val_case'] = [np.nan]
                    r_df['CfMx_val_case'] = [np.nan]
                    r_df['fpr_val_case'] = [np.nan]
                    r_df['tpr_val_case'] = [np.nan]
                    r_df['roc_auc_val_case'] = [np.nan]
                    r_df['val_case_df'] = [np.nan]

                # save metrics
                r_df.to_pickle( (destination_results + '/results.pkl') )

            gc.collect()
            #SystemExit()
            # Save history of the trained model
            # Save predictions of the model for cases and instances with respect to labels
            # For classification we save for training and val. set:
            # class_real, class_pred, prob_pred
            # For regression we save for training and val set:
            # num_real, num_pred




########################################################################
## Once we are here we can start analyzing the results and comparing different learning platforms
##########################################################################

skipMcNemar = True

inputnn = "../inputnn/"
outputnnmodel = "../outputnnmodel/"
outputnnresults = "../outputnnresults/"


if skipMcNemar == False:

    list_labels_McNemar = [
                            ["array", "_labels2D", stages_dont_do_anything, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],
                            ["array_iq", "_labels2D", stages_dont_do_anything, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],
                            ["array_env", "_labels2D", stages_dont_do_anything, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],
                            ["array_log", "_labels2D", stages_dont_do_anything, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],
                            ["array_den", "_labels2D", stages_dont_do_anything, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],
                            ["array_ps", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_03, "model_3D_210411_03", dict_train_settings3D],
                            ["array_ps", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],
                            ["array_psden", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_03, "model_3D_210411_03", dict_train_settings3D],
                            ["array_psden", "_labels2D", stages_reshape3D_2Dpatch, model_2D_210411_02, "model_2D_210411_02", dict_train_settings2D],                        
                            
                            
                            ["array", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210411_01, "model_1D_210411_01", dict_train_settings1D], 
                            ["array_iq", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210411_01, "model_1D_210411_01", dict_train_settings1D],
                            ["array_env", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210411_01, "model_1D_210411_01", dict_train_settings1D],
                            ["array_log", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210411_01, "model_1D_210411_01", dict_train_settings1D],
                            ["array_den", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210411_01, "model_1D_210411_01", dict_train_settings1D],
 
                            ]

    # McNemar method - create agreement matrices and compare.
    comparison_list = [ [0, 1], [0, 2], [0,3], [0,4], # 1D models from rf to b-mode 
                        [5,6], [5,7], [5,8], [5,9],   # 2D models from rf to b-mode
                        [0, 5], [1, 6], [2,7], [3,8], [4,9], # 2D vs 1D
                        [10, 4], [11, 4], [12, 4], [13, 4], # Different spectral representations vs B-mode (2D-based)
                        [10, 5], [11, 5], [12, 5], [13, 5]] # Different spectral representations vs rf data (2D-based)
    '''
    list_labels_McNemar = [["rf_dataset_2020_12_17_4_33_rfB", model_201512_01_rf, "model_201512_01", dict_train_settings_B], 
        ["rf_dataset_2020_12_15_0_6_iqB", model_201512_01_iq, "model_201512_01",dict_train_settings_B], 
        ["rf_dataset_2020_12_15_15_31_envB",model_201512_01_env, "model_201512_01", dict_train_settings_B], 
        ["rf_dataset_2020_12_15_19_53_logB",model_201512_01_log,"model_201512_01", dict_train_settings_B], 
        ["rf_dataset_2020_12_16_0_35_denB",model_201512_01_den, "model_201512_01",dict_train_settings_B]] # Small dataset
    comparison_list = [ [0, 1], [0, 2], [0,3], [0,4] , [1,2], [1,3], [1,4], [2,3] , [2, 4], [3,4]]
    '''

    for x in comparison_list:
        i1 = x[0]; i2 = x[1]; 
        model1  = list_labels_McNemar[i1][4] + '__' + list_labels_McNemar[i1][0]
        model2  = list_labels_McNemar[i2][4] + '__' + list_labels_McNemar[i2][0]
        val_real_labels_i_1 = np.load(outputnnresults + model1 + '/val_real_labels_i.npy')
        val_test_labels_i_1 = np.load(outputnnresults + model1 + '/val_test_labels_i.npy')
        val_real_labels_c_1 = np.load(outputnnresults + model1 + '/val_real_labels_c.npy')
        val_test_labels_c_1 = np.load(outputnnresults + model1 + '/val_test_labels_c.npy')
        val_real_labels_i_2 = np.load(outputnnresults + model2 + '/val_real_labels_i.npy')
        val_test_labels_i_2 = np.load(outputnnresults + model2 + '/val_test_labels_i.npy')
        val_real_labels_c_2 = np.load(outputnnresults + model2 + '/val_real_labels_c.npy')
        val_test_labels_c_2 = np.load(outputnnresults + model2 + '/val_test_labels_c.npy')
    
   #     performMcNemarTest (val_real_labels_i_1 , val_test_labels_i_1 , 
   #     val_real_labels_i_2 , val_test_labels_i_2, 
   #     outputnnresults + 'McNemar_i_' + model1 + '__' + model2)
        performMcNemarTest (val_real_labels_c_1 , val_test_labels_c_1 , 
        val_real_labels_c_2 , val_test_labels_c_2, 
        outputnnresults + 'McNemar_c_' + model1 + '__' + model2)

    print("That's all folks!")


########### 5 - Perform McNemar analysis ##############################################














############# Dummy code pieces #####################################
'''
### Count mask area
maskArea = np.zeros(trainRF_mask.shape[0])
coverage = np.array(maskArea)
for index in range(trainRF_mask.shape[0]):
    print(str(index))
    largest_connected_region =  find_largest_connected_region(trainRF_mask[index,:,:])
    mask_patch, coverage[index] = mask_patch_extraction(largest_connected_region, largest_connected_region, 393, 27, 15)
    if coverage[index] > 0:
        maskArea[index] = np.sum(largest_connected_region)
    #plt.imshow(trainRF_mask[index, :,:])
    #plt.title(str(index))
    #plt.pause(0.1) # Animated plot loop as in matlab
'''







