import numpy as np
import pickle
import os 
from keras.utils import to_categorical
from numpy.core.numeric import NaN
from sklearn.metrics import roc_auc_score
from sklearn import metrics# import sys, os, glob
from sklearn.utils import shuffle, class_weight

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker
import pickle

import numpy as np
#%matplotlib inline

outputnnresults = '../outputnnresults_PHILIPSBMODE_RF/'
#outputnnresults  = '../outputnnresults_PHILIPSBMODE/'
exec(open("0_settings_preprocessing_rfvsbmode_OMAR.py").read())
exec(open("0_settings_sergiorfvsbmodeliver_OMAR.py").read())

'''
label_multiclass = '_4L'



list_labels_training = [
              #            ["array_denv2p0", "_labels2D", "Run210517_model_2D_210411_06bis" + label_multiclass, "DEN_2_bis"],                     
                        ["array_carrierv3p0", "_labels2D",  "Run210517_model_2D_210411_06" + label_multiclass, "CARRIER_2"],
              #         ["array_denv2p0", "_labels2D",  "Run210517_model_2D_210411_06" + label_multiclass,"DEN_2"],                     

                        ["array", "_labels1D",  "Run210425_model_1D_210411_08" + label_multiclass, "ARRAY_1D"],#, dict_train_settings1D], 
                        ["array_iqv2p0", "_labels1D", "Run210516_model_1D_210411_08" + label_multiclass, "IQ_1D"],
                        ["array_env", "_labels1D", "Run210425_model_1D_210411_08b" + label_multiclass, "ENV_1D"],#, dict_train_settings1D],
                        ["array_log", "_labels1D",  "Run210425_model_1D_210411_08b" + label_multiclass ,"LOG_1D"],#, dict_train_settings1D],
                        ["array_den", "_labels1D",  "Run210425_model_1D_210411_08b" + label_multiclass, "DEN_1D"],#, dict_train_settings1D],

                        ["array", "_labels2D", "Run210425_model_2D_210411_06" + label_multiclass, "RF"],#, dict_train_settings2D],
                        ["array_iqv2p0", "_labels2D",  "Run210516_model_2D_210411_06" + label_multiclass, "IQ"],#, dict_train_settings2D]
                        ["array_env", "_labels2D",  "Run210425_model_2D_210411_06bis" + label_multiclass, "ENV"],#, dict_train_settings2D],
                        ["array_log", "_labels2D",  "Run210425_model_2D_210411_06bis" + label_multiclass, "LOG"],#, dict_train_settings2D],
                        ["array_den", "_labels2D",  "Run210425_model_2D_210411_06bis" + label_multiclass, "DEN"],#, dict_train_settings2D],

                        ["array_ps", "_labels2D","Run210425_model_2D_210411_06" + label_multiclass, "PS_2D"],#, dict_train_settings2D],
                        ["array_pslog", "_labels2D",  "Run210425_model_2D_210411_06" + label_multiclass, "PS_LOG_2D"],#, dict_train_settings2D],
                        ["array_pslognorm", "_labels2D", "Run210425_model_2D_210411_06" + label_multiclass, "PS_LOG_NORM_2D"],#, dict_train_settings2D],                        
                        ["array_pslognormden", "_labels2D",   "Run210425_model_2D_210411_06" + label_multiclass, "PS_LOG_NORM_DEN"],#, dict_train_settings2D],   

                        ["array_ps", "_labels2D", "Run210425_model_3D_210411_07bis" + label_multiclass, "PS_3D"],#, dict_train_settings3D],
                        ["array_pslog", "_labels2D",  "Run210425_model_3D_210411_07bis" + label_multiclass, "PS_LOG_3D"],#, dict_train_settings3D],
                        ["array_pslognorm", "_labels2D", "Run210425_model_3D_210411_07bis" + label_multiclass, "PS_LOG_NORM_3D"],#, dict_train_settings3D],
                        ["array_pslognormden", "_labels2D",  "Run210425_model_3D_210411_07bis" + label_multiclass, "PS_LOG_NORM_DEN_3D"],#, dict_train_settings3D],

                        ["array_carrierv3p0", "_labels2D",  "Run210517_model_2D_210411_06" + label_multiclass, "CARRIER_2"],
              #          ["array_carrierv2p0", "_labels2D",  "Run210516_model_2D_210411_06" + label_multiclass, "CARRIER"],#, dict_train_settings2D],
                        ["array_anglev2p0", "_labels2D", "Run210516_model_2D_210411_06" + label_multiclass, "ANGLE"],#, dict_train_settings2D],
                        ["array_psphasev2p0", "_labels2D",  "Run210516_model_2D_210411_06" + label_multiclass, "PHASE_2D"],#, dict_train_settings2D],
                        ["array_psphasev2p0", "_labels2D", "Run210516_model_3D_210411_07bis" + label_multiclass, "PHASE_3D"],#, dict_train_settings3D],
#    ["array_phasefun", "_labels2D", "Run210512_model_2D_210411_06" + label_multiclass, "PHASE_REV"], #, dict_train_settings2D], #stages_dont_do_anything, model_2D_210411_06b_4classes
                   #     ["array_iq", "_labels2D",  "Run210425_model_2D_210411_06" + label_multiclass, "IQ"],#, dict_train_settings2D],
                    #    ["array", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08_4classes, "Run210425_model_1D_210411_08" + label_multiclass, dict_train_settings1D], 
                    #    ["array_phasefun", "_labels2D", "Run210425_model_2D_210411_06" + label_multiclass, "PHASE"], #, dict_train_settings2D], #stages_dont_do_anything, model_2D_210411_06b_4classes
                     #    ["array_pslogfirst", "_labels2D",   "Run210425_model_2D_210411_06" + label_multiclass],#, dict_train_settings2D],                     
                    #    ["array_pslogfirst", "_labels2D",  "Run210425_model_3D_210411_07bis" + label_multiclass],#, dict_train_settings3D],
                   #     ["array_iq", "_labels1D",  "Run210425_model_1D_210411_08" + label_multiclass],#, dict_train_settings1D],
                #    ["array_phasefun", "_labels1D",  "Run210425_model_1D_210411_08" + label_multiclass, dict_train_settings1D],
                        ]
'''
list_labels_training = [
                      ["array", "_labels1D",  "Run210425_model_1D_210411_08", "RF_1D"], 
                        ["array_iq", "_labels1D", "Run210425_model_1D_210411_08", "IQ_1D"],
                        ["array_env", "_labels1D",  "Run210425_model_1D_210411_08b", "ENV_1D"],
                        ["array_log", "_labels1D",  "Run210425_model_1D_210411_08b", "LOG_1D"],
                        ["array_den", "_labels1D",  "Run210425_model_1D_210411_08b", "DEN_1D"],
                        ["array", "_labels2D",  "Run210425_model_2D_210411_06", "RF_2D"],
                        ["array_iqv2p0", "_labels2D", "Run210516_model_2D_210411_06", "IQ_2D"],
                        ["array_env", "_labels2D", "Run210425_model_2D_210411_06bis", "ENV_2D"],
                        ["array_log", "_labels2D",  "Run210425_model_2D_210411_06bis", "LOG_2D"],
                        ["array_den", "_labels2D",  "Run210425_model_2D_210411_06bis","DEN_2D"],
                        ["array_ps", "_labels2D",  "Run210425_model_2D_210411_06", "PS_2D"],
                        ["array_pslog", "_labels2D",  "Run210425_model_2D_210411_06", "PS_LOG_2D"],
                        ["array_pslognorm", "_labels2D",   "Run210425_model_2D_210411_06", "PS_LOG_NORM_2D"],                        
                        ["array_pslognormden", "_labels2D",   "Run210425_model_2D_210411_06", "PS_LOG_NORM_DEN_2D"],   
                        ["array_ps", "_labels2D", "Run210425_model_3D_210411_07bis", "PS_3D"],
                        ["array_pslog", "_labels2D",  "Run210425_model_3D_210411_07bis", "PS_LOG_3D"],
                        ["array_pslognorm", "_labels2D", "Run210425_model_3D_210411_07bis", "PS_LOG_NORM_3D"],
                        ["array_pslognormden", "_labels2D",  "Run210425_model_3D_210411_07bis","PS_LOG_NORM_DEN_3D"],
                        ["array_carrierv2p0", "_labels2D",  "Run210516_model_2D_210411_06" , "CARRIER_2"],
              #          ["array_carrierv2p0", "_labels2D",  "Run210516_model_2D_210411_06" + label_multiclass, "CARRIER"],#, dict_train_settings2D],
                        ["array_anglev2p0", "_labels2D", "Run210516_model_2D_210411_06" , "ANGLE"],#, dict_train_settings2D],
                        ["array_psphasev2p0", "_labels2D",  "Run210516_model_2D_210411_06" , "PHASE_2D"],#, dict_train_settings2D],
                        ["array_psphasev2p0", "_labels2D", "Run210516_model_3D_210411_07bis" , "PHASE_3D"],#, dict_train_settings3D],
                       # ["array_pslogfirst", "_labels2D",  stages_reshape3D_2Dpatch, model_2D_210411_06,  "Run210425_model_2D_210411_06", dict_train_settings2D],                     
                       # ["array_pslogfirst", "_labels2D", stages_reshape3D_4Dpatch, model_3D_210411_07_optim, "Run210425_model_3D_210411_07bis", dict_train_settings3D],
   
                        ]
                      
#list_labels_training =  [["array", "_labels2D",  "Run210430_PHILIPS_N35_bmode_model_2D_210411_06b_0p01LR_30batch" + label_multiclass, "Clinical B-mode"]] #Bmode
#list_labels_training =  [["array", "_labels2D",  "Run210430_PHILIPS_N35_bmode_model_2D_210430_4class" + label_multiclass, "Clinical B-mode"]] #Bmode
'''
list_labels_training = [
                      # ["array", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08, "Run210421_model_1D_210411_08", dict_train_settings1D], 
                      # ["array_iq", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08, "Run210421_model_1D_210411_08", dict_train_settings1D],
                      # ["array_env", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08b, "Run210421_model_1D_210411_08b", dict_train_settings1D],
                      # ["array_log", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08b, "Run210421_model_1D_210411_08b", dict_train_settings1D],
                      #  ["array_den", "_labels1D", stages_reshape2D_1Dpatch, model_1D_210418_08b, "Run210421_model_1D_210411_08b", dict_train_settings1D],
                        ["array", "_labels2D",  "Run210424_PHILIPS_N35_bmode_model_2D_210411_06b_0p01LR_30batch", "Clinical B-mode"]]
    

cross_validation_folds = [0,1,2,3,4] #[4]  #, 4
'''
cross_validation_folds = [4]

# As a result we should have a folder for each data/model containing trained models, and accuracy/loss plots

colors = ['black','blue','red', 'magenta', 'black']
for x in list_labels_training:
    train_real_labels = []
    train_test_labels = []
    train_labels_frac = []
    val_real_labels = []
    val_test_labels = []
    val_labels_frac = []
    # Loop on validation folds
    fig, axs = plt.subplots(2)
    for kfold in cross_validation_folds:        #Data evaluation loop
        print("*********************")
        print("KFOLD " + str(kfold))
        dataset_name = x[0]
        labels_name = x[1]
        model_label = x[2]
        title_label = x[3]
        
        try:
            destination_results =  outputnnresults + model_label + '__' + dataset_name  + "_kfold" + str(kfold)
            print(destination_results)
            with open(destination_results + '/results.pkl', 'rb') as f:
                data = pickle.load(f)
        except:
            destination_results =  outputnnresults + model_label + '__' + dataset_name  
            print(destination_results)
            with open(destination_results + '/results.pkl', 'rb') as f:
                data = pickle.load(f)
        val_acc = data.val_acc[0]
        train_acc = data.acc[0]
        val_loss = data.val_loss[0]
        train_loss = data.loss[0]
        axs[0].plot(val_loss, colors[kfold])
        axs[0].plot(train_loss, colors[kfold], linestyle='dashed')
        axs[0].set_title(title_label)
        axs[0].set_ylim([0.2,0.8])
        axs[0].set(ylabel = 'Loss')
        axs[0].grid()
        axs[1].plot(val_acc, colors[kfold])
        axs[1].plot(train_acc,  colors[kfold], linestyle='dashed')
        axs[1].set(ylabel = 'Acc', xlabel ='Epoch')
        axs[1].set_ylim([0.5,1])
        axs[1].grid()
       
    plt.show()
