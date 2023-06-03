# SCRIPT TO SAVE MODELS

#===============================================================================================
# import all necessary libraries

import _1_imports
from _1_imports import *

#===============================================================================================


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

# training and train/val/test split data
#r_df['training_type'] = [train_type]
r_df['epochs'] = [numEpochs]
r_df['batch size'] = [batchSize]

r_df['train size'] = [len(train_df)]
r_df['train split'] = [np.asarray( train_df['Label'].value_counts() )]
r_df['val size'] = [len(val_df)]
r_df['val split'] = [np.asarray( val_df['Label'].value_counts() )]
try:
    r_df['test size'] = [len(test_df)]
    r_df['test split'] = [np.asarray( test_df['Label'].value_counts() )]
except:
    r_df['test size'] = [np.nan]
    r_df['test split'] = [np.nan]

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
    
# test analysis
try:
    r_df['test num cases'] = [len(test_case_df)]
    r_df['test case split'] = [np.asarray( test_case_df['Label'].value_counts() )]
    r_df['f1_test'] = [f1_test]
    r_df['accuracy_test'] = [accuracy_test]
    r_df['CfMx_test'] = [CfMx_test]
    r_df['fpr_test'] = [fpr_test]
    r_df['tpr_test'] = [tpr_test]
    r_df['roc_auc_test'] = [roc_auc_test]
    r_df['f1_test_case'] = [f1_test_case]
    r_df['accuracy_test_case'] = [accuracy_test_case]
    r_df['CfMx_test_case'] = [CfMx_test_case]
    r_df['fpr_test_case'] = [fpr_test_case]
    r_df['tpr_test_case'] = [tpr_test_case]
    r_df['roc_auc_test_case'] = [roc_auc_test_case]
    r_df['test_case_df'] = [test_case_df]
except:
    r_df['test num cases'] = [np.nan]
    r_df['test case split'] = [np.nan]
    r_df['f1_test'] = [np.nan]
    r_df['accuracy_test'] = [np.nan]
    r_df['CfMx_test'] = [np.nan]
    r_df['fpr_test'] = [np.nan]
    r_df['tpr_test'] = [np.nan]
    r_df['roc_auc_test'] = [np.nan]
    r_df['f1_test_case'] = [np.nan]
    r_df['accuracy_test_case'] = [np.nan]
    r_df['CfMx_test_case'] = [np.nan]
    r_df['fpr_test_case'] = [np.nan]
    r_df['tpr_test_case'] = [np.nan]
    r_df['roc_auc_test_case'] = [np.nan]
    r_df['test_case_df'] = [np.nan]

# save metrics
r_df.to_pickle( (destination_results + '/results.pkl') )



