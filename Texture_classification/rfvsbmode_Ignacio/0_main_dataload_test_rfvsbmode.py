# main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
# this script handles import of a dataset that has already been prepared and saved with previous use of 0_main_dataprep.py
exec(open("0_settings_preprocessing.py").read())

dataset_names =  [[0, "rf_dataset_2020_12_13_14_17_rfS"], 
    [1, "rf_dataset_2020_12_13_13_57_iqS"], [2, "rf_dataset_2020_12_13_14_13_denS"], 
    [3, "rf_dataset_2020_12_13_14_2_envS"],
    [4, "rf_dataset_2020_12_13_14_7_logS"]]


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORT METADATA
train_df = list()
val_df = list()
test_df = list()
train_data = list()
val_data =list()

for x in dataset_names:
    index = x[0]
    dataset_name = x[1]
    train_df.append(pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/train.pkl') ))
    val_df.append(pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/val.pkl') ))
    test_df.append(pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/test.pkl') ))
    info_df.append(pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/info.pkl') ))
    print('Metadata imported.\nImporting data.')
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # IMPORT DATA
    train_val_RAM = True
    train_data.append( np.load( (path_to_main+'saved_datasets/'+dataset_name + '/train.npy') ))
    val_data.append( np.load( (path_to_main+'saved_datasets/'+dataset_name + '/val.npy') ))





print('Data imported.')
### Sergio - we could introduce code here to modify the labels (e.g., add CAP or prepare for regression), it would be necessary to have necessary data from dataframe.
# get labels, dictionaries
IDs_train , labels_train , labelDict_train = labels_dicts( train_df )
IDs_val , labels_val , labelDict_val = labels_dicts( val_df )
IDs_test , labels_test , labelDict_test = labels_dicts( test_df )
# get 1hot labels
labels_train_1hot = one_hot_encode( numClasses , labels_train )
labels_val_1hot = one_hot_encode( numClasses , labels_val )
# set validation tuple

val_tuple = ( val_data , labels_val_1hot )

# set generator for predictions on test set
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
testing_pred_generator = DataGenerator(IDs_test, labelDict_test, **pred_params)
    
gc.collect()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('----------DONE----------')