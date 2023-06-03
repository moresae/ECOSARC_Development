# main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
# this script handles import of a dataset that has already been prepared and saved with previous use of 0_main_dataprep.py
exec(open("0_settings_preprocessing.py").read())

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORT METADATA

train_df = pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/train.pkl') )
val_df = pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/val.pkl') )
test_df = pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/test.pkl') )

info_df = pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/info.pkl') )

data_type = info_df['data type']
data_type = data_type[0]
numClasses = info_df['numClasses']
numClasses = numClasses[0]
labelling = info_df['labelling']
labelling = labelling[0]
data_shape = info_df['data shape']
data_shape = data_shape[0]
instance_rate = info_df['instance_rate']
instance_rate = instance_rate[0] # Sergio 09.12.2020
data_dim = data_shape[:len(data_shape)-1]
data_channels = data_shape[len(data_shape)-1]

modify_function = info_df['modify_function']
modify_function = modify_function[0]

print('Metadata imported.\nImporting data.')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORT DATA

train_val_RAM = True

train_data = np.load( (path_to_main+'saved_datasets/'+dataset_name + '/train.npy') )
val_data = np.load( (path_to_main+'saved_datasets/'+dataset_name + '/val.npy') )

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