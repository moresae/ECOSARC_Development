# main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
# this script handles testing a model


#----------------------------------
# Sergio: We are going to save all these outputs into files also, to be able to test multiple configurations at once.

# datetime(year, month, day, hour, minute, second, microsecond)
right_now = datetime.datetime.now()
date_s = ( '{}_{}_{}_{}_{}'.format(right_now.year,right_now.month,right_now.day,right_now.hour,right_now.minute) )
date_s
# name model
model_name = data_type + '_model_' + date_s

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LEARNING CURVES from training

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plot_learning( acc , val_acc , loss , val_loss , model_name)

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

if pred_train:
    
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
    
    # instance level
    print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
    f1_train , accuracy_train , CfMx_train , fpr_train , tpr_train , roc_auc_train = results_instance( pred_train , train_df )
    
    # case level
    print('\n\\\\\\\\\\ CASE LEVEL /////')
    train_case_df , f1_train_case , accuracy_train_case , CfMx_train_case , fpr_train_case , tpr_train_case , roc_auc_train_case = results_case( pred_train , train_df , threshold )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# validation set

if pred_val:
    
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
    
    # instance level
    print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
    f1_val , accuracy_val , CfMx_val , fpr_val , tpr_val , roc_auc_val = results_instance( pred_val , val_df )
    
    # case level
    print('\n\\\\\\\\\\ CASE LEVEL /////')
    val_case_df , f1_val_case , accuracy_val_case , CfMx_val_case , fpr_val_case , tpr_val_case , roc_auc_val_case = results_case( pred_val , val_df , threshold )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (reserved) test set

if pred_test:
    
    print('\n\nTEST SET ANALYSIS----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    
    # sequential loading
    testing_pred_generator = DataGenerator(IDs_test, labelDict_test, **pred_params)
    pred_test = model.predict( testing_pred_generator , verbose=1 , workers=numCores , use_multiprocessing=True )
    
    gc.collect()
    
    # analyze
    
    # instance level
    print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
    f1_test , accuracy_test , CfMx_test , fpr_test , tpr_test , roc_auc_test = results_instance( pred_test , test_df )
    
    # case level
    print('\n\\\\\\\\\\ CASE LEVEL /////')
    test_case_df , f1_test_case , accuracy_test_case , CfMx_test_case , fpr_test_case , tpr_test_case , roc_auc_test_case = results_case( pred_test , test_df , threshold )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------