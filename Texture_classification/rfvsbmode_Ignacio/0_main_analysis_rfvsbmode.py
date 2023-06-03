# main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
# this script handles testing a model

# datetime(year, month, day, hour, minute, second, microsecond)
#right_now = datetime.datetime.now()
#date_s = ( '{}_{}_{}_{}_{}'.format(right_now.year,right_now.month,right_now.day,right_now.hour,right_now.minute) )
#date_s
# name model
#model_name = data_type + '_model_' + date_s

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
    
    # instance level
    print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
    train_real_labels_i, train_test_maxfrac_i, train_test_labels_i, train_metrics_i = results_instance( pred_train , train_df , destination_results, '_train_')
    np.save(destination_results + "/train_real_labels_i.npy", np.array(train_real_labels_i))
    np.save(destination_results + "/train_test_labels_i.npy", np.array(train_test_labels_i))

    # case level
    print('\n\\\\\\\\\\ CASE LEVEL /////')
    train_real_labels_c, train_test_case_df, train_test_labels_c, train_metrics_c = results_case( pred_train , train_df ,destination_results, '_train_', threshold )
    np.save(destination_results + "/train_real_labels_c.npy", np.array(train_real_labels_c))
    np.save(destination_results + "/train_test_labels_c.npy", np.array(train_test_labels_c))

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
      
    # instance level
    print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
    val_real_labels_i, val_test_maxfrac_i, val_test_labels_i, val_metrics_i = results_instance( pred_val , val_df , destination_results, '_val_')
    np.save(destination_results + "/val_real_labels_i.npy", np.array(val_real_labels_i))
    np.save(destination_results + "/val_test_labels_i.npy", np.array(val_test_labels_i))

    # case level
    print('\n\\\\\\\\\\ CASE LEVEL /////')
    val_real_labels_c, val_test_case_df, val_test_labels_c, val_metrics_c = results_case( pred_val , val_df ,destination_results, '_val_', threshold )
    np.save(destination_results + "/val_real_labels_c.npy", np.array(val_real_labels_c))
    np.save(destination_results + "/val_test_labels_c.npy", np.array(val_test_labels_c))


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (reserved) test set

if do_pred_test:
    
    print('\n\nTEST SET ANALYSIS----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    
    # sequential loading
    testing_pred_generator = DataGenerator(IDs_test, labelDict_test, **pred_params)
    pred_test = model.predict( testing_pred_generator , verbose=1 , workers=numCores , use_multiprocessing=True )
    
    gc.collect()
    
    # analyze
      
    # instance level
    print('\n\\\\\\\\\\ INSTANCE LEVEL /////')
    test_real_labels_i, test_test_maxfrac_i, test_test_labels_i, test_metrics_i = results_instance( pred_test , test_df , destination_results, '_test_')
    np.save(destination_results + "/test_real_labels_i.npy", np.array(test_real_labels_i))
    np.save(destination_results + "/test_test_labels_i.npy", np.array(test_test_labels_i))

    # case level
    print('\n\\\\\\\\\\ CASE LEVEL /////')
    test_real_labels_c, test_test_case_df, test_test_labels_c, test_metrics_c = results_case( pred_test , test_df ,destination_results, '_test_', threshold )
    np.save(destination_results + "/test_real_labels_c.npy", np.array(test_real_labels_c))
    np.save(destination_results + "/test_test_labels_c.npy", np.array(test_test_labels_c))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## It would be good to save pred_train, pred_val, pred_test for further analysis

