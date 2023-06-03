# main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
# this script handles training a model on the prepared data

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# set callbacks

# create a callback to save checkpoints
filepath = 'models/model_weights_checkpoint.hdf5'
# accuracy or loss option for checkpoints
#checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# create a callback to clear memory every epoch, since model.fit(...) uses an insane amount of RAM
class GarbageCollector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

callbacks_list = [ checkpoint , GarbageCollector() ] + additional_callbacks

# class weights
weights = None
if use_class_weights:
    weights = get_class_weights( labels_train_1hot )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# if data has been loaded into RAM

if train_val_RAM:
    
    history = model.fit(
                        x=train_data,
                        y=labels_train_1hot,
                        epochs=numEpochs,
                        batch_size = batchSize,
                        verbose=1,
        
                        callbacks=callbacks_list, 
        
                        validation_data=val_tuple,
                        shuffle=False,
                        #workers=1,
                        #use_multiprocessing=False,
                        class_weight = weights
                       )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# if not loaded into RAM, use sequential loading

else:
    
    shuffle = False
    # shuffle each batch if there are multiple instances per frame
    if instance_rate > 1:
        shuffle = True
    
    # TRAIN Parameters
    train_params = { 
                    'batch_size': batchSize,
                    'dim': data_dim,
                    'n_channels': data_channels,
                    'n_classes': numClasses,
                    'shuffle': shuffle,
                    'n_cores': num_cores,
                    'modify_function': modify_function,
                    'n_instances': instance_rate
                    }
    # VALIDATION Parameters - same except shuffling is never necessary
    val_params = { 
                    'batch_size': batchSize,
                    'dim': data_dim,
                    'n_channels': data_channels,
                    'n_classes': numClasses,
                    'shuffle': False,
                    'n_cores': num_cores,
                    'modify_function': modify_function,
                    'n_instances': instance_rate
                 }
    
    # TRAINING Generators
    training_generator = DataGenerator(IDs_train, labelDict_train, **train_params)
    validation_generator = DataGenerator(IDs_val, labelDict_val, **val_params)
    
    history = model.fit(
                        x=training_generator,
                        epochs=numEpochs,
                        verbose=1,
        
                        callbacks=callbacks_list,
        
                        validation_data=validation_generator,
                        shuffle=False,
                        #workers=1,
                        #use_multiprocessing=False,
                        class_weight = weights
                       )




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

'''