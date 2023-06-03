# settings for training

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# We move this variables outside
# number of epochs to train for
#numEpochs = 5
# number of instances per batch
#batchSize = 70
# set the initial learning rate. If no learning rate scheduler is made then this will be the constant learning rate
#learning_rate = 0.01
numEpochs = dict_train_settings['numEpochs']
batchSize = dict_train_settings['batchSize']
learning_rate = dict_train_settings['learning_rate']

# if you are using sequential loading and there are multiple instances per frame, batch size must be an integer multiple of the instance rate
if instance_rate > 1 and train_val_RAM != True:
    batchSize = instance_rate * num_cores * 5

# use weights to balance the penalty for misclassifications according to the relative amounts of data for the different classes
use_class_weights = True

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    #boundaries = [100000, 110000]
    #values = [1.0, 0.5, 0.1]
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
    #lr_schedule = LearningRateScheduler(exp_decay)
    
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
my_optimizer = tf.keras.optimizers.Adadelta( learning_rate=lr_schedule
                                            , rho=0.95 , epsilon=1e-07 )
#else:
#    my_optimizer = tf.keras.optimizers.rmsprop()

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

#if (what_is_my_aim == 1) || (what_is_my_aim == 4):
    # compile with the optimizer (and therefore learning rate schedule) chosen in settings
model.compile(
    optimizer=my_optimizer, 
    loss= tfk.losses.categorical_crossentropy,
    metrics=[ 'accuracy' ], 
    loss_weights=None, 
    sample_weight_mode=None, 
    weighted_metrics=None, 
    target_tensors=None)
'''
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
