from configparser import Interpolation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def model_1D_210411_01(): ## One-dimensional model for 1D RF patches (392x 1) <-> Inspired by Raul's rf_best()
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (392, 1)

    img_inputs_1 = Input(shape=data_shape)
    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(img_inputs_1)
    BatchNorm_1 = BatchNormalization()(conv_1)
    Activation_1 = ELU()(BatchNorm_1)
    Avpool_1 = AveragePooling1D((3))(Activation_1)

    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((3))(conv_2)

    conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    Avpool_3 = AveragePooling1D((3))(conv_3)

    flatten_1 = Flatten()(Avpool_3)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)

    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model


def model_2D_210411_06(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(8, (5,5), padding='same', activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_2D_210411_06_4classes(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 4
    model = Sequential()
    model.add(Conv2D(8, (5,5), padding='same', activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model


def model_2D_210411_06_v02(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding='same', activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_2D_210411_06_v02_4classes(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 4
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding='same', activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model



def model_2D_210411_06b_4classes(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 4
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_2D_210411_06b(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_2D_210411_06bEcofrailComprimido01():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (600, 60, 1)
    num_classes = 2
    model = Sequential()
    #model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())    
    model.add(MaxPooling2D(pool_size=(2, 1))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model 

def model_2D_210411_06bEcofrailComprimido02():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (600, 30, 1)
    num_classes = 2
    model = Sequential()
    #model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(3, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(3, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 1))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model     

def model_2D_210411_06bEcofrail(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    #data_shape = (120, 120, 1)

    IMG_HEIGHT=120
    IMG_WIDTH=120
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # , kernel_regularizer=regularizers.l2(reg)
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(16, 3, padding='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    '''
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    '''
    return model
def model_2D_210411_06bEcofrail01():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model    
def model_2D_210411_06bEcofrail02():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def model_2D_210411_06bEcofrail03():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def model_2D_210411_06bEcofrail04():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model 
def model_2D_210411_06bEcofrail05():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model 
def model_2D_210411_06bEcofrail06():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def model_2D_210411_06bEcofrail07():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(124, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model
#data_augmentation = tf.keras.Sequential([
#  layers.RandomFlip("horizontal"),
#  layers.RandomBrightness(
#    (-0.5,0.5), value_range=(0, 1), seed=None),
#])    
def model_2D_210411_06bEcofrail08():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(RandomFlip("horizontal")),
    model.add(RandomContrast(0.8, seed=1)),
    model.add(RandomCrop(120,120, seed=1)),
    #model.add(RandomRotation((-0.2,0.3),fill_mode = 'nearest',interpolation='nearest',seed=1,fill_value=0.0)),
    #model.add(RandomBrightness((-0.9,0.9), value_range=(0, 1), seed=None)),
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model       
def model_2D_210411_06bEcofrail09():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(num_classes, activation='softmax'))
    return model 

def model_2D_210411_06bEcofrail10():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model   

def model_2D_210411_06bEcofrail11():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(4, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_2D_210411_06bEcofrail12():
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (120, 120, 1)
    num_classes = 2
    model = Sequential()
    model.add(RandomFlip("horizontal")),
    model.add(RandomContrast(0.8, seed=1)),
    model.add(RandomCrop(120,120, seed=1)),
    #model.add(RandomRotation((-0.2,0.3),fill_mode = 'nearest',interpolation='nearest',seed=1,fill_value=0.0)),
    #model.add(RandomBrightness((-0.9,0.9), value_range=(0, 1), seed=None)),
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (9,9), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model       
'''
def model_2D_210411_06b_v02(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(AveragePooling2D(pool_size =(1,8), input_shape=data_shape)) # Demodulation
    model.add(Conv2D(8, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model
'''

def model_2D_210411_02(): ## Two-dimensional model for 2D RF patches (x2) <-> Inspired by IEEE IUS 2020 RF vs Bmode paper
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(16, (5,5), padding='same', activation='relu', input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Dropout(0.2))
    model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_2D_210420_12(): ## Two-dimensional model for 2D RF patches (x2) <-> Inspired by IEEE IUS 2020 RF vs Bmode paper
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(MaxPooling2D(pool_size = (1,8), input_shape=data_shape))
    model.add(Conv2D(16, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
def model_2D_210420_13(): ## Try pretrained model architectures InceptionV3
    data_shape = (27, 392, 1)
    num_classes = 2
    img_inputs_1 = Input(shape=data_shape)
    #maxpool_1 = MaxPooling2D((1,8))(img_inputs_1)

    base_model = InceptionV3(weights = 'imagenet', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a fully connected layer
    x = Dense(1024, activation = 'relu')(x)
    # Add a softmax layer
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs=base_model.input,outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False # Freeze all layers of InceptionV3 model

    '''
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    '''

    return model


def model_3D_210411_03(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv3D(32, (3,3,7), padding='same', strides = (1,1,1), activation='relu', input_shape=data_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,32)))
    model.add(Conv1D(128, padding = 'same', kernel_size = 3))
    model.add(Dropout(0.2))
    model.add(AveragePooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model



def model_2D_210411_04_nodropout(): ## Two-dimensional model for 2D RF patches (x2) <-> Inspired by IEEE IUS 2020 RF vs Bmode paper
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(16, (5,5), padding='same', activation='relu', input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_3D_210411_05_nodropout(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv3D(32, (3,3,7), padding='same', strides = (1,1,1), activation='relu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,32)))
    model.add(Conv1D(128, padding = 'same', kernel_size = 3))
    model.add(AveragePooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_3D_210411_07(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv3D(32, (3,3,7), padding='same', strides = (1,1,1), activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,32)))
    model.add(Conv1D(128, padding = 'same', kernel_size = 3, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_3D_210411_07_optim(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv3D(16, (3,3,7), padding='same', strides = (1,1,1), activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,16)))
    model.add(Conv1D(64, padding = 'same', kernel_size = 3, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model    

def model_3D_210411_07_optim_4classes(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 4
    model = Sequential()
    model.add(Conv3D(16, (3,3,7), padding='same', strides = (1,1,1), activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,16)))
    model.add(Conv1D(64, padding = 'same', kernel_size = 3, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model   

#####

def model_1D_210418_08_4classes(): ## One-dimensional model for 1D RF patches (392x 1) <-> Inspired by Raul's rf_best()
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (392, 1)
    img_inputs_1 = Input(shape=data_shape)
    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(img_inputs_1)
    Activation_1 = ELU()(conv_1)   # Modification - Batchnorm after activation
    BatchNorm_1 = BatchNormalization()(Activation_1)
    Avpool_1 = AveragePooling1D((3))(Activation_1)
    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((3))(conv_2)
    conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    Avpool_3 = AveragePooling1D((3))(conv_3)
    flatten_1 = Flatten()(Avpool_3)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(4, activation='softmax')(dense_2)
    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model

def model_1D_210418_08(): ## One-dimensional model for 1D RF patches (392x 1) <-> Inspired by Raul's rf_best()
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (392, 1)
    img_inputs_1 = Input(shape=data_shape)
    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(img_inputs_1)
    Activation_1 = ELU()(conv_1)   # Modification - Batchnorm after activation
    BatchNorm_1 = BatchNormalization()(Activation_1)
    Avpool_1 = AveragePooling1D((3))(Activation_1)
    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((3))(conv_2)
    conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    Avpool_3 = AveragePooling1D((3))(conv_3)
    flatten_1 = Flatten()(Avpool_3)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)
    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model

def model_1D_210418_08b_4classes(): ## One-dimensional model for 1D RF patches (392x 1) <-> Inspired by Raul's rf_best()
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (392, 1)
    img_inputs_1 = Input(shape=data_shape)
    Avpool_0 = AveragePooling1D((8))(img_inputs_1)

    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(Avpool_0)
    Activation_1 = ELU()(conv_1)   # Modification - Batchnorm after activation
    BatchNorm_1 = BatchNormalization()(Activation_1)
    Avpool_1 = AveragePooling1D((2))(Activation_1)
    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((2))(conv_2)
    #conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    #Avpool_3 = AveragePooling1D((3))(conv_3)
    #flatten_1 = Flatten()(Avpool_3)
    flatten_1 = Flatten()(Avpool_2)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(4, activation='softmax')(dense_2)
    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model

def model_1D_210418_08b(): ## One-dimensional model for 1D RF patches (392x 1) <-> Inspired by Raul's rf_best()
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (392, 1)
    img_inputs_1 = Input(shape=data_shape)
    Avpool_0 = AveragePooling1D((8))(img_inputs_1)

    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(Avpool_0)
    Activation_1 = ELU()(conv_1)   # Modification - Batchnorm after activation
    BatchNorm_1 = BatchNormalization()(Activation_1)
    Avpool_1 = AveragePooling1D((2))(Activation_1)
    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((2))(conv_2)
    #conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    #Avpool_3 = AveragePooling1D((3))(conv_3)
    #flatten_1 = Flatten()(Avpool_3)
    flatten_1 = Flatten()(Avpool_2)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)
    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model

def model_2D_210418_09(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(16, (5,5), padding='same', activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(8, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model





##### First iteration on models - 1D rf lines for all
def model_201512_01_rf():
    return rf_best_()

def model_201512_01_iq():
    return rf_best_()

def model_201512_01_env():
    return rf_best_()

def model_201512_01_log():
    return rf_best_()

def model_201512_01_den():
    return rf_best_()

##### Second iteration on models - introduce decimation on demodulated models


#### Third iteration - 2D inception for all


#### Fourth iteration - introduce decimation to 2D model




##### Utility functions ################################

# DEFINE MODEL

# filters = 5
# dropout = 0.5
# kernel_size = ( 4 , 3 , 10 )
# activ = 'relu'
# my_params = [ filters , dropout , kernel_size , activ ]
# model = nps_1( my_params )
def nps_1(params):
    # parameters
    filters = params[0]
    dropout = params[1]
    kernel_size = params[2]
    activ = params[3]

    # model
    input_1 = tfk.Input(shape=data_shape)

    conv_1 = tfk.layers.Conv3D(filters, kernel_size, strides=(
        1, 1, 1), activation=activ)(input_1)

    flat_1 = tfk.layers.Flatten()(conv_1)

    drop_1 = tfk.layers.Dropout(dropout)(flat_1)

    output = tfk.layers.Dense(numClasses, activation='softmax')(drop_1)

    # finish
    model = tfk.Model(inputs=input_1, outputs=output)
    return model


def rf_best_():
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (2928, 1)

    img_inputs_1 = Input(shape=data_shape)
    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(img_inputs_1)
    BatchNorm_1 = BatchNormalization()(conv_1)
    Activation_1 = ELU()(BatchNorm_1)
    Avpool_1 = AveragePooling1D((3))(Activation_1)

    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((3))(conv_2)

    conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    Avpool_3 = AveragePooling1D((3))(conv_3)

    conv_4 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_3)
    Avpool_4 = AveragePooling1D((3))(conv_4)

    flatten_1 = Flatten()(Avpool_4)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)

    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model


def rf_best_2():
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (2928, 1)

    img_inputs_1 = Input(shape=data_shape)
    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(img_inputs_1)
    BatchNorm_1 = BatchNormalization()(conv_1)
    Activation_1 = ELU()(BatchNorm_1)
    Avpool_1 = AveragePooling1D((3))(Activation_1)

    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((3))(conv_2)

    conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    Avpool_3 = AveragePooling1D((3))(conv_3)

    conv_4 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_3)
    Avpool_4 = AveragePooling1D((3))(conv_4)

    flatten_1 = Flatten()(Avpool_4)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)

    model = tfk.Model(inputs=img_inputs_1, outputs=output)
    return model

def rf_best_bis():
    filters1 = 9
    kernel_size1 = 7
    filters2 = 9
    kernel_size2 = 7
    dropout = 0.3
    activ = 'elu'
    data_shape = (2928, 1)

    img_inputs_1 = Input(shape=data_shape)
    Avpool_0 = AveragePooling1D((8))(img_inputs_1)


    conv_1 = Conv1D(filters1, kernel_size=kernel_size1)(Avpool_0)
    BatchNorm_1 = BatchNormalization()(conv_1)
    Activation_1 = ELU()(BatchNorm_1)
    Avpool_1 = AveragePooling1D((2))(Activation_1)

    conv_2 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_1)
    Avpool_2 = AveragePooling1D((2))(conv_2)

    conv_3 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_2)
    Avpool_3 = AveragePooling1D((2))(conv_3)

    conv_4 = Conv1D(10, kernel_size=3, activation='elu')(Avpool_3)
    Avpool_4 = AveragePooling1D((2))(conv_4)

    flatten_1 = Flatten()(Avpool_4)
    dense_1 = Dense(20, activation='elu')(flatten_1)
    dense_2 = Dense(10, activation='elu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)

    model = tfk.Model(inputs=img_inputs_1, outputs=output)


    return model





def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None ): 
    kernel_init = 'random_normal'
    bias_init = 'zeros'
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)
    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)
    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output


def inception():
    kernel_init = 'random_normal'
    bias_init = 'zeros'
    input_layer = Input(shape=(2928, 192, 1))
    
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)
    
    x = inception_module(x,
                             filters_1x1=64,
                             filters_3x3_reduce=96,
                             filters_3x3=128,
                             filters_5x5_reduce=16,
                             filters_5x5=32,
                             filters_pool_proj=32,
                             name='inception_3a')
    x = inception_module(x,
                             filters_1x1=128,
                             filters_3x3_reduce=128,
                             filters_3x3=192,
                             filters_5x5_reduce=32,
                             filters_5x5=96,
                             filters_pool_proj=64,
                             name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    
    x = inception_module(x,
                             filters_1x1=192,
                             filters_3x3_reduce=96,
                             filters_3x3=208,
                             filters_5x5_reduce=16,
                             filters_5x5=48,
                             filters_pool_proj=64,
                             name='inception_4a')
    
    '''
    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(2, activation='softmax', name='auxilliary_output_1')(x1)
    '''
    x = inception_module(x,
                             filters_1x1=160,
                             filters_3x3_reduce=112,
                             filters_3x3=224,
                             filters_5x5_reduce=24,
                             filters_5x5=64,
                             filters_pool_proj=64,
                             name='inception_4b')
    
    x = inception_module(x,
                             filters_1x1=128,
                             filters_3x3_reduce=128,
                             filters_3x3=256,
                             filters_5x5_reduce=24,
                             filters_5x5=64,
                             filters_pool_proj=64,
                             name='inception_4c')
    
    x = inception_module(x,
                             filters_1x1=112,
                             filters_3x3_reduce=144,
                             filters_3x3=288,
                             filters_5x5_reduce=32,
                             filters_5x5=64,
                             filters_pool_proj=64,
                             name='inception_4d')
    
    '''
    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(2, activation='softmax', name='auxilliary_output_2')(x2)
    '''
    x = inception_module(x,
                             filters_1x1=256,
                             filters_3x3_reduce=160,
                             filters_3x3=320,
                             filters_5x5_reduce=32,
                             filters_5x5=128,
                             filters_pool_proj=128,
                             name='inception_4e')
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    
    x = inception_module(x,
                             filters_1x1=256,
                             filters_3x3_reduce=160,
                             filters_3x3=320,
                             filters_5x5_reduce=32,
                             filters_5x5=128,
                             filters_pool_proj=128,
                             name='inception_5a')
     
    x = inception_module(x,
                             filters_1x1=384,
                             filters_3x3_reduce=192,
                             filters_3x3=384,
                             filters_5x5_reduce=48,
                             filters_5x5=128,
                             filters_pool_proj=128,
                             name='inception_5b')
     
    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
     
    x = Dropout(0.4)(x)
     
    x = Dense(2, activation='softmax', name='output')(x)
     
        # model = tfk.Model(inputs=input_layer, outputs=x)\n",
     
#    model = tfk.Model(input_layer, [x, x1, x2], name='inception_v1')
    model = tfk.Model(input_layer, x, name='inception_v1')
    
    return model

def inception_2():
    input_layer = Input(shape=(2928, 192, 1)) 
    kernel_init = 'random_normal'
    bias_init = 'zeros'
    x = Conv2D(10, (15, 7), padding='same', strides=(10, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer) 
    x = BatchNorm((10, 3), padding='same', strides=(10, 2), name='max_pool_1_3x3/2')(x) 
    x = MaxPool2D((10, 3), padding='same', strides=(10, 2), name='max_pool_1_3x3/2')(x) 
    x = Conv2D(10, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x) 
    x = Conv2D(10, (10, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x) 
    x = MaxPool2D((10, 3), padding='same', strides=(10, 2), name='max_pool_2_3x3/2')(x) 
    
    x = inception_module(x, 
                             filters_1x1=10, 
                             filters_3x3_reduce=10, 
                             filters_3x3=10, 
                             filters_5x5_reduce=10, 
                             filters_5x5=5, 
                             filters_pool_proj=5, 
                             name='inception_3a') 
        
    x = inception_module(x, 
                             filters_1x1=10, 
                             filters_3x3_reduce=10, 
                             filters_3x3=10, 
                             filters_5x5_reduce=5, 
                             filters_5x5=10, 
                             filters_pool_proj=8, 
                             name='inception_3b') 
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x) 
    
    x = AveragePooling2D((3, 3), padding='same', strides=(2, 2), name='AV_pool_3_3x3/2')(x) 
    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x) 
    
    x = Dropout(0.4)(x) 
    
    x = Dense(2, activation='softmax', name='output')(x) 
    
    # model = tfk.Model(inputs=input_layer, outputs=x) 
    model = tfk.Model(input_layer, [x], name='inception_v1') 
    
    return model


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
def model_2D_pretrainInceptionV3_fullimage(): ## Try pretrained model architectures InceptionV3
    data_shape = (2928, 192)
    num_classes = 2
    img_inputs_1 = Input(shape=data_shape)
    img_inputs_rgb = tf.stack([img_inputs_1, img_inputs_1, img_inputs_1], axis = 3)
#    img_inputs_rgb = tf.image.grayscale_to_rgb(img_inputs_1, name=None)

    maxpool_1 = MaxPooling2D((8,1))(img_inputs_rgb) # 2928/1, 192, x1

    base_model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = maxpool_1, input_shape = (366,192,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a fully connected layer
    x = Dense(256, activation = 'relu')(x)
    # Add a softmax layer
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs=img_inputs_1,outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False # Freeze all layers of InceptionV3 model

    '''
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    '''

    return model


def PHILIPS_N35_bmode_model():
    IMG_HEIGHT=51
    IMG_WIDTH=51
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # , kernel_regularizer=regularizers.l2(reg)
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def PHILIPS_N35_bmode_model_2D_210411_06b_4class(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape =(51, 51, 1)
    num_classes = 4
    model = Sequential()
    model.add(Conv2D(8, (5,5), padding='same', activation='relu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def PHILIPS_N35_bmode_model_2D_210430_4class():
    IMG_HEIGHT=51
    IMG_WIDTH=51
    num_classes = 4
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # , kernel_regularizer=regularizers.l2(reg)
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model



# Hyperparameter optimization on May 15, 2021
# 1) rf2D with ELU in last 2 layers -> model_2D_210515_06_elu_last2 <- model_2D_210411_06, <- OK ####
# 2) ps, pslog, pslognormden 3D architecture with all RELUs -> model_3D_210515_07_optim_allrelus <- model_3D_210411_07_optim <- OK
# 4) ps, pslog, pslognormden 3D architecture with all MaxPools -> model_3D_210515_07_optim_maxpools <- OK
# 2) ps, pslog, pslognormden 2D architecture with all RELUs -> model_2D_210515_06_allrelus <- model_2D_210515_06_allrelus <- OK  #####
# 5) env, den trained with model_2D_210411_06 <- OK

def model_2D_210515_06_elu_last2(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(8, (5,5), padding='same', activation='elu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_2D_210515_06_allrelus(): 
    ## Trying to optimize 2D model according to literature research for spectrogram classification
    data_shape = (27, 392, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(8, (5,5), padding='same', activation='relu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3)))  
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 3))) 
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2))) 
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))	
    return model

def model_3D_210515_07_optim_allrelus(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv3D(16, (3,3,7), padding='same', strides = (1,1,1), activation='relu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,16)))
    model.add(Conv1D(64, padding = 'same', kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model  


def model_3D_210515_07_optim_maxpools(): ## Three-dimensional model for 3D RF patches  <-> Inspired by Ahmed's prostate cancer tests 
    data_shape = (27, 14, 28, 1)
    num_classes = 2
    model = Sequential()
    model.add(Conv3D(16, (3,3,7), padding='same', strides = (1,1,1), activation='relu', input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(27,14,1)))
    model.add(Reshape((28,16)))
    model.add(Conv1D(64, padding = 'same', kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model 