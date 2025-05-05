####### Importing Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from SCConv import self_cal_Conv1D
from Transformer import Transformer
from ArcFace import *

####### Loading Dataset
X_train = np.array(np.load('./',allow_pickle=True)['arr_0'],dtype=np.float16)
y_train = np.load('./',allow_pickle=True)['arr_0']

y_train_ohot = tf.keras.utils.to_categorical(y_train)
#y_dev_ohot = tf.keras.utils.to_categorical(y_dev)

####### Model Training

###### LGTraNet
#### Defining Hyperparameters
num_layers = 2
d_model = 512
num_heads = 8
dff = 1024
max_seq_len = 1280 #X_train.shape[1]
pe_input = 320
rate = 0.5
num_features = 1
num_classes = 89

##### Defining Layers
Input_layer = tf.keras.layers.Input(shape=(max_seq_len,num_features))
self_conv1 = self_cal_Conv1D(128,15,128)
self_conv2 = self_cal_Conv1D(128,20,128) # Newly Added
self_conv3 = self_cal_Conv1D(256,15,128)
self_conv4 = self_cal_Conv1D(256,20,256) # Newly Added
self_conv5 = self_cal_Conv1D(512,15,256)
self_conv6 = self_cal_Conv1D(512,20,512) # Newly Added
self_conv7 = self_cal_Conv1D(1024,3,512)
self_conv8 = self_cal_Conv1D(1024,5,1024) # Newly Added
conv_initial = tf.keras.layers.Conv1D(32,15,padding='same',activation='relu')
conv_second = tf.keras.layers.Conv1D(64,15,padding='same',activation='relu')
conv_third = tf.keras.layers.Conv1D(128,15,padding='same',activation='relu')
#lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,activation='tanh',return_sequences=True),merge_mode='ave')
transform_1 = tf.keras.layers.Conv1D(128,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transform_2 = tf.keras.layers.Conv1D(256,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transform_3 = tf.keras.layers.Conv1D(512,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transform_4 = tf.keras.layers.Conv1D(1024,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transformer = Transformer(num_layers,d_model,num_heads,dff,pe_input,rate)
gap_layer = tf.keras.layers.GlobalAveragePooling1D()
arc_logit_layer = ArcFace(89,30.0,0.3,tf.keras.regularizers.l2(1e-4))

##### Defining Architecture

#### Input Layer
Inputs = Input_layer
Input_Labels = tf.keras.layers.Input(shape=(num_classes,))

#### 1D-CNN Backbone
conv_initial = conv_initial(Inputs)
conv_second = conv_second(conv_initial)
conv_third = conv_third(conv_second)

#### SCNRNet
### 1st Residual Block
transform_1 = transform_1(conv_third)
conv1 = self_conv1(conv_third)
conv2 = self_conv2(conv1)
conv2 = tf.keras.layers.Add()([conv2,transform_1])
conv2 = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(conv2)

### 2nd Residual Block
transform_2 = transform_2(conv2)
conv3 = self_conv3(conv2)
conv4 = self_conv4(conv3)
conv4 = tf.keras.layers.Add()([conv4,transform_2])
conv4 = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(conv4)

### 3rd Residual Block
transform_3 = transform_3(conv4)
conv5 = self_conv5(conv4)
conv6 = self_conv6(conv5)
conv6 = tf.keras.layers.Add()([conv6,transform_3])

#### Transformer
embeddings =  transformer(inp=conv6,enc_padding_mask=None)

#### Output Layers
### Initial Layers
gap_op = gap_layer(embeddings)
dense1 = tf.keras.layers.Dense(256,activation='relu')(gap_op)
dropout1 = tf.keras.layers.Dropout(rate)(dense1)

### ArcFace Output Network
dense2 = tf.keras.layers.Dense(256,kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
##dense2 = tf.keras.layers.BatchNormalization()(dense2)
dense3 = arc_logit_layer(([dense2,Input_Labels]))

#### Compiling Architecture            
### ArcFace Model Compilation
model = tf.keras.models.Model(inputs=[Inputs,Input_Labels],outputs=dense3)
model.compile(tf.keras.optimizers.Adam(lr=1e-4,clipnorm=1.0),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#### Model training
filepath = './'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
history = model.fit((X_train,y_train_ohot),y_train_ohot,epochs=500,batch_size=64,
                validation_split=0.1,validation_batch_size=64,
               callbacks=checkpoint)
