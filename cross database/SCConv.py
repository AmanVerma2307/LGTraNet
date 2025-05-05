####### Importing Libraries
import numpy as np
import tensorflow as tf

####### Self-Calibrated Convolutions
class self_cal_Conv1D(tf.keras.layers.Layer):

    """ 
    This is inherited class from keras.layers and shall be instatition of self-calibrated convolutions
    """
    
    def __init__(self,num_filters,kernel_size,num_features):
    
        #### Defining Essentials
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_features = num_features # Number of Channels in Input

        #### Defining Layers
        self.conv2 = tf.keras.layers.Conv1D(self.num_features/2,self.kernel_size,padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-5),dtype='float32',activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(self.num_features/2,self.kernel_size,padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-5),dtype='float32',activation='relu')
        self.conv4 = tf.keras.layers.Conv1D(self.num_filters/2,self.kernel_size,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-5),dtype='float32')
        self.conv1 = tf.keras.layers.Conv1D(self.num_filters/2,self.kernel_size,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-5),dtype='float32')
        self.upsample = tf.keras.layers.Conv1DTranspose(filters=int(self.num_features/2),kernel_size=4,strides=4)
        #self.attention_layer = tf.keras.layers.Attention()
        #self.lstm = tf.keras.layers.LSTM(int(self.num_features/2),return_sequences=True)
        #self.layernorm = tf.keras.layers.LayerNormalization()
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_features': self.num_features
        })
        return config
    
    
    def call(self,X):
       
        """
          INPUTS : 1) X - Input Tensor of shape (batch_size,sequence_length,num_features)
          OUTPUTS : 1) X - Output Tensor of shape (batch_size,sequence_length,num_features)
        """
        
        #### Dimension Extraction
        b_s = (X.shape)[0] 
        seq_len = (X.shape)[1]
        num_features = (X.shape)[2]
        
        #### Channel-Wise Division
        X_attention = X[:,:,0:int(self.num_features/2)]
        X_global = X[:,:,int(self.num_features/2):]
        
        #### Self Calibration Block

        ### Local Feature Detection

        ## Down-Sampling
        #x1 = X_attention[:,0:int(seq_len/5),:]
        #x2 = X_attention[:,int(seq_len/5):int(seq_len*(2/5)),:]
        #x3 = X_attention[:,int(seq_len*(2/5)):int(seq_len*(3/5)),:]
        #x4 = X_attention[:,int(seq_len*(3/5)):int(seq_len*(4/5)),:]
        #x5 = X_attention[:,int(seq_len*(4/5)):seq_len,:]
        x_down_sampled = tf.keras.layers.AveragePooling1D(pool_size=4,strides=4)(X_attention)
        
        ## Convoluting Down Sampled Sequence 
        #x1 = self.conv2(x1)
        #x2 = self.conv2(x2)
        #x3 = self.conv2(x3)
        #x4 = self.conv2(x4)
        #x5 = self.conv2(x5)
        x_down_conv = self.conv2(x_down_sampled)
        #x_down_feature = self.attention_layer([x_down_sampled,x_down_sampled])
        #x_down_feature = self.lstm(x_down_sampled)
        #x_down_feature = self.layernorm(x_down_feature)
        
        ## Up-Sampling
        x_down_upsampled = self.upsample(x_down_conv)   
        #X_local_upsampled = tf.keras.layers.concatenate([x1,x2,x3,x4,x5],axis=1)

        ## Local-CAM
        X_local = X_attention + x_down_upsampled  #X_local_upsampled

        ## Local Importance 
        X_2 = tf.keras.activations.sigmoid(X_local)

        ### Self-Calibration

        ## Global Convolution
        X_3 = self.conv3(X_attention)

        ## Attention Determination
        X_attention = tf.math.multiply(X_2,X_3)

        #### Self-Calibration Feature Extraction
        X_4 = self.conv4(X_attention)

        #### Normal Feature Extraction
        X_1 = self.conv1(X_global)

        #### Concatenating and Returning Output
        return (tf.keras.layers.concatenate([X_1,X_4],axis=2))