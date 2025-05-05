####### Importing Libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import tensorflow as tf
import os                                                                                                         
import gc
import math
#import pydot
from scipy.spatial import distance

####### Loading Dataset
Train_Embeddings = np.load('./',allow_pickle=True)['arr_0']
Test_Embeddings = np.load('./',allow_pickle=True)['arr_0']
y_train = np.load('./',allow_pickle=True)['arr_0']
y_dev = np.load('./',allow_pickle=True)['arr_0']

###### Score File Generation
##### Defining Essentials
match_text_genuine = open("./Genuine.txt",'w') # Matching Text File Creation for Genuine
match_text_imposter = open("./Imposter.txt",'w') # Matching Text File Creation for Imposter 
num_probe_samples = 113 # Number of Subjects to be considered in Probe            
num_gallery_samples = 113 # Number of Subjects to be considered in Gallery

##### Probe Selection
for probe_class in range(num_probe_samples): # Looping over Probe Examples
    
    print('Currently Processing for Probe Class '+str(probe_class))

    #### Curating List of Examples belonging to Class 'probe_class'
    item_index = []
    for j in range(y_dev.shape[0]):
        if(y_dev[j] == probe_class):
            item_index.append(j)   

    for probe_id,probe_loc in enumerate(item_index):
        probe = Test_Embeddings[probe_loc] # Probe Selection 

        ##### Gallery Selection
        for gallery_class in range(num_gallery_samples): # Looping over Gallery Examples

        #    print('Currently Processing for Gallery Class '+str(gallery_class))

            #### Curating List of Examples belonging to Class 'probe_class'
            item_index = []
            for j in range(y_train.shape[0]):
                if(y_train[j] == gallery_class):
                    item_index.append(j)   

            for gallery_id,gallery_loc in enumerate(item_index):
                gallery = Train_Embeddings[gallery_loc] # Probe Selection 

                ### Metrics Computation 
                cos_distance = distance.cosine(probe,gallery) # Cosine Distance between Probe and Gallery
                
                if(probe_class == gallery_class): # Identity Flag

                    ### Formulation of Current Matrix
                    identity_flag = 1
                    current_matching_matrix = [probe_class+1,probe_id+1,gallery_class+1,gallery_id+1,identity_flag,cos_distance]

                    ### For Text File Appending
                    for item_idx,item in enumerate(current_matching_matrix):
                        if(item_idx <= 4):
                            match_text_genuine.write(str(item)+'            ')
                        else:
                            match_text_genuine.write(str(item)+"\n")

                else:

                    ### Formulation of Current Matrix
                    identity_flag = 0
                    current_matching_matrix = [probe_class+1,probe_id+1,gallery_class+1,gallery_id+1,identity_flag,cos_distance]

                    ### For Text File Appending
                    for item_idx,item in enumerate(current_matching_matrix):
                        if(item_idx <= 4):
                            match_text_imposter.write(str(item)+'            ')
                        else:
                            match_text_imposter.write(str(item)+"\n")

