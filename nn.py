#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:15:07 2019

@author: marcos
"""

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def contractive_loss(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)

    #W = K.variable(value=autoencoder.get_layer(index=1).get_weights()[0])  # N x N_hidden
    W = K.variable(value=autoencoder.get_layer(index=3).get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    #h = autoencoder.get_layer(index=1).output
    h = autoencoder.get_layer(index=3).output
    dh = h * (1 - h)  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
    
    
    return mse + contractive

def on_epoch_end(self):
  'Updates indexes after each epoch'
  self.indexes = np.arange(len(self.list_IDs))
  if self.shuffle == True:
      np.random.shuffle(self.indexes)
      
      
def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
             n_classes=10, shuffle=True):
    'Initialization'
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()

selected_line =['8700-10-1',
                '7545-10-1',
                '7545-10-0',   #Less two links
                '6450-10-1',
                '6450-10-0',
                '3301-10-1',   #Less two links
                '2290-10-1',
                '2290-10-0',
                '477P-10-0',        
                '3301-10-0',   #Less two links
                '574J-10-1',   #Less two links
                '574J-10-0',   #Less two links
                '477P-10-1',   #Less two links
                '351F-10-1',
                '351F-10-0'] 

selected_line = ['6450-10-0']

### Size of the steps to group
frequencies = ['20min', '30min', '1H', '3H', '1d', '7d', '1m']
frequencies = ['30min']

for line in selected_line:  
    filename = str(line) + '_temp.csv.gz'
    df = pd.read_csv(filename, compression='gzip', sep=',')
    df['exact_time'] = pd.to_datetime(df['exact_time'], format = '%Y-%m-%d %H:%M')
    df.index = df['exact_time']
    
    df.loc[(df['time_link'] > 5),'time_link'] = np.ceil(df['time_link'].mean())
    
    start_date = pd.to_datetime('2017-1-1', format = '%Y-%m-%d')
    end_date = pd.to_datetime('2017-9-25', format = '%Y-%m-%d')
    df = df.loc[(df['holiday'] != 1) & ((df['weekday'] > 0) & (df['weekday'] < 5))]
 
    
    frequency = '60min'            
    rolling_win = 3
    df = df.drop(df[df['link'] == max(df['link'])].index)
    if (line == '7545-10-1') | (line == '477P-10-1') | (line == '3301-10-0') | \
            (line == '3301-10-0') | (line == '574J-10-1') | (line == '574J-10-1'):
            df = df.drop(df[df['link'] == max(df['link'])].index)

    X_Temp = df.groupby([pd.Grouper(freq=str(frequency)), 'link'], as_index=True ).mean()['time_link'].unstack()    
    X_Temp = X_Temp.transform(lambda x: x.fillna(method='ffill')).dropna()
    
    
    from keras.layers import Input
    from keras.layers import Dense
    from keras.models import Model
    
    nn_input_dim = 5*3
    nn_output_dim = 5
    
    nn_input_data = Input(shape=(nn_input_dim,))
    layer1 = Dense(20, activation='relu')(nn_input_data)
    layer2 = Dense(10, activation='relu')(layer1)
    layer3 = Dense(nn_output_dim, activation='sigmoid')(layer2)
    
    # this model maps an input to its reconstruction
    nn = Model(nn_input_data, layer3)
    
    # Creates the train and test sets 
    test_samples = 500
    
    train = X_Temp.values[:-test_samples,:-2] # two last columns have large errors
    test = X_Temp.values[-test_samples:,:-2] # two last columns have large errors
    
    
    import keras.backend as K
    import numpy as np
    from keras.layers import Input
    from keras.layers import Dense
    from keras.models import Model
    
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    lam = 1e-4


    # this is the size of our encoded representations
    encoding_dim = 5  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    input_dim = 27
    
    number_neurons_input = [10]
    number_neurons_output = [10]
    
    configs = list()
    for neurons_input in number_neurons_input:
        for neurons_output in number_neurons_output:
            # this is our input placeholder
            input_data = Input(shape=(input_dim,))
            encoded1 = Dense(neurons_input, activation='relu')(input_data)
            encoded2 = Dense(encoding_dim, activation='sigmoid')(encoded1)
            
            decoded1 = Dense(neurons_output, activation='relu')(encoded2)
            decoded2 = Dense(input_dim, activation='sigmoid')(decoded1)
            #encoded1 = Dense(20, activation='sigmoid')(input_data)
            #encoded2 = Dense(10, activation='sigmoid')(encoded1)
            #encoded3 = Dense(encoding_dim, activation='sigmoid')(encoded2)
            
            #decoded1 = Dense(10, activation='sigmoid')(encoded3)
            #decoded2 = Dense(20, activation='sigmoid')(decoded1)
            #decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)
            
            
            # this model maps an input to its reconstruction
            autoencoder = Model(input_data, decoded2)
            autoencoder.compile(optimizer='adam', loss="binary_crossentropy") #contractive_loss, binary_crossentropy, mean_squared_error
            
            i1 = autoencoder.get_layer(index=0)
            e1 = autoencoder.get_layer(index=1)
            e2 = autoencoder.get_layer(index=2)
#            e3 = autoencoder.get_layer(index=3)
            
            o1 = autoencoder.get_layer(index=3)
            o2 = autoencoder.get_layer(index=4)
#            o3 = autoencoder.get_layer(index=5)
            
            # this model maps an input to its encoded representation
            encoder = Model(input_data, e2(e1(input_data)))  
        
            enc_train = encoder.predict(train)
            enc_test  = encoder.predict(test)
            train_X_enc = np.concatenate([enc_train[:-3], np.roll(enc_train,-1,axis=0)[:-3], np.roll(enc_train,-2,axis=0)[:-3]], axis=1)
            train_Y_enc = np.roll(enc_train,-3,axis=0)[:-3]
            test_X_enc  = np.concatenate([enc_test[:-3],  np.roll(enc_test,-1,axis=0)[:-3],  np.roll(enc_test,-2,axis=0)[:-3]],  axis=1)
            test_Y_enc  = np.roll(enc_test,-3,axis=0)[:-3]
            
            #Normalize the inputs
            nn_trmin = train_X_enc.min(); nn_trmax = train_X_enc.max();
            nn_temin = test_X_enc.min(); nn_temax = test_X_enc.max();
            
            train_X = (train_X_enc-nn_trmin)/(nn_trmax-nn_trmin)
            test_X  = (test_X_enc -nn_temin)/(nn_temax-nn_temin)
            train_Y = (train_Y_enc-nn_trmin)/(nn_trmax-nn_trmin)
            test_Y  = (test_Y_enc -nn_temin)/(nn_temax-nn_temin)
            
            
            n_epochs = [100, 200]
            n_batch = [32, 64]
            n_optmizer = ['rmsprop', 'adam']
        #    n_optmizer = ['adam']
            n_loss = ['categorical_crossentropy', 'binary_crossentropy']
        #    n_loss = ['binary_crossentropy']
            
            
            
            
            for i in n_epochs:
                for j in n_batch:
                    for k in n_optmizer:
                        for l in n_loss:
                            nn.compile(optimizer=k, loss=l) #mean_squared_error
                            
                            nn.fit(train_X, train_Y,
                                epochs=i,
                                    batch_size=j,
                                    shuffle=True,
                                    validation_data=(test_X, test_Y),
                                    verbose=1)
                    
                    
                    from sklearn.metrics import mean_absolute_error
                    from sklearn.metrics import mean_squared_error
                    
            
                    predicted_Y = nn.predict(test_X)
                
        #            print(mean_absolute_error(predicted_Y, test_Y)/test_Y.mean())
                    
                    cfg = [i, j, mean_absolute_error(predicted_Y, test_Y)/test_Y.mean(), neurons_input, neurons_output]
                    configs.append(cfg)
                    
                    errors = (predicted_Y-test_Y)/test_Y.mean()
                    errors_pd = pd.DataFrame(errors)
        #            errors_pd.describe()
print(configs)
