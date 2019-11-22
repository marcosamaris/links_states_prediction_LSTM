#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:03:06 2019

@author: marcos
"""

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

lam = 1e-4

def contractive_loss(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)

    W = K.variable(value=autoencoder.get_layer(index=1).get_weights()[0])  # N x N_hidden
    #W = K.variable(value=autoencoder.get_layer(index=3).get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = autoencoder.get_layer(index=1).output
    #h = autoencoder.get_layer(index=3).output
    dh = h * (1 - h)  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

    return mse + contractive
    
    
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


import tensorflow.keras.backend as K
import numpy as np
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.optimizers import RMSprop


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
    rolling_win = 1
    df = df.drop(df[df['link'] == max(df['link'])].index)
    if (line == '7545-10-1') | (line == '477P-10-1') | (line == '3301-10-0') | \
            (line == '3301-10-0') | (line == '574J-10-1') | (line == '574J-10-1'):
            df = df.drop(df[df['link'] == max(df['link'])].index)

    X_Temp = df.groupby([pd.Grouper(freq=str(frequency)), 'link'], as_index=True ).mean()['time_link'].unstack()    
    X_Temp = X_Temp.transform(lambda x: x.fillna(method='ffill')).dropna()
    
    X_Temp = X_Temp.iloc[X_Temp.index.indexer_between_time('06:00', '23:00')]
    
    X_Temp.reset_index(drop=False, inplace=True)

    X_Temp['exact_time'] = pd.to_datetime(X_Temp['exact_time']).dt.date

    result = X_Temp.groupby('exact_time').count()[0]

    var = result.loc[result == result.max()].index

    X_Temp.index = X_Temp['exact_time']
    del X_Temp['exact_time']

    X_Temp = X_Temp.loc[pd.to_datetime(list(var))] 
    
    # this is the size of our encoded representations
    encoding_dim = 5  
    input_dim_x = 3
    input_dim_y = 27
    
    number_test_samples = 10
    # Creates the train and test sets 
    test_samples = result.max()*number_test_samples
    train = X_Temp.values[:-test_samples,:-2] # two last columns have large errors
    test = X_Temp.values[-test_samples:,:-2] # two last columns have large errors

    
        #Normalize the inputs
    trmin = train.min(); trmax = train.max();
    temin = test.min(); temax = test.max();
    train_norm = (train-trmin)/(trmax-trmin)
    test_norm  = (test-temin)/(temax-temin)
    
    train_norm = np.reshape(train_norm, (int(np.shape(train_norm)[0]/3),int(result.max()/6), np.shape(train_norm)[1], 1))
    test_norm = np.reshape(test_norm, (int(np.shape(test_norm)[0]/3),int(result.max()/6), np.shape(test_norm)[1], 1))
    
    print(np.shape(test_norm))
    print(np.shape(train_norm))    

    train_x =  np.delete(train_norm, (np.shape(train_norm)[0]-1), axis=0)   
    train_y =  np.delete(train_norm, (0), axis=0)   
    
    test_x =   np.delete(test_norm, (np.shape(test_norm)[0]-1), axis=0)   
    test_y =   np.delete(test_norm, (0), axis=0) 
    
        
    input_data = Input(shape=(input_dim_x, input_dim_y,1))
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_data)
    x =  Dense(encoding_dim, activation='relu')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Dense(1, activation='sigmoid')(x)
    
    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')
    autoencoder.summary()
    
    print(np.shape(train_norm))
    print(np.shape(test_norm))
    autoencoder.fit(train_x, train_y,
                epochs=5000,
                validation_data=(test_x, test_y),
                batch_size=256,
                shuffle=True)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt
    
    encoded_states = autoencoder.predict(test_x)
    decoded_states = autoencoder.predict(encoded_states)
    #decoded_states = autoencoder.predict(test)
    
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    
    print(np.shape(decoded_states), np.shape(test_y))
    
    test_y = np.reshape(test_y, (number_test_samples*6 -1, input_dim_x*input_dim_y))  # adapt this if using `channels_first` image data format
    decoded_states = np.reshape(decoded_states, (number_test_samples*6-1, input_dim_x* input_dim_y))  # adapt this if using `channels_first` image data format
    print(np.shape(decoded_states), np.shape(test_y))
#    decoded_states = decoded_states *(trmax-trmin)+trmin
    
    print(mean_absolute_error(decoded_states, test_y))
    