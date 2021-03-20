#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:53:08 2019

@author: marcos
"""

'''Example of VAE on MNIST dataset using CNN

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''


from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose, Cropping2D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
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
    filename = '../' + str(line) + '_temp.csv.gz'
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
    scaler = StandardScaler()    
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.fit_transform(test)    
    
    
    train_norm = np.reshape(train_norm, (int(np.shape(train_norm)[0]/3),int(result.max()/6), np.shape(train_norm)[1], 1))
    test_norm = np.reshape(test_norm, (int(np.shape(test_norm)[0]/3),int(result.max()/6), np.shape(test_norm)[1], 1))
    
    
    # network parameters
    
    input_shape = Input(shape=(input_dim_x, input_dim_y,1))
    batch_size = 128
    kernel_size = 3
    filters = 16
    latent_dim = 5
    epochs = 5000
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(input_dim_x, input_dim_y,1))
    x = inputs
    for i in range(2):
        filters *= 3
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
    
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    
    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 3
    
    x = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    
    outputs = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(x) # this is the added step
    
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    
    
    models = (encoder, decoder)
    
    
    # VAE loss = mse_loss or xent_loss + kl_loss
    
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    
    reconstruction_loss *= input_dim_x * input_dim_y
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    #vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop', loss='mse')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
    
    #   
    #    # train the autoencoder
    vae.fit(train_norm, train_norm,
                    epochs=5000,
                    validation_data=(test_norm, test_norm),
                    batch_size=1024,
                    shuffle=True)

  
    
    encoded_states_train = encoder.predict(train_norm)
    encoded_states_test = encoder.predict(test_norm)
     
    from neupy import algorithms    
    from sklearn import tree
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor    
    from sklearn.metrics import mean_absolute_error
        
    encoded_states_train = np.array(encoded_states_train[2][:][:])
    encoded_states_test = np.array(encoded_states_test[2][:][:])
    
    train_x =  x = np.delete(encoded_states_train, (np.shape(encoded_states_train)[0]-1), axis=0)   
    train_y =  x = np.delete(encoded_states_train, (0), axis=0)   
    
    test_x =  x = np.delete(encoded_states_test, (np.shape(encoded_states_test)[0]-1), axis=0)   
    test_y =  x = np.delete(encoded_states_test, (0), axis=0) 
    
    test_norm = np.delete(test_norm, (0), axis=0) 
    
    MAE_decode = []
    MAE_encode = []
    test_norm = np.reshape(test_norm, (number_test_samples*6 -1, input_dim_x*input_dim_y))  # adapt this if using `channels_first` image data format
    MAE_decode.append(mean_absolute_error(test_norm[:,1:], test_norm[:,:np.shape(test_norm)[1]-1]))
    ML_alg = ['lr', 'svm', 'tree', 'rf', 'lasso', 'bayridge', 'ann']
    for ml in ML_alg:
        
        if ml == 'lr':
            model = LinearRegression()
        elif ml == 'svm':
            SVR(gamma=0.001, C=1.0, epsilon=0.2)
        elif  ml == 'tree':
            model = tree.DecisionTreeRegressor()
        elif ml == 'rf':
            model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        elif ml == 'grnn':
            model = algorithms.GRNN(std=0.1, verbose=False)
        elif ml == 'lasso':
            model = Lasso(alpha=0.1)
        elif  ml == 'bayridge':
            model = BayesianRidge()
        elif ml == 'ann':
            model = MLPRegressor(hidden_layer_sizes=(10,))
        
        if ml == 'lr' or  ml == 'tree' or  ml == 'lasso' or  ml == 'ann':
            model.fit(train_x, train_y)    
            y_pred = model.predict(test_x)
        else:
            y_pred = []            
            for i in range(5):
                model.fit(train_x, train_y[:,i])    
                y_pred.append(model.predict(test_x))
        
            y_pred = np.reshape(y_pred, (np.shape(y_pred)[1], np.shape(y_pred)[0]))
    
        MAE_encode.append(mean_absolute_error(test_y, y_pred))
            
        decoded_states = decoder.predict(y_pred)        
        
        decoded_states = np.reshape(decoded_states, (number_test_samples*6-1, input_dim_x* input_dim_y))  # adapt this if using `channels_first` image data format
        
        MAE_decode.append(mean_absolute_error(decoded_states, test_norm))

    x = np.arange(len(ML_alg))
    plt.figure(0)
    plt.bar(x, MAE_encode)
    plt.xticks(x, (ML_alg))
    plt.savefig('MAE_encode_std_norm_deconv_' + line + '.png')
    
    
    x = np.arange(len(ML_alg) + 1)
    plt.figure(1)
    plt.bar(x, MAE_decode)
    ML_alg.insert(0, "Next")
    plt.xticks(x, (ML_alg))
    plt.savefig('MAE_decode_std_norm_deconv_' + line + '.png')
    