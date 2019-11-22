"""
Created on Thu Apr 11 11:55:48 2019

@author: marcos
"""

from keras.models import Model
from keras.layers import (Conv2D, Activation, Dense, Lambda, Input,
    MaxPooling2D, Dropout, Flatten, Reshape, UpSampling2D, Concatenate)
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
import pandas as pd
import numpy as np

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
        
    X_Temp = X_Temp.iloc[X_Temp.index.indexer_between_time('07:00', '23:00')]
    
    X_Temp.reset_index(drop=False, inplace=True)

    X_Temp['exact_time'] = pd.to_datetime(X_Temp['exact_time']).dt.date

    result = X_Temp.groupby('exact_time').count()[0]

    var = result.loc[result == result.max()].index

    X_Temp.index = X_Temp['exact_time']
    del X_Temp['exact_time']

    X_Temp = X_Temp.loc[pd.to_datetime(list(var))]

    
    
    
    # this is the size of our encoded representations
    encoding_dim = 5  
    input_dim_x = 459
    input_dim_y = 500
    
    # Creates the train and test sets 
    test_samples = result.max()
    train = X_Temp.values[:-test_samples,:-2] # two last columns have large errors
    test = X_Temp.values[-test_samples:,:-2] # two last columns have large errors

    
    #Normalize the inputs
    trmin = train.min(); trmax = train.max();
    temin = test.min(); temax = test.max();
    train_norm = (train-trmin)/(trmax-trmin)
    test_norm  = (test-temin)/(temax-temin)
    
    train_norm = np.reshape(train_norm, (int(np.shape(train_norm)[0]/result.max()),result.max(), np.shape(train_norm)[1]))
    test_norm = np.reshape(test_norm, (int(np.shape(test_norm)[0]/result.max()),result.max(),  np.shape(test_norm)[1]))
    

image_shape = (result.max(), 27, 1)
original_dim = image_shape[0] * image_shape[1]
input_shape = (original_dim,)
num_classes = 27
batch_size = 128
latent_dim = 5

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_model():
    """Builds/compiles the model and returns (encoder, decoder, vae)."""

    # encoder
    inputs = Input(shape=input_shape)
    x = Reshape(image_shape)(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    label_inputs = Input(shape=(num_classes,), name='label')
    x = Concatenate()([latent_inputs, label_inputs])
    x = Dense(128, activation='relu')(x)
    x = Dense(14 * 14 * 32, activation='relu')(x)
    x = Reshape((14, 14, 32))(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    outputs = Reshape(input_shape)(x)

    decoder = Model([latent_inputs, label_inputs], outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # variational autoencoder
    outputs = decoder([encoder(inputs)[2], label_inputs])
    vae = Model([inputs, label_inputs], outputs, name='vae_mlp')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    # loss function
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae

def from_weights():
    """Build a model and load its pretrained weights from `weights_file`."""
    encoder, decoder, vae = build_model()
    
    return encoder, decoder, vae

if __name__ == '__main__':
    encoder, decoder, vae = from_weights()
    # decoder.save('decoder.h5')
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(decoder, 'docs')
