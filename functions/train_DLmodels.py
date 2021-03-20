#!/usr/bin/python


import sys 
import os 
sys.path.append(os.getcwd())

import functions.utils as utils

import numpy as np
import time
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Flatten, Reshape, RepeatVector, TimeDistributed, Dense, Conv1D, LSTM, ConvLSTM2D, MaxPooling1D, BatchNormalization, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.utils import plot_model

print('tensorflow ver.: ' + tf.__version__) 




    
def LSTM_model(groupedDataScaled, line, links, n_steps_in, n_steps_out, epochs):
    
    train = groupedDataScaled[:int(len(groupedDataScaled*.8))]
    test = groupedDataScaled[int(len(groupedDataScaled)*.8):]

    X, y = utils.split_sequences(train, n_steps_in, n_steps_out)
    X_test, y_test = utils.split_sequences(test, n_steps_in, n_steps_out)

    # the dataset knows the number of features, e.g. 2
    n_features_in = X.shape[2]
    n_features_out = y.shape[2]

    # define model
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (n_steps_in, n_features_in)))
    model.add(LSTM(name ='lstm_1',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(LSTM(name ='lstm_2',
                   units = 64,
                   return_sequences = False))
    
    model.add(Dropout(0.1, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(RepeatVector(n_steps_out))
    
    model.add(LSTM(name ='lstm_3',
                   units = 64,
                   return_sequences = True))
    
    model.add(Dropout(0.1, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))
    
    model.add(LSTM(name ='lstm_4',
                   units = n_features_out,
                   return_sequences = True))
    
    model.add(TimeDistributed(Dense(units=n_features_out, name = 'dense_1', activation = 'linear')))

    optimizer = RMSprop()
    model.compile(loss = "mse", optimizer = optimizer)
    
    logdir = "logs/" + str(n_steps_in) + '_LSTM_' + line +'_'+str(epochs)+ '.log'
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    csv_logger = CSVLogger(logdir,separator=",", append=True)    

    # fit model    
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=128, validation_data = (X_test, y_test), callbacks=[csv_logger])    
    model.save('models/n_steps_in_'+ str(n_steps_in) + '_LSTM_' + line+'_'+str(epochs)+ '.h5')    
    return model

def BidirectionalLSTM_model(groupedDataScaled, line, links, n_steps_in, n_steps_out, epochs):
    train = groupedDataScaled[:int(len(groupedDataScaled*.8))]
    test = groupedDataScaled[int(len(groupedDataScaled)*.8):]
       
    # convert into input/output
    X, y = utils.split_sequences(train, n_steps_in, n_steps_out)
    X_test, y_test = utils.split_sequences(test, n_steps_in, n_steps_out)

    # flatten output
    n_output = y.shape[1] * y.shape[2]
    y = y.reshape((y.shape[0], n_output))
    y_test = y_test.reshape((y_test.shape[0], n_output))

    # the dataset knows the number of features, e.g. 2
    n_features_in = X.shape[2]

    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(n_steps_in, n_features_in)))
    
    model.add(Dense(n_output))
    opt = 'adam'
    
    model.compile(optimizer='adam', loss='mse')
    
    logdir = "logs/" + str(n_steps_in) + '_BidirectionalLSTM_' + line + '_'+str(epochs)+ '.log'
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    csv_logger = CSVLogger(logdir,separator=",", append=True)    

    # fit model    
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=128, validation_data = (X_test, y_test), callbacks=[early_stop, csv_logger])    
    model.save('models/n_steps_in_'+ str(n_steps_in) + '_BidirectionalLSTM_' + line + '_' + str(epochs) + '.h5')  
    
    
    return model
    

def ConvLSTM1D_model(groupedDataScaled, line, links, n_steps_in, n_steps_out, epochs):
    
    train = groupedDataScaled[:int(len(groupedDataScaled*.8))]
    test = groupedDataScaled[int(len(groupedDataScaled)*.8):]

    X, y = utils.split_sequences(train, n_steps_in, n_steps_out)
    X_test, y_test = utils.split_sequences(test, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], n_output))
    y_test = y_test.reshape((y_test.shape[0], n_output))

    n_features_in = X.shape[2]
    n_features_out = y.shape[1]

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=5, activation='relu'), 
                                          input_shape=(n_steps_in, n_features_in,1)))

    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))                                            
                           
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=5, activation='relu')))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, activation='relu'))

    model.add(Dense(n_features_out))

    model.compile(optimizer='adam', loss='mse')
    
    logdir = "logs/" + str(n_steps_in) + '_ConvLSTM1D_' + line + '_'+str(epochs)+ '.log'
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = CSVLogger(logdir,separator=",", append=True)    

    # fit model    
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=128, validation_data = (X_test, y_test), callbacks=[early_stop, csv_logger])    
    model.save('models/n_steps_in_'+ str(n_steps_in) + '_ConvLSTM1D_' + line+'_'+str(epochs)+ '.h5')  
    
    
    return model


                                 
             

def ConvLSTM2D_model(groupedDataScaled, line, links, n_steps_in, n_steps_out, epochs):
    
    train = groupedDataScaled[:int(len(groupedDataScaled*.8))]
    test = groupedDataScaled[int(len(groupedDataScaled)*.8):]

    X, y = utils.split_sequences(train, n_steps_in, n_steps_out)
    X_test, y_test = utils.split_sequences(test, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], 1)

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], n_output))
    y_test = y_test.reshape((y_test.shape[0], n_output))

    n_features_in = X.shape[3]
    n_features_out = y.shape[1]

    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(5,1), activation='relu', 
                         input_shape=(1, n_steps_in, n_features_in, 1), return_sequences=True))
    
    #model.add(Dropout(0.2, name = 'dropout_1'))
    #model.add(BatchNormalization(name = 'batch_norm_1'))    
    #model.add(ConvLSTM2D(filters=32, kernel_size=(5,1), activation='relu', return_sequences=True))

    #model.add(Dropout(0.2, name = 'dropout_2'))
    #model.add(BatchNormalization(name = 'batch_norm_2'))
    #model.add(ConvLSTM2D(filters=64, kernel_size=(5,1), activation='relu', return_sequences=True))
    
    #model.add(Dropout(0.1, name = 'dropout_3'))
    #model.add(BatchNormalization(name = 'batch_norm_3'))
    #model.add(ConvLSTM2D(filters=128, kernel_size=(5,1), activation='relu', return_sequences=True))
   
    model.add(Flatten())                                    
    #model.add(Dense(128))
    #model.add(Dense(64))
    #model.add(Dense(32))
    model.add(Dense(n_features_out))
    
    model.add(Dense(n_features_out))
    model.compile(optimizer='adam', loss='mse')
    #plot_model(model, to_file='ConvLSTM2D_model.png',rankdir='TB', show_shapes=False, show_layer_names=True)
    #print(model.summary())
    
    logdir = "logs/" + str(n_steps_in) + '_ConvLSTM2D_' + line +'_'+str(epochs)+  '.log'
    
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    csv_logger = CSVLogger(logdir,separator=",", append=True)    

    # fit model    
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=128, validation_data = (X_test, y_test), callbacks=[early_stop, csv_logger])    
    model.save('models/n_steps_in_'+ str(n_steps_in) + '_ConvLSTM2D_' + line+'_'+str(epochs)+ '.h5')  
    
    return model


def ConvLSTM2D_Deep_model(groupedDataScaled, line, links, n_steps_in, n_steps_out, epochs):
    
    train = groupedDataScaled[:int(len(groupedDataScaled*.8))]
    test = groupedDataScaled[int(len(groupedDataScaled)*.8):]

    X, y = utils.split_sequences(train, n_steps_in, n_steps_out)
    X_test, y_test = utils.split_sequences(test, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1, 1)

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], y.shape[1], y.shape[2], 1, 1))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], 1, 1))

    n_features_in = X.shape[2]
    n_features_out = y.shape[2]
    

    # define model
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (n_steps_in, n_features_in, 1, 1)))
    model.add(ConvLSTM2D(name ='conv_lstm_1',
                         filters = 64, kernel_size = (10, 1),                       
                         padding = 'same', 
                         return_sequences = True))
    
    model.add(Dropout(0.2, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(ConvLSTM2D(name ='conv_lstm_2',
                         filters = 64, kernel_size = (5, 1), 
                         padding='same',
                         return_sequences = False))
    
    model.add(Dropout(0.1, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(Flatten())
    model.add(RepeatVector(n_steps_out))
    model.add(Reshape((n_steps_out, n_features_out, 1, 64)))
    
    model.add(ConvLSTM2D(name ='conv_lstm_3',
                         filters = 64, kernel_size = (10, 1), 
                         padding='same',
                         return_sequences = True))
    
    model.add(Dropout(0.1, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))
    
    model.add(ConvLSTM2D(name ='conv_lstm_4',
                         filters = 64, kernel_size = (5, 1), 
                         padding='same',
                         return_sequences = True))
    
    model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))
    #model.add(Dense(units=1, name = 'dense_2'))

    optimizer = RMSprop() #lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)
    model.compile(loss = "mse", optimizer = optimizer)
 
    logdir = "logs/" + str(n_steps_in) + '_ConvLSTM2D_Deep' + line + '_'+str(epochs)+ '.log'
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = CSVLogger(logdir,separator=",", append=True)    

    # fit model    
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=128, validation_data = (X_test, y_test), callbacks=[early_stop, csv_logger])    
    model.save('models/n_steps_in_'+ str(n_steps_in) + '_ConvLSTM2D_Deep_' + line+'_'+str(epochs)+ '.h5')  
    
    return model

    
def train_DL_models(groupedDataScaled, samples_test, each_trip, links, n_steps_in, n_steps_out, epochs, time_print, machine):

    start_time = time.time()
    LSTM_model(groupedDataScaled[:-samples_test], each_trip, links, n_steps_in, n_steps_out, epochs)   

    if time_print == True:
        time_model = (time.time() - start_time)            
        print(machine + ',' + each_trip + ',' + str(len(links)) + ',' + str(n_steps_in) + ','+str(time_model) + ', LSTM,'+str(epochs))

    start_time = time.time()
    BidirectionalLSTM_model(groupedDataScaled[:-samples_test], each_trip, links, n_steps_in, n_steps_out, epochs)   

    if time_print == True:
        time_model = (time.time() - start_time)            
        print(machine + ',' + each_trip + ',' + str(len(links)) + ',' + str(n_steps_in) + ','+str(time_model) + ', BidirectionalLSTM, ' +str(epochs))        

    start_time = time.time()
    ConvLSTM1D_model(groupedDataScaled[:-samples_test], each_trip, links, n_steps_in, n_steps_out, epochs)   

    if time_print == True:
        time_model = (time.time() - start_time)
        print(machine + ',' + each_trip + ',' + str(len(links)) + ',' + str(n_steps_in) + ','+str(time_model) + ', ConvLSTM1D,'+str(epochs)) 
    start_time = time.time()

    ConvLSTM2D_model(groupedDataScaled[:-samples_test], each_trip, links, n_steps_in, n_steps_out, epochs)
    if time_print == True:
        time_model = (time.time() - start_time)
        print(machine + ','+ each_trip + ',' + str(len(links)) + ',' + str(n_steps_in) + ','+str(time_model) + ', ConvLSTM2D,'+str(epochs))

    start_time = time.time()
    ConvLSTM2D_Deep_model(groupedDataScaled[:-samples_test], each_trip, links, n_steps_in, n_steps_out,epochs)   
    if time_print == True:
        time_model = (time.time() - start_time)
        print(machine + ','+ each_trip + ',' + str(len(links)) + ',' + str(n_steps_in) + ','+str(time_model) + ', ConvLSTM2D-Deep,'+str(epochs))

