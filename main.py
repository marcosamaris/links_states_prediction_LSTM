#!/usr/bin/python


import warnings
warnings.filterwarnings('ignore')

import sys
import os 
sys.path.append(os.getcwd())

import time
import pickle

from functions import train_DLmodels as  train
from functions import utils as utils
from functions import areas_data_structure as ads

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

with open('data/linesedges.pkl', 'rb') as input_file:
    linesedge = pickle.load(input_file)


time_print = True
n_steps_out = 3       
machine = 'Personal_Desktop'
epochs = 1

save = True
samples_test = 252
        
for n_steps_in in [ 16, 24, 32]:
    for i in [1,2,4,5,6]:

        each_trip = str(i)
        
        with open('data/linktt/links_area_' + each_trip + '.pkl', 'rb') as input_file:
            dataset = pickle.load(input_file)

        links = dataset['link'].unique()
        
        groupedData = utils.clean_group_data(dataset, links, 3, '30min')
        groupedData= groupedData[links].loc[(groupedData.index.hour > 5)]     
        
        # define input sequence
        scaler_std = StandardScaler()
        scaler = MinMaxScaler()        
        groupedDataScaled = scaler_std.fit_transform(np.log(groupedData.values))        
        groupedDataScaled = scaler.fit_transform(groupedDataScaled)        

        train.train_DL_models(groupedDataScaled, samples_test, each_trip, links, n_steps_in, n_steps_out, epochs, time_print, machine)


        for each_trip in ads.a[i]['trips'][:1]:
               
            with open('data/linktt/trip_' + each_trip + '.pkl', 'rb') as input_file:
                dataset = pickle.load(input_file)


            links_tuples = linesedge[each_trip]
            links = utils.linesedge_tuples2string(links_tuples)
        
        
            groupedData = utils.clean_group_data(dataset, links, 3, '30min')
            groupedData= groupedData[links].loc[(groupedData.index.hour > 5)]     
            
            # define input sequence
            scaler_std = StandardScaler()
            scaler = MinMaxScaler()        
            groupedDataScaled = scaler_std.fit_transform(np.log(groupedData.values))        
            groupedDataScaled = scaler.fit_transform(groupedDataScaled)        

            train.train_DL_models(groupedDataScaled, samples_test, each_trip, links, n_steps_in, n_steps_out, epochs, time_print, machine)
        
        



