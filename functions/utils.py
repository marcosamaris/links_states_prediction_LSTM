#!/usr/bin/python

import numpy as np
import pandas as pd 
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def clean_group_data(dataset, links, range_upper, frequency):
    q1, q3 = np.percentile(dataset['ltt'],[25,75])
    iqr = q3 - q1
    lower_bound = 10
    upper_bound = q3 + range_upper*(1.5 * iqr)  

    # Deleting lower bound and upper bound from the dataset LinkTT2
    dataset = dataset.loc[(dataset['ltt'] >= lower_bound) & 
                                              (dataset['ltt'] <= upper_bound)]

    dataset.index = dataset['aproxlinkstart']

    groupedData = dataset.groupby([pd.Grouper(freq='30min'), 'link'], sort=False)['ltt'].mean().unstack().reset_index(
                        ).set_index('aproxlinkstart').resample('30min').mean().transform(
                        lambda x: x.fillna(method='ffill')).transform(
                        lambda x: x.fillna(method='backfill')).transform(
                        lambda x: x.fillna(method='pad')).dropna()


    return groupedData


def linesedge_tuples2string(links_tuples):
    links = []
    for i in links_tuples:
        links.append('(' + str(i[0]) + ', ' + str(i[1]) + ')')
    return links
        
