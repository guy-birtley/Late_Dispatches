
from datetime import datetime
import numpy as np
from sklearn.preprocessing import RobustScaler

def tprint(text):
    print(datetime.now(), text)


def scale(obs_list, mask = None, categorical_cols = 0, scaler = None):
    '''
    params: list of observations, number of categorical cols at start of observation, scaler - if scaler not provided will fit one
    function: stacks list into array dimension, fills nan with 0, scales with minmaxscaler from final dimension
    returns: scaled array and scaler
    '''
    x_all = np.stack(obs_list) #list to array
    x_all = np.nan_to_num(x_all, nan=0) #fill nans with 0

    x_cat = x_all[...,:categorical_cols]
    x_cont = x_all[...,categorical_cols:].astype('float32')

    orig_shape = x_cont.shape #remember original shape
    x2d = x_cont.reshape(-1, orig_shape[-1]) #make 2d for scale transform
    if scaler is None:
        #Robust scaler scales by iqr so stable with outliers and centred on 0
        scaler = RobustScaler(with_centering=False) #different scaler for each input
        #scaler = MinMaxScaler() #scale all inputs together to preserve relative 
        if mask is None:
            scaler.fit(x2d)
        else:
            scaler.fit(x2d[mask.reshape(-1).astype(bool)]) #ignore padded 0s if mask provided
    x2d = scaler.transform(x2d) #transform in 2d
    x_scaled = x2d.reshape(orig_shape) #return to original shape
    if categorical_cols > 0:
        x_scaled = np.concatenate([x_cat, x_scaled], axis=-1) #add back categorical columns if removed for scaling
    return x_scaled, scaler

def pad_temporal_in(temporal_in, temporal_length = 512): #pad temporal with 0s (with <30 transactions) so all observations are the same shape
    temporal_padded = np.zeros((len(temporal_in), temporal_length, temporal_in[0].shape[1]), dtype=np.float32) # Pre-allocate zero-filled array
    mask = np.zeros((len(temporal_in), temporal_length), dtype=bool)
    for i, seq in enumerate(temporal_in):
        if seq is None: #ignore empty rows
            continue
        temporal_padded[i, -len(seq):, :] = seq #replace 0s with observation data
        mask[i, -len(seq):] = True
    return temporal_padded, mask