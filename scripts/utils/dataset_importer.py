from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import torch
from torch import from_numpy
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
import pickle
import sys
import psutil
import random

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max

class Scaler:
    def __init__(self, data, range_new = [-1.0, 1.0]):
        self.range_old = MinMax(min = data.min(), max = data.max())
        self.range_new = MinMax(min = range_new[0], max = range_new[1])
        self.old_data = data
        self.normalized_data = self.normalize()

    def normalize(self):
        normalized_data = (self.range_new.max - self.range_new.min) * \
                          (self.old_data      - self.range_old.min) / \
                          (self.range_old.max - self.range_old.min) + \
                           self.range_new.min
        return normalized_data

    def denormalize(self, X):
        denormalized_data = (X - self.range_new.min) / \
                            (self.range_new.max - self.range_new.min) * \
                            (self.range_old.max - self.range_old.min) + \
                             self.range_old.min
        return denormalized_data

def ndarray_to_str(arr):
    assert len(arr.shape) == 2
    s = ''
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            s += str(arr[i, j])
            s += ','
        s = s[:-1]
        s += ';'
    return s


class PickleDataLoader:
    def __init__(self, train_param):
        self.train_param        = train_param
        self.memory_limit       = self.train_param.memory_percentage_limit
        self.model_param        = self.train_param.model_param
        self.data_limit         = self.train_param.data_limit
        self.keys_to_normalize  = self.model_param.keys_to_normalize
        self.input_mode         = self.model_param.input_mode
        self.output_mode        = self.model_param.output_mode
        self.data               = pickle.load(open(train_param.dataset_path, 'rb'))
        self.scaler             = {}
        self.data_length        = None if self.data_limit == None else self.data_limit
        self.max_segments       = None
        self.combined_inputs    = []
        self.combined_outputs   = []

        missing_keys = []
        for key in self.input_mode:
            if key not in self.data: missing_keys.append(key)
        for key in self.output_mode:
            if key not in self.data: missing_keys.append(key)
        assert len(missing_keys) == 0, str(missing_keys) + ' missing from dataset'

        for key in self.keys_to_normalize:
            self.scaler[key] = Scaler(self.data[key])
            self.data[key] = self.scaler[key].normalized_data
        
        for key in self.data:
            if isinstance(self.data[key], list) or isinstance(self.data[key], ndarray):
                self.data[key] = self.data[key][:self.data_limit]
                if self.data_length == None:
                    if isinstance(self.data[key], list):
                        self.data_length = len(self.data[key])
                    elif isinstance(self.data[key], ndarray):
                        self.data_length = self.data[key].shape[0]
            if isinstance(self.data[key], ndarray) and len(self.data[key].shape) == 1:
                self.data[key] = self.data[key].reshape(-1, 1)
            if key == 'image':
                if self.data[key].max() > 1:
                    self.data[key] = self.data[key] / 255
                if len(self.data[key].shape) == 3:
                    self.data[key] = self.data[key].reshape(self.data[key].shape[0], 1, self.data[key].shape[1], self.data[key].shape[2])
            if key == 'max_segments':
                assert type(self.data[key]) == int
                self.max_segments = self.data[key]
        
        for idx in range(self.data_length):
            if psutil.virtual_memory().percent > self.memory_limit - 5:
                raise MemoryError('Out of Memory (>{}%)'.format(self.memory_limit - 5))

            inputs = {}
            for key in self.input_mode:
                if isinstance(self.data[key], ndarray):
                    inputs[key] = from_numpy(self.data[key][idx]).float().to(DEVICE)
                else:
                    inputs[key] = self.data[key][idx]
            self.combined_inputs.append(inputs)
            
            outputs = {}
            for key in self.data:
                if isinstance(self.data[key], ndarray) and key != 'original_trajectory':
                    outputs[key] = from_numpy(self.data[key][idx]).float().to(DEVICE)
                elif not isinstance(self.data[key], float) and not isinstance(self.data[key], int) and not isinstance(self.data[key], tuple):
                    if key == 'original_trajectory':
                        outputs[key] = ndarray_to_str(self.data[key][idx])
                    else:
                        outputs[key] = self.data[key][idx]
                else: 
                    outputs[key] = self.data[key]
            self.combined_outputs.append(outputs)

        if self.train_param.shuffle_data:
            combined_data = list(zip(self.combined_inputs, self.combined_outputs))
            random.shuffle(combined_data)
            self.combined_inputs, self.combined_outputs = zip(*combined_data)

    def getData(self):
        return self.combined_inputs, self.combined_outputs

    def getDataLoader(self, data_ratio = [7, 2, 1], batch_size = 50):
        
        X_train, X_val, Y_train, Y_val  = train_test_split(self.combined_inputs,
                                                           self.combined_outputs,
                                                           test_size=(data_ratio[1]+data_ratio[2])/sum(data_ratio))
        X_val, X_test, Y_val, Y_test    = train_test_split(X_val,
                                                           Y_val, 
                                                           test_size=data_ratio[2]/(data_ratio[1]+data_ratio[2]))

        train_dataset                   = DMPDataset(X = X_train, Y = Y_train)
        val_dataset                     = DMPDataset(X = X_val, Y = Y_val)
        test_dataset                    = DMPDataset(X = X_test, Y = Y_test)

        train_loader                    = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader                      = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)
        test_loader                     = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

        if self.scaler != {}:
            scaling = self.scaler
        else:
            scaling = None

        return [train_loader, val_loader, test_loader], scaling
    
class DMPDataset(Dataset):
    def __init__(self, X, Y = None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = self.X[idx]
        if self.Y != None: 
            labels = self.Y[idx]
            return (inputs, labels)
        return inputs