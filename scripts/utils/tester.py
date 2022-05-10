#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:44:42 2022

@author: edgar
"""
import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from numpy.random import randint
from utils.networks import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester:
    def __init__(self, dataloaders: [DataLoader, DataLoader, DataLoader]):
        self.train_loaders = dataloaders[0]
        self.val_loaders = dataloaders[1]
        self.test_loaders = dataloaders[2]
        self.train_dataset = self.train_loaders.dataset
        self.val_dataset = self.val_loaders.dataset
        self.test_dataset = self.test_loaders.dataset
        
        self.read_dataset()
    
    def read_dataset(self):
        print('Dataset lengths:')
        print('- Train dataset: {}'.format(len(self.train_dataset)))
        print('- Val dataset  : {}'.format(len(self.val_dataset)))
        print('- Test dataset : {}'.format(len(self.test_dataset)))
        print('Total          : {}\n'.format(len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)))
        print('Available inputs : {}'.format([i for i in self.train_dataset[0][0]]))
        print('Available outputs: {}'.format([i for i in self.train_dataset[0][1]]))
        
        if len([i for i in self.train_dataset[0][0]]) == 1:
            self.input_mode = [i for i in self.train_dataset[0][0]][0]
        else:
            print('Set input mode with set_input_mode')
            
    def set_input_mode(self, mode):
        self.input_mode = mode
        
    def predict_test(self, model, idx = None):
        scaler = model.train_param.scaler
        if idx == None: idx = randint(len(self.test_dataset))
        data = self.test_dataset[idx][0][self.input_mode].to(DEVICE)
        data = torch.unsqueeze(data, 0).to(DEVICE)
        pred = model(data)
        
        rescaled_pred = []
        for i, key in enumerate(mpdel.model_param.keys_to_normalize):
            rescaled_pred_segment.append(scaler[key].denormalize(pred[i]))
            
        return rescaled_pred

if __name__=='__main__':
    MODEL_ROOT = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/'
    MODEL_SUB_ROOT = 'SegmentedDMPJoinedNetwork'
    MODEL_NAME = 'Model_SegmentedDMPJoinedNetwork_2022-05-10_00-55-25'
    
    assert exists(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME))
    data_loaders = pkl.load(open(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME, 'data_loaders.pkl'), 'rb'))    
    train_param_segment = pkl.load(open(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME, 'train-model-dmp_param.pkl'), 'rb'))
    tester = Tester(data_loaders)
    model_segmented = SegmentedDMPJoinedNetwork(train_param_segment).to(DEVICE)
    tester.predict_test(model_segmented)