#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:14:42 2021

@author: edgar
"""
from utils.dataset_importer import MatDataLoader
from utils.networks import CNNDMPNet
from utils.trainer import Trainer
from parameters import TrainingParameters, ModelParameters

from os.path import join, isdir, dirname
from os import getcwd
from torchsummary import summary
import numpy as np

ROOT_DIR = dirname(getcwd())
FILE_NAME = 'carrot_50x50_grayscale_dmp_n_50_dt_0.01.mat'
FILE_PATH = join(ROOT_DIR, 'data', FILE_NAME)
#%%
train_param = TrainingParameters()
model_param = ModelParameters()
mat_data_loader = MatDataLoader(FILE_PATH)
data_loaders, scale = \
    mat_data_loader.getDataLoader(data_ratio = train_param.data_ratio,
                                  batch_size = train_param.batch_size)
inputs, outputs = mat_data_loader.getData()
tau = mat_data_loader.getTau()
model_param.scale = scale
#%%
model = CNNDMPNet(model_param)
summary(model, model_param.image_dim)
#%%
trainer = Trainer(model, train_param)
# trainer.train(model, data_loaders)
test_data = [i for i in data_loaders[2]]
predictions, losses = trainer.getLosses(data_loaders[2], train = False)

scaled_output = np.array(outputs[0]['outputs'].cpu())
traj = np.array(outputs[0]['trajectory'].cpu())

output = (scaled_output - scale.y_min) * (scale.x_max[1:] - scale.x_min[1:])/(scale.y_max - scale.y_min) + scale.x_min[1:]
