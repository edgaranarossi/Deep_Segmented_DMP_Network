#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 02:29:01 2022

@author: edgar
"""

from os import listdir
from os.path import join, isdir
import pandas as pd

net_desc_name = 'network_description.txt'
model_dir = '/home/edgar/rllab/scripts/dmp/Segmented-Deep-DMPs/models'
models = [i for i in listdir(model_dir) if isdir(join(model_dir,i)) and
                                           'Fixed' in i and
                                           net_desc_name in listdir(join(model_dir,i))]
models = sorted(models)

descs = []
for model in models:
    with open(join(model_dir, model, net_desc_name)) as f:
        lines = f.readlines()
    descs.append(lines)
    
valid_models = [i for i in descs if len(i) > 0 and
                                    # 'Best Validation Loss' in i[-1] and
                                    ' :: Epoch : ' not in i[-1] and
                                    'Network created' in i[1]]

parsed = []
for model in valid_models:
    parsed.append([model[1].split(': ')[-1][:-1], float(model[-1].split(' : ')[-1])])
df = pd.DataFrame(parsed)
#%%
models_with_img = [i for i in listdir(model_dir) if isdir(join(model_dir,i)) and
                                                    'Integrator' in i and
                                                    net_desc_name in listdir(join(model_dir,i)) and
                                                    len([j for j in listdir(join(model_dir,i)) if 'png' in j]) > 0]
models_with_img = sorted(models_with_img)

descs = []
for model in models_with_img:
    with open(join(model_dir, model, net_desc_name)) as f:
        lines = f.readlines()
    descs.append(lines)

valid_models = [i for i in descs if 'Best Validation Loss' in i[-1] and
                                    'tensor' not in i[-1] and
                                    'Network created' in i[1]]

parsed = []
for model in valid_models:
    parsed.append([model[1].split(': ')[-1][:-1], 
                   model[3].split(' : ')[-1].split('/')[-1].replace('generated\\', ''), 
                   float(model[-1].split(' : ')[-1])])
df = pd.DataFrame(parsed)

#%%
import pickle as pkl

model_name = 'Model_FixedSegmentDictDMPNet_2022-01-30_20-37-33'
train_param = pkl.load(open(join(model_dir, model_name, 'train-model-dmp_param.pkl'), 'rb'))
model_param = train_param.model_param
dmp_param = model_param.dmp_param