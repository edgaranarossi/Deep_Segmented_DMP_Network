#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 02:29:01 2022

@author: edgar
"""

from os import listdir
from os.path import join, isdir
import pandas as pd
import pickle as pkl
#%%

net_desc_name = 'network_description.txt'
model_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'
# model_dir = 'D:\\rllab\\scripts\\dmp\\Segmented_Deep_DMPs\\models'
models = [i for i in listdir(model_dir) if isdir(join(model_dir,i)) and
                                           'DSDNet' in i and
                                           net_desc_name in listdir(join(model_dir,i))]
models = sorted(models)

descs = []
for model in models:
    with open(join(model_dir, model, net_desc_name)) as f:
        lines = f.readlines()
    descs.append([model, lines])
    
valid_models = [i for i in descs if len(i[1]) > 0 and
                                    'Final Validation Loss' in i[1][-2] and
                                    ' :: Epoch : ' not in i[1][-1] and
                                    'Network created' in i[1][1]]

parsed = []
for model in valid_models:
    parsed.append([model[0], model[1][1].split(': ')[-1][:-1], float(model[1][-1].split(' : ')[-1])])
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

model_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/DSDNet'
# model_dir = 'D:\\rllab\\scripts\\dmp\\Segmented_Deep_DMPs\\models\\SegmentDMPCNN'
model_name = 'Model_SegmentDMPCNN_2022-04-06_13-42-39'
train_param = pkl.load(open(join(model_dir, model_name, 'train-model-dmp_param.pkl'), 'rb'))
model_param = train_param.model_param
dmp_param = model_param.dmp_param

#%%
model_root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'
model_name = 'Model_DSDNet_2022-07-28_13-22-12'
data_path = join(model_root_dir, model_name.split('_')[1], model_name, 'data_loaders.pkl')
data_loaders = pkl.load(open(data_path,'rb'))

train_loader, val_loader, test_loader = data_loaders
train_dataset = train_loader.dataset
val_dataset = val_loader.dataset
test_dataset = test_loader.dataset

#%%
import numpy as np
import pickle as pkl
from os import listdir
from os.path import join, isdir

data_root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/stacking'
data_name = 'stacking_[1, 2, 3][num-data-9000][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-09-09_05-20-31].pkl'
# data_name = 'stacking_[1, 2, 3][num-data-450][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-09-11_00-44-55].pkl'
# data_name = 'stacking_[1, 2, 3].num_data_15000_num_seg_24.normal_dmp_bf_48_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.024_2022-07-31_18-07-28.pkl'

data_root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/cutting'
# data_name = 'cutting_100.num_data_100_num_seg_15.normal_dmp_bf_300_ay_100_dt_0.001.seg_dmp_bf_20_ay_7_dt_0.015_2022-09-11_06-56-12.pkl'
# data_name = 'rotated_real_distanced_trajectory.num_data_54_num_seg_3.normal_dmp_bf_250_ay_25_dt_0.001.seg_dmp_bf_50_ay_15_dt_0.003.2022-09-12_02-17-27.pkl'
data_name = 'cutting_5000.num_data_5000_num_seg_15.normal_dmp_bf_300_ay_100_dt_0.001.seg_dmp_bf_20_ay_7_dt_0.2022-09-12_03-15-37.pkl'
data_path = join(data_root_dir, data_name)
dataset = pkl.load(open(data_path, 'rb'))

print('Dataset name\n{}\n'.format(data_name))

print('Normal DMP data')
print('y0:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_y0'].min(),
                                                 dataset['normal_dmp_y0'].max(),
                                                 np.abs(dataset['normal_dmp_y0'].max() - dataset['normal_dmp_y0'].min())))
print('goal:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_goal'].min(),
                                                   dataset['normal_dmp_goal'].max(),
                                                   np.abs(dataset['normal_dmp_goal'].max() - dataset['normal_dmp_goal'].min())))
print('w:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_w'].min(),
                                                dataset['normal_dmp_w'].max(),
                                                np.abs(dataset['normal_dmp_w'].max() - dataset['normal_dmp_w'].min())))

if 'normal_dmp_L_y0' in dataset:
    print('\nNormal DMP L data')
    print('y0:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_L_y0'].min(),
                                                     dataset['normal_dmp_L_y0'].max(),
                                                     np.abs(dataset['normal_dmp_L_y0'].max() - dataset['normal_dmp_L_y0'].min())))
    print('goal:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_L_goal'].min(),
                                                       dataset['normal_dmp_L_goal'].max(),
                                                       np.abs(dataset['normal_dmp_L_goal'].max() - dataset['normal_dmp_L_goal'].min())))
    print('w:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_L_w'].min(),
                                                    dataset['normal_dmp_L_w'].max(),
                                                    np.abs(dataset['normal_dmp_L_w'].max() - dataset['normal_dmp_L_w'].min())))

print('\nSegmented DMP data')
print('y0:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['segmented_dmp_y0'].min(),
                                                 dataset['segmented_dmp_y0'].max(),
                                                 np.abs(dataset['segmented_dmp_y0'].max() - dataset['segmented_dmp_y0'].min())))
print('goal:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['segmented_dmp_goal'].min(),
                                                   dataset['segmented_dmp_goal'].max(),
                                                   np.abs(dataset['segmented_dmp_goal'].max() - dataset['segmented_dmp_goal'].min())))
print('w:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['segmented_dmp_w'].min(),
                                                dataset['segmented_dmp_w'].max(),
                                                np.abs(dataset['segmented_dmp_w'].max() - dataset['segmented_dmp_w'].min())))

#%%
from datetime import datetime

new_data_name = data_name.split('.')
new_data_name[-2] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_data_name = '.'.join(new_data_name)

pkl.dump(dataset, open(join(data_root_dir, new_data_name), 'wb'))