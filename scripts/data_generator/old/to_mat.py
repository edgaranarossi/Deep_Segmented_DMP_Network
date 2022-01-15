#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:53:54 2021

@author: edgar
"""

import json
import scipy.io as sio
from PIL import Image, ImageOps
from os.path import join
from os import listdir
import numpy as np
# from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.data.smnist_loader import Mapping
from time import sleep
import pickle as pkl

ROOT_DIR = '/home/edgar/rllab/scripts/dmp/data/'
traj_name = 'distance_10_orientation_0_height_25_step_5'
pkl_dir = join(ROOT_DIR, 'pkl/', traj_name)
img_json_dir = join(ROOT_DIR, 'img_json/')
traj_dir = join(img_json_dir, traj_name, 'trajectories/')
img_dir = join(img_json_dir, traj_name, 'image/')
img_50_dir = join(img_dir, '50x50')
# img_500_dir = join(img_dir, '500x500')
files = range(40000)

while len(listdir(traj_dir)) < len(files):
    print("Data :", len(listdir(traj_dir)), '/', len(files))
    sleep(5)
    
#%
images = []
lengths = []
cuts = []
additional_heights = []
distances = []
orientations = []
traj_steps = []
trajs = []
# direct_trajs = []
dmp_trajs = []
# lowered_dmp_trajs = []
captions = []
outputs = []
dt = 0

for f in files:
    with Image.open(join(img_50_dir, 'Image_'+str(f)+'.jpg')) as im:
        images.append(np.array(ImageOps.grayscale(im)))
        
    data = json.loads(open(join(traj_dir, 'Image_'+str(f)+'.json')).read())
    
    if (int(data['Image number'])+1)%100 == 0:
        print("Reading", int(data['Image number'])+1, '/', len(files))
    
    lengths.append(data['Carrot length'])
    cuts.append(data['Cut number'])
    additional_heights.append(data['Additional height'])
    distances.append(data['Cut distance'])
    orientations.append(data['Orientation'])    
    traj_steps.append(data['Trajectory step'])
    
    traj = [i.split(', ') for i in data['Trajectory'][1:-1].split('], [') if i != '']
    traj = np.array(traj, dtype = 'float64')
    trajs.append(traj)
    
    # direct_traj = [i.split(', ') for i in data['Direct trajectory'][1:-1].split('], [') if i != '']
    # direct_traj = np.array(direct_traj, dtype = 'float64')
    # direct_trajs.append(direct_traj)
    
    dmp_traj = [i.split(', ') for i in data['DMP trajectory'][1:-1].split('], [') if i != '']
    dmp_traj = np.array(dmp_traj, dtype = 'float64')
    dmp_trajs.append(dmp_traj)
    
    # lowered_dmp_traj = [i.split(', ') for i in data['Lowered DMP traj'][1:-1].split('], [') if i != '']
    # lowered_dmp_traj = np.array(lowered_dmp_traj, dtype = 'float64')
    # lowered_dmp_trajs.append(lowered_dmp_traj)
    
    dmp_tau = data['DMP tau']
    dmp_y0 = [float(i) for i in data['DMP y0'][1:-1].split(', ')]
    dmp_goal = [float(i) for i in data['DMP goal'][1:-1].split(', ')]
    
    dmp_weight = [i.split(', ') for i in data['DMP weight'][1:-1].split('], [') if i != '']
    dmp_weight = np.array(dmp_weight, dtype = 'float64')
    
    output = np.array([dmp_tau])
    output = np.append(output, np.array(dmp_y0))
    output = np.append(output, np.array(dmp_goal))
    output = np.append(output, np.array(dmp_weight).reshape(-1))
    outputs.append(output)
    
    dt = data["DMP dt"]
    
    captions.append(data['Caption'])
outputs = np.array(outputs)
#%%
# trajs = np.array(trajs, dtype='float64')
# direct_trajs = np.array(direct_trajs, dtype='float64')
# dmp_trajs = np.array(dmp_trajs, dtype='float64')
# lowered_dmp_trajs = np.array(lowered_dmp_trajs, dtype='float64')
#%%
# trajectories = or_traj[:]
# # max_len = 0
# # for traj in trajectories:
# #     if traj.shape[0] > max_len:
# #         max_len = traj.shape[0]
# # #%
# # for traj in range(len(trajectories)):
# #     if trajectories[traj].shape[0] < max_len:
# #         last_state = trajectories[traj][-1].reshape((1,-1))
# #         while trajectories[traj].shape[0] != max_len:
# #             last_state[0][-1] += 1
# #             trajectories[traj] = np.append(trajectories[traj], last_state, axis=0)
# #             # print(traj.shape)
# # #%
# images = np.array(images)
# images = images.reshape(images.shape[0], -1, images.shape[1], images.shape[2])
# trajectories_np = np.array(trajectories, dtype='float64')

# #%%
# N = 50
# sampling_time = 0.01
# # tau = 3
# dmps = Trainer.create_dmps(trajectories_np, N, sampling_time)
#%%
# outputs = []
# for dmp in dmps:
#     tau = dmp.tau[0]
#     w = dmp.w
#     goal = dmp.goal
#     y0 = dmp.y0
#     # dy0 = np.array([0,0])
#     learn = np.append(tau, y0)
#     learn = np.append(learn, goal)
#     learn = np.append(learn, w)
#     outputs.append(learn)
    
# outputs = np.array(outputs)
#%%
print("Calculating output scale")
y_max = 1
y_min = -1
x_max = np.array([outputs[:, i].max() for i in range(0, 5)])
x_max = np.concatenate((x_max, np.array([outputs[:, 5:outputs.shape[1]].max() for i in range(5, outputs.shape[1])])))
x_min = np.array([outputs[:, i].min() for i in range(0, 5)])
x_min = np.concatenate((x_min, np.array([outputs[:, 5:outputs.shape[1]].min() for i in range(5, outputs.shape[1])])))
scale = x_max-x_min
scale[np.where(scale == 0)] = 1
scaled_outputs = (y_max - y_min) * (outputs-x_min) / scale + y_min
# print(outputs[0])

scaling = Mapping()
scaling.x_max = x_max
scaling.x_min = x_min
scaling.y_max = y_max
scaling.y_min = y_min
#%%
data = {
        "image"            : images,
        "caption"          : captions,
        "traj"             : trajs,
        # "direct_traj"      : direct_trajs,
        "dmp_traj"         : dmp_trajs,
        # "lowered_dmp_traj" : lowered_dmp_trajs,
        "scaled_outputs"   : scaled_outputs,
        # "unscaled_outputs" : outputs,
        "scaling"          : scaling,
        "dt"               : dt,
        "cut_distance"     : distances
        }
filename = join(pkl_dir, 'carrot_' + traj_name + '_grayscale_DMP_BF_' +\
                str(dmp_weight.shape[0]) + '_dt_'+str(dt) + '.pkl')
print("Saving", filename)
# sio.savemat(filename, data)
pkl.dump(data, open(filename, "wb"))