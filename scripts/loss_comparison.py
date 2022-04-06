#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:48:30 2022

@author: edgar
"""

import torch
from torch import tensor, cat, cdist, ones, zeros, nn, from_numpy
import pickle as pkl
import numpy as np
from utils.pydmps_torch import DMPs_discrete_torch
from utils.networks import SegmentDMPCNN, CNNDeepDMP
from os.path import join
from matplotlib import pyplot as plt

data_path = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/cutting_traj/image_num-seg_y0_goals_ws_N_1000+seg=n-bf_20_ay_7_dt0.02_max-seg_15+cut=dist_0.2_top-pad_0.2_side-pad_0.05_normal-dmp_limited_y_2022-03-30_15-01-55.pkl'
data = pkl.load(open(data_path, 'rb'))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir_segment = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/SegmentDMPCNN/Model_SegmentDMPCNN_2022-03-24_12-48-45'
model_dir_deep_dmp = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/CNNDeepDMP/Model_CNNDeepDMP_2022-03-23_12-41-10'

train_param_segment = pkl.load(open(join(model_dir_segment, 'train-model-dmp_param.pkl'), 'rb'))
model_param_segment = train_param_segment.model_param
dmp_param_segment = model_param_segment.dmp_param
train_param_deep_dmp = pkl.load(open(join(model_dir_deep_dmp, 'train-model-dmp_param.pkl'), 'rb'))
model_param_deep_dmp = train_param_deep_dmp.model_param
dmp_param_deep_dmp = model_param_deep_dmp.dmp_param

model_segment = SegmentDMPCNN(train_param_segment).to(DEVICE)
model_segment.load_state_dict(torch.load(join(model_dir_segment, 'best_net_parameters')))
model_deep_dmp = CNNDeepDMP(train_param_deep_dmp).to(DEVICE)
model_deep_dmp.load_state_dict(torch.load(join(model_dir_deep_dmp, 'best_net_parameters')))

#%%
img_np = np.flipud(data['image'][0])
img = from_numpy(np.flipud(img_np).copy()).to(DEVICE).float().reshape(1, 1, 50, 50) / 255

pred_segment = model_segment(img)
pred_deep_dmp = model_deep_dmp(img)

rescaled_pred_segment = []
for idx, key in enumerate(model_param_segment.keys_to_normalize):
    rescaled_pred_segment.append(dmp_param_segment.scale[key].denormalize(pred_segment[idx][0]))

rescaled_pred_deep_dmp = []
for idx, key in enumerate(model_param_deep_dmp.keys_to_normalize):
    rescaled_pred_deep_dmp.append(dmp_param_deep_dmp.scale[key].denormalize(pred_deep_dmp[idx][0]))
    
num_segments_pred = int(torch.round(rescaled_pred_segment[0]).reshape(1).item())

y_pred = zeros(num_segments_pred, int(1 / dmp_param_segment.dt), dmp_param_segment.dof).to(DEVICE)
all_pos_pred = cat([rescaled_pred_segment[1].reshape(1, dmp_param_segment.dof, 1), rescaled_pred_segment[2].reshape(-1, dmp_param_segment.dof, 1)], dim = 0)
y0s_pred = all_pos_pred[:-1]
goals_pred = all_pos_pred[1:]
dmp_pred = DMPs_discrete_torch(n_dmps = dmp_param_segment.dof, 
                                                   n_bfs = dmp_param_segment.n_bf, 
                                                   ay = dmp_param_segment.ay, 
                                                   dt = dmp_param_segment.dt)
# dmp_pred.y0 = rescaled_pred[1].reshape(1, self.dmp_param.dof, 1)
dmp_pred.y0         = y0s_pred[:num_segments_pred]
dmp_pred.goal       = goals_pred[:num_segments_pred]
dmp_pred.w          = rescaled_pred_segment[3][:num_segments_pred].reshape(num_segments_pred, dmp_param_segment.dof, dmp_param_segment.n_bf)
y_track_pred, _, _  = dmp_pred.rollout()

y_pred = y_track_pred.reshape(-1, dmp_param_segment.dof)

padding = 3
multiplier = 28
y_pred = ((y_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, dmp_param_segment.dof)

plt.scatter(y_pred[:, 0], y_pred[:, 1])
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.imshow(img_np, cmap = 'Greys_r')
plt.show()


num_segments_pred = 1
y_pred = zeros(num_segments_pred, int(1 / dmp_param_deep_dmp.dt), dmp_param_deep_dmp.dof).to(DEVICE)
all_pos_pred = cat([rescaled_pred_deep_dmp[0].reshape(1, dmp_param_deep_dmp.dof, 1), rescaled_pred_deep_dmp[1].reshape(-1, dmp_param_deep_dmp.dof, 1)], dim = 0)
y0s_pred = all_pos_pred[:-1]
goals_pred = all_pos_pred[1:]
dmp_pred = DMPs_discrete_torch(n_dmps = dmp_param_deep_dmp.dof, 
                                                   n_bfs = dmp_param_deep_dmp.n_bf, 
                                                   ay = dmp_param_deep_dmp.ay, 
                                                   dt = dmp_param_deep_dmp.dt)
# dmp_pred.y0 = rescaled_pred[1].reshape(1, self.dmp_param.dof, 1)
dmp_pred.y0         = y0s_pred[:num_segments_pred]
dmp_pred.goal       = goals_pred[:num_segments_pred]
dmp_pred.w          = rescaled_pred_deep_dmp[2][:num_segments_pred].reshape(num_segments_pred, dmp_param_deep_dmp.dof, dmp_param_deep_dmp.n_bf)
y_track_pred, _, _  = dmp_pred.rollout()
y_pred = y_track_pred.reshape(-1, dmp_param_deep_dmp.dof)
y_pred = ((y_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, dmp_param_deep_dmp.dof)

plt.scatter(y_pred[:, 0], y_pred[:, 1])
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.imshow(img_np, cmap = 'Greys_r')
plt.show()

#%%
img = from_numpy(data['image'] / 255).to(DEVICE).float().reshape(1000, 1, 50, 50)

pred_segment = model_segment(img)
pred_deep_dmp = model_deep_dmp(img)

rescaled_pred_segment = []
for idx, key in enumerate(model_param_segment.keys_to_normalize):
    rescaled_pred_segment.append(dmp_param_segment.scale[key].denormalize(pred_segment[idx]))

rescaled_pred_deep_dmp = []
for idx, key in enumerate(model_param_deep_dmp.keys_to_normalize):
    rescaled_pred_deep_dmp.append(dmp_param_deep_dmp.scale[key].denormalize(pred_deep_dmp[idx]))
#%%
all_pos_pred = cat([rescaled_pred_segment[1].reshape(-1, 1, dmp_param_segment.dof, 1), rescaled_pred_segment[2].reshape(-1, model_param_segment.max_segments, dmp_param_segment.dof, 1)], dim = 1)
y0s_pred = all_pos_pred[:, :-1]
goals_pred = all_pos_pred[:, 1:]
dmp_pred = DMPs_discrete_torch(n_dmps = dmp_param_segment.dof, 
                                                   n_bfs = dmp_param_segment.n_bf, 
                                                   ay = dmp_param_segment.ay, 
                                                   dt = dmp_param_segment.dt)
# dmp_pred.y0 = rescaled_pred[1].reshape(1, self.dmp_param.dof, 1)
dmp_pred.y0         = y0s_pred.reshape(-1, dmp_param_segment.dof, 1)
dmp_pred.goal       = goals_pred.reshape(-1, dmp_param_segment.dof, 1)
dmp_pred.w          = rescaled_pred_segment[3].reshape(-1, dmp_param_segment.dof, dmp_param_segment.n_bf)
y_track_pred_segment, _, _  = dmp_pred.rollout()
y_track_pred_segment = y_track_pred_segment.reshape(1000, model_param_segment.max_segments, -1, dmp_param_segment.dof)


all_pos_pred = cat([rescaled_pred_deep_dmp[0].reshape(-1, 1, dmp_param_deep_dmp.dof, 1), rescaled_pred_deep_dmp[1].reshape(-1, model_param_deep_dmp.max_segments, dmp_param_deep_dmp.dof, 1)], dim = 1)
y0s_pred = all_pos_pred[:, :-1]
goals_pred = all_pos_pred[:, 1:]
dmp_pred = DMPs_discrete_torch(n_dmps = dmp_param_deep_dmp.dof, 
                                                   n_bfs = dmp_param_deep_dmp.n_bf, 
                                                   ay = dmp_param_deep_dmp.ay, 
                                                   dt = dmp_param_deep_dmp.dt)
# dmp_pred.y0 = rescaled_pred[1].reshape(1, self.dmp_param.dof, 1)
dmp_pred.y0         = y0s_pred.reshape(-1, dmp_param_deep_dmp.dof, 1)
dmp_pred.goal       = goals_pred.reshape(-1, dmp_param_deep_dmp.dof, 1)
dmp_pred.w          = rescaled_pred_deep_dmp[2].reshape(-1, dmp_param_deep_dmp.dof, dmp_param_deep_dmp.n_bf)
y_track_pred_deep_dmp, _, _  = dmp_pred.rollout()
y_track_pred_deep_dmp = y_track_pred_deep_dmp.reshape(1000, 1, -1, dmp_param_segment.dof)
#%%
y_track_pred_segment_limited = []
for i in range(y_track_pred_segment.shape[0]):
    num_segment = int(torch.round(rescaled_pred_segment[0][i, 0]).item())
    y_track_pred_segment_limited.append(np.array((y_track_pred_segment[i, :num_segment].reshape(-1, dmp_param_segment.dof)).detach().cpu()))

y_track_pred_segment_np = np.array(y_track_pred_segment.detach().cpu())
y_track_pred_deep_dmp_np = np.array(y_track_pred_deep_dmp.detach().cpu()).reshape(1000, 1000, 2)

preds = {'y_track_pred_segment': y_track_pred_segment_np,
         'y_track_pred_segment_limited':y_track_pred_segment_limited,
         'y_track_pred_deep_dmp':y_track_pred_deep_dmp_np,
         'y_track_label': data['dmp_traj']}
pkl.dump(preds, open('/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/loss_preds.pkl', 'wb'))
#%%
import pickle as pkl
import numpy as np
from scipy.spatial import distance

to_compare = pkl.load(open('/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/loss_preds.pkl', 'rb'))

def compare_closest(y_label, y_pred):
    loss = distance.cdist(y_label, y_pred, 'euclidean').min(axis = 1).sum() / len(y_label)
    return loss

y_labels = to_compare['y_track_label']
y_preds_deep_dmp = to_compare['y_track_pred_deep_dmp']
y_preds_segment = to_compare['y_track_pred_segment_limited']
total_loss_segment = 0
total_loss_deep_dmp = 0
for i in range(len(y_labels)):
    total_loss_segment += compare_closest(y_labels[i], y_preds_segment[i])
    total_loss_deep_dmp += compare_closest(y_labels[i], y_preds_deep_dmp[i])
total_loss_segment /= len(y_labels)
total_loss_deep_dmp /= len(y_labels)

print(np.round(total_loss_segment, 7))
print(np.round(total_loss_deep_dmp, 7))