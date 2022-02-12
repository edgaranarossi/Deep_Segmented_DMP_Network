#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 02:44:47 2022

@author: edgar
"""

from parameters import TrainingParameters
from utils.losses import DMPIntegrationMSE
from torch import ones, deg2rad
import torch
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_param = TrainingParameters()
loss_fn = DMPIntegrationMSE(train_param)

# loss = loss_fn(ones(3, 404).to(DEVICE), ones(3, 100, 2).to(DEVICE))

rot = 90
loss = loss_fn(X.float().to(DEVICE), 
               Y.float().to(DEVICE), 
               # rot_deg = deg2rad(torch.ones(X.shape[0], 1)*rot)).to(DEVICE)
                rot_deg = None)
traj = loss_fn.y_track
traj_np = traj.detach().cpu().numpy()
idx = 7
plt.plot(traj_np[idx, :, 0], traj_np[idx, :, 1])
plt.axis('equal')