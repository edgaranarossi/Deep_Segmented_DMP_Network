#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 02:44:47 2022

@author: edgar
"""

from parameters import TrainingParameters
from utils.losses import DMPIntegrationMSE
from torch import ones
import torch
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_param = TrainingParameters()
loss_fn = DMPIntegrationMSE(train_param)

loss = loss_fn(ones(3, 404).to(DEVICE), ones(3, 100, 2).to(DEVICE))
# loss, traj = loss_fn(X.float().to(DEVICE), Y.float().to(DEVICE))
# #%%
# traj_np = traj.detach().cpu().numpy()
# idx = 7
# plt.plot(traj_np[idx, :, 0], traj_np[idx, :, 1])