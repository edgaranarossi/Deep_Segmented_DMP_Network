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

def plot_trajectory(traj_np, idx):
    """
    Plot the trajectory.
    
    Parameters:
    traj_np (np.ndarray): The trajectory data.
    idx (int): The index of the trajectory to plot.
    """
    plt.plot(traj_np[idx, :, 0], traj_np[idx, :, 1])
    plt.scatter(traj_np[idx, :, 0], traj_np[idx, :, 1])
    plt.axis('equal')
    plt.show()

def plot_velocity(dy_np, idx):
    """
    Plot the velocity.
    
    Parameters:
    dy_np (np.ndarray): The velocity data.
    idx (int): The index of the velocity to plot.
    """
    plt.plot(range(dy_np.shape[1]), dy_np[idx, :, 0])
    plt.plot(range(dy_np.shape[1]), dy_np[idx, :, 1])
    plt.scatter(range(dy_np.shape[1]), dy_np[idx, :, 0])
    plt.scatter(range(dy_np.shape[1]), dy_np[idx, :, 1])
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    train_param = TrainingParameters()
    loss_fn = DMPIntegrationMSE(train_param)
    
    # Calculate loss
    rot = 0
    loss = loss_fn(X.float().to(DEVICE), 
                   Y.float().to(DEVICE), 
                   rot_deg=None)
    
    # Plot trajectory
    traj = loss_fn.y_track
    traj_np = traj.detach().cpu().numpy()
    idx = 7
    plot_trajectory(traj_np, idx)
    
    # Plot velocity
    dy = loss_fn.dy_track
    dy_np = dy.detach().cpu().numpy()
    plot_velocity(dy_np, idx)