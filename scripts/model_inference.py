#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:31:30 2022

@author: edgar
"""

import numpy as np
import torch
import pickle as pkl
from os.path import join
from utils.networks import *
from matplotlib import pyplot as plt
from bagpy import create_fig
from pydmps import DMPs_discrete
from utils.dataset_importer import str_to_ndarray

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model_parameters(model_path):
    """
    Load model parameters from the specified path.
    
    Parameters:
    model_path (str): Path to the model parameters.
    
    Returns:
    dict: Loaded model parameters.
    """
    return pkl.load(open(join(model_path, 'train-model-dmp_param.pkl'), 'rb'))

def load_data_loaders(model_path):
    """
    Load data loaders from the specified path.
    
    Parameters:
    model_path (str): Path to the data loaders.
    
    Returns:
    tuple: Train, validation, and test data loaders.
    """
    return pkl.load(open(join(model_path, 'data_loaders.pkl'), 'rb'))

if __name__=='__main__':
    ROOT_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
    MODEL_NAME = 'Model_SegmentedDMPNetwork_2022-07-09_20-15-36'
    MODEL_TYPE = MODEL_NAME.split('_')[1]
    MODEL_PATH = join(ROOT_DIR, 'models', MODEL_TYPE, MODEL_NAME)
    BEST_PARAM_PATH = join(MODEL_PATH, 'best_net_parameters')
    
    train_param = load_model_parameters(MODEL_PATH)
    model_param = train_param.model_param
    dmp_param_sd = model_param.dmp_param
    scaler = train_param.scaler
    output_mode = model_param.output_mode
    keys_to_normalize = model_param.keys_to_normalize
    train_loader, val_loader, test_loader = load_data_loaders(MODEL_PATH)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
#%%
    if MODEL_TYPE == 'SegmentedDMPNetwork':
        model = DSDNet(train_param)
    elif MODEL_TYPE == 'SegmentDMPCNN':
        model = SegmentDMPCNN(train_param)
    elif MODEL_TYPE == 'NormalDMPJoinedNetwork':
        model = NormalDMPJoinedNetwork(train_param)
    elif MODEL_TYPE == 'SegmentedDMPNetworkV2':
        POS_MODEL_NAME = 'Model_PositionNetwork_2022-06-07_16-07-58'
        POS_MODEL_TYPE = POS_MODEL_NAME.split('_')[1]
        POS_MODEL_PATH = join(ROOT_DIR, 'models', POS_MODEL_TYPE, POS_MODEL_NAME)
        POS_BEST_PARAM_PATH = join(POS_MODEL_PATH, 'best_net_parameters')
        
        pos_model_train_param = load_model_parameters(POS_MODEL_PATH)
        
        model = SegmentedDMPNetworkV2(train_param)
        pos_model = PositionNetwork(pos_model_train_param)
        pos_model.load_state_dict(torch.load(POS_BEST_PARAM_PATH))
        max_segments = model_param.max_segments
        
    model.load_state_dict(torch.load(BEST_PARAM_PATH))
    
    test_idx = 5
    input_image = test_dataset[test_idx][0]['image']
    outputs = test_dataset[test_idx][1]
    if 'original_trajectory' in outputs: 
        original_traj = str_to_ndarray(outputs['original_trajectory'])
    if 'segmented_dmp_trajectory' in outputs: 
        seg_dmp_traj = str_to_ndarray(outputs['segmented_dmp_trajectory'])
    if 'normal_dmp_trajectory' in outputs: 
        normal_dmp_traj = str_to_ndarray(outputs['normal_dmp_trajectory'])
    input_image = input_image.reshape(1, model_param.image_dim[0], model_param.image_dim[1], model_param.image_dim[2])
    pred_sd_dmp = model(input_image)
#%
    if input_image.shape[1] == 1:
        img = input_image[0, 0].detach().cpu().numpy()
    elif input_image.shape[1] == 3:    
        img = input_image[0].detach().cpu().numpy().reshape(100,100,3)
    plt.figure(figsize = (8, 8))
    plt.imshow(img)
    plt.title(MODEL_NAME)
    plt.show()
    
    denormalized_pred_sd_dmp = []
    for i, key in enumerate(output_mode):
        if key in keys_to_normalize:
            denormalized_pred_sd_dmp.append(scaler[key].denormalize(pred_sd_dmp[i]).detach().cpu().numpy())
        else:
            denormalized_pred_sd_dmp.append(pred_sd_dmp[i].detach().cpu().numpy())
            
#%%
    """MODEL_NAME = 'Model_NormalDMPJoinedNetwork_2022-06-13_20-59-49'
    MODEL_TYPE = MODEL_NAME.split('_')[1]
    MODEL_PATH = join(ROOT_DIR, 'models', MODEL_TYPE, MODEL_NAME)
    # BEST_PARAM_PATH = join(MODEL_PATH, 'final_net_parameters')
    BEST_PARAM_PATH = join(MODEL_PATH, 'best_net_parameters')
    
    train_param = pkl.load(open(join(MODEL_PATH, 'train-model-dmp_param.pkl'), 'rb'))
    model_param = train_param.model_param
    dmp_param_deep_dmp = model_param.dmp_param
    scaler = train_param.scaler
    output_mode = model_param.output_mode
    keys_to_normalize = model_param.keys_to_normalize
    train_loader, val_loader, test_loader = pkl.load(open(join(MODEL_PATH, 'data_loaders.pkl'), 'rb'))
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
    if MODEL_TYPE == 'SegmentedDMPNetwork':
        model = SegmentedDMPNetwork(train_param)
    elif MODEL_TYPE == 'SegmentDMPCNN':
        model = SegmentDMPCNN(train_param)
    elif MODEL_TYPE == 'NormalDMPJoinedNetwork':
        model = NormalDMPJoinedNetwork(train_param)
    elif MODEL_TYPE == 'SegmentedDMPNetworkV2':
        POS_MODEL_NAME = 'Model_PositionNetwork_2022-06-07_16-07-58'
        POS_MODEL_TYPE = POS_MODEL_NAME.split('_')[1]
        POS_MODEL_PATH = join(ROOT_DIR, 'models', POS_MODEL_TYPE, POS_MODEL_NAME)
        POS_BEST_PARAM_PATH = join(POS_MODEL_PATH, 'best_net_parameters')
        
        pos_model_train_param = pkl.load(open(join(POS_MODEL_PATH, 'train-model-dmp_param.pkl'), 'rb'))
        
        model = SegmentedDMPNetworkV2(train_param)
        pos_model = PositionNetwork(pos_model_train_param)
        pos_model.load_state_dict(torch.load(POS_BEST_PARAM_PATH))
        max_segments = model_param.max_segments
        
    model.load_state_dict(torch.load(BEST_PARAM_PATH))
    
    test_idx = 4
    input_image = test_dataset[test_idx][0]['image']
    outputs = test_dataset[test_idx][1]
    if 'original_trajectory' in outputs: original_traj = str_to_ndarray(outputs['rotated_trajectory'])
    input_image = input_image.reshape(1, model_param.image_dim[0], model_param.image_dim[1], model_param.image_dim[2])
    pred_deep_dmp = model(input_image)
#%
    img = input_image[0, 0].detach().cpu().numpy()
    # plt.figure(figsize = (8, 8))
    # plt.imshow(img)
    # plt.title(MODEL_NAME)
    # plt.show()
    
    denormalized_pred_deep_dmp = []
    for i, key in enumerate(output_mode):
        if key in keys_to_normalize:
            denormalized_pred_deep_dmp.append(scaler[key].denormalize(pred_deep_dmp[i]).detach().cpu().numpy())
        else:
            denormalized_pred_deep_dmp.append(pred_deep_dmp[i].detach().cpu().numpy())"""
#%%
    fig, ax = create_fig(dmp_param_sd.dof)
    
    # ax[0].legend(['SD-DMPs', 'Deep-DMPs'])
    # ax[1].legend(['SD-DMPs', 'Deep-DMPs'])
    # ax[2].legend(['SD-DMPs', 'Deep-DMPs'])
    # ax[3].legend(['SD-DMPs', 'Deep-DMPs'])
    
    # if 'original_trajectory' in outputs: 
    #     ax[0].scatter(range(original_traj.shape[0]), original_traj[:, 0], c = 'r')
    #     ax[1].scatter(range(original_traj.shape[0]), original_traj[:, 1], c = 'r')
    #     if dmp_param_sd.dof == 3:
    #         ax[2].scatter(range(original_traj.shape[0]), original_traj[:, 2], c = 'r')
    #     if dmp_param_sd.dof == 4:
    #         ax[2].scatter(range(original_traj.shape[0]), original_traj[:, 2], c = 'r')
    #         ax[3].scatter(range(original_traj.shape[0]), original_traj[:, 3], c = 'r')
        
    if 'segmented_dmp_trajectory' in outputs: 
        ax[0].scatter(range(seg_dmp_traj.shape[0]), seg_dmp_traj[:, 0], c = 'c')
        ax[1].scatter(range(seg_dmp_traj.shape[0]), seg_dmp_traj[:, 1], c = 'c')
        if dmp_param_sd.dof == 3:
            ax[2].scatter(range(seg_dmp_traj.shape[0]), seg_dmp_traj[:, 2], c = 'c')
        if dmp_param_sd.dof == 4:
            ax[2].scatter(range(seg_dmp_traj.shape[0]), seg_dmp_traj[:, 2], c = 'c')
            ax[3].scatter(range(seg_dmp_traj.shape[0]), seg_dmp_traj[:, 3], c = 'c')
            
    if 'normal_dmp_trajectory' in outputs: 
        ax[0].scatter(range(normal_dmp_traj.shape[0]), normal_dmp_traj[:, 0], c = 'b')
        ax[1].scatter(range(normal_dmp_traj.shape[0]), normal_dmp_traj[:, 1], c = 'b')
        if dmp_param_sd.dof == 3:
            ax[2].scatter(range(normal_dmp_traj.shape[0]), normal_dmp_traj[:, 2], c = 'b')
        if dmp_param_sd.dof == 4:
            ax[2].scatter(range(normal_dmp_traj.shape[0]), normal_dmp_traj[:, 2], c = 'b')
            ax[3].scatter(range(normal_dmp_traj.shape[0]), normal_dmp_traj[:, 3], c = 'b')
        
    # ax[0].set_ylim([-0.13, 0.2])
    # ax[1].set_ylim([0.0, 0.4])
    # ax[2].set_ylim([-0.35, 0.15])
    
    # if MODEL_TYPE in ['SegmentedDMPNetwork', 'SegmentDMPCNN']:
    num_segments = int(np.round(denormalized_pred_sd_dmp[0].reshape(1)[0]))
    dmp_y0 = denormalized_pred_sd_dmp[1][0]
    # dmp_y0 = outputs['segmented_dmp_y0'].detach().cpu().numpy()
    dmp_goal = denormalized_pred_sd_dmp[2][0]
    # dmp_y0[1:] = dmp_goal[:-1]
    # dmp_goal[:-1] = dmp_y0[1:]
    # dmp_goal = outputs['segmented_dmp_goal'].detach().cpu().numpy()
    dmp_w = denormalized_pred_sd_dmp[3][0]
    # dmp_w = outputs['segmented_dmp_w'].detach().cpu().numpy()
    # dmp_tau = denormalized_pred_sd_dmp[4][0]
    dmp_tau = outputs['segmented_dmp_tau'].detach().cpu().numpy()
    
    
    seg_trajs = []
    start_idx = 0
    for i in range(num_segments):
        dmp = DMPs_discrete(n_dmps = dmp_param_sd.dof,
                            n_bfs = dmp_param_sd.n_bf,
                            ay = np.ones(dmp_param_sd.dof) * dmp_param_sd.ay,
                            dt = dmp_param_sd.dt)
        dmp.w = dmp_w[i]
        dmp.y0 = dmp_y0[i]
        dmp.goal = dmp_goal[i]
        y_track, dy_track, ddy_track = dmp.rollout(tau = dmp_tau[i])
        seg_trajs.append(y_track)
        
        # ax[0].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 0], c = 'g', zorder = 99)
        # ax[1].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 1], c = 'g', zorder = 99)
        # if dmp_param_sd.dof == 3:
        #     ax[2].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 2], c = 'g', zorder = 99)
        # if dmp_param_sd.dof == 4:
        #     ax[2].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 2], c = 'g', zorder = 99)
        #     ax[3].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 3], c = 'g', zorder = 99)
        start_idx = start_idx + y_track.shape[0]
        
    
    
    # if 'original_trajectory' in outputs: 
    #     ax[0].scatter(range(original_traj.shape[0]), original_traj[:, 0], c = 'r')
    #     ax[1].scatter(range(original_traj.shape[0]), original_traj[:, 1], c = 'r')
    #     ax[2].scatter(range(original_traj.shape[0]), original_traj[:, 2], c = 'r')    
    # plt.show()
    
    # if 'original_trajectory' in outputs: 
    #     plt.figure(figsize = (20, 20))
    #     plt.axis('equal')
    #     plt.scatter(-original_traj[:, 2], original_traj[:, 1], c = 'r')
        
    # plt.figure(figsize = (20, 20))
    # plt.axis('equal')
    # for seg_traj in seg_trajs[:-1]:
    #     plt.scatter(-seg_traj[:, 2], seg_traj[:, 1], c = 'g')
    # plt.show()
        
    # elif MODEL_TYPE == 'SegmentedDMPNetworkV2':
    #     num_segments = torch.tensor([np.round(denormalized_pred_sd_dmp[0].reshape(1)[0])]).to(DEVICE).reshape(1, 1)
    #     dmp_y0 = torch.tensor(denormalized_pred_sd_dmp[1]).to(DEVICE)
    #     goals = [pos_model(num_segments, dmp_y0)[0]]
    #     for i in range(max_segments - 1):
    #         goals.append(pos_model(num_segments, goals[-1])[0])
    #     all_points = [dmp_y0] + goals
        
    #     dmp_y0 = np.array([i.detach().cpu().numpy() for i in all_points[:-1]]).reshape(max_segments, dmp_param.dof)
    #     dmp_goal = np.array([i.detach().cpu().numpy() for i in all_points[1:]]).reshape(max_segments, dmp_param.dof)
    #     dmp_w = denormalized_pred_sd_dmp[2][0]
    #     dmp_tau = denormalized_pred_sd_dmp[3][0]
        
    #     # fig, ax = create_fig(dmp_param.dof)
        
    #     start_idx = 0
    #     for i in range(int(num_segments.item())):
    #         dmp = DMPs_discrete(n_dmps = dmp_param.dof,
    #                             n_bfs = dmp_param.n_bf,
    #                             ay = np.ones(dmp_param.dof) * dmp_param.ay,
    #                             dt = dmp_param.dt)
    #         dmp.w = dmp_w[i]
    #         dmp.y0 = dmp_y0[i]
    #         dmp.goal = dmp_goal[i]
    #         y_track, dy_track, ddy_track = dmp.rollout()
            
    #         ax[0].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 0], c = 'g', zorder = 99)
    #         ax[1].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 1], c = 'g', zorder = 99)
    #         if dmp_param.dof == 3:
    #             ax[2].scatter(range(start_idx, start_idx + y_track.shape[0]), y_track[:, 2], c = 'g', zorder = 99)
    #         start_idx = start_idx + y_track.shape[0]
    #%%
    # elif MODEL_TYPE == 'NormalDMPJoinedNetwork':
    """dmp_y0 = denormalized_pred_deep_dmp[0][0]
    dmp_goal = denormalized_pred_deep_dmp[1][0]
    dmp_w = denormalized_pred_deep_dmp[2][0]
    dmp_tau = denormalized_pred_deep_dmp[3][0]
    
    # fig, ax = create_fig(dmp_param.dof)
    dmp = DMPs_discrete(n_dmps = dmp_param_deep_dmp.dof,
                        n_bfs = dmp_param_deep_dmp.n_bf,
                        ay = np.ones(dmp_param_deep_dmp.dof) * 25,
                        dt = 1e-4)
    dmp.w = dmp_w
    dmp.y0 = dmp_y0
    dmp.goal = dmp_goal
    
    y_track, dy_track, ddy_track = dmp.rollout(tau = dmp_tau)
    
    ax[0].scatter(range(y_track.shape[0]), y_track[:, 0], c = 'b')
    ax[1].scatter(range(y_track.shape[0]), y_track[:, 1], c = 'b')
    if dmp_param_deep_dmp.dof == 3:
        ax[2].scatter(range(y_track.shape[0]), y_track[:, 2], c = 'b')
        
    plt.show()"""
        
        # if 'original_trajectory' in outputs: 
        #     plt.figure(figsize = (20, 20))
        #     plt.axis('equal')
        #     plt.scatter(-original_traj[:, 2], original_traj[:, 1], c = 'r')
            
        # plt.figure(figsize = (20, 20))
        # plt.axis('equal')
        # plt.scatter(-y_track[:, 2], y_track[:, 1], c = 'b')