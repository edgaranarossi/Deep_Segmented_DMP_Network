#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Testing Script

This script contains functions to test various tasks for the Deep Segmented DMP Network.
"""

import numpy as np
from numpy import sin, cos, tan, array
import torch
import pickle as pkl
from os.path import join, isdir
from os import listdir, makedirs
from utils.networks import DSDNetV1, DSDNetV2, CIMEDNet
from utils.dataset_importer import str_to_ndarray, ndarray_to_str
from utils.data_generator.pick_and_place_generator import PickAndPlaceGenerator
from generate_dataset_cutting import generate_cutting_traj
from matplotlib import pyplot as plt
from pydmps import DMPs_discrete
from copy import copy, deepcopy
from scipy.signal import decimate
from utils.data_generator import ObjectGenerator
from PIL import Image
from numpy.random import rand, randint
from dtaidistance import dtw

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rot3d(origin, traj, degrees, order=None):
    """
    Rotate a 3D trajectory around a given origin.

    Parameters:
    origin (array): The origin point for rotation.
    traj (array): The trajectory to be rotated.
    degrees (tuple): The rotation angles in degrees for x, y, and z axes.
    order (list): The order of rotation axes.

    Returns:
    array: The rotated trajectory.
    """
    deg_x, deg_y, deg_z = degrees
    deg_x = np.deg2rad(deg_x)
    deg_y = np.deg2rad(deg_y)
    deg_z = np.deg2rad(deg_z)
    if order == None: order = ['x', 'y', 'z']
    
    rot_x = np.array([[1., 0., 0.], 
                      [0., cos(deg_x), -sin(deg_x)],
                      [0., sin(deg_x), cos(deg_x)]])
    rot_y = np.array([[cos(deg_y), 0., sin(deg_y)],
                      [0., 1., 0.],
                      [-sin(deg_y), 0., cos(deg_y)]])
    rot_z = np.array([[cos(deg_z), -sin(deg_z), 0.],
                      [sin(deg_z), cos(deg_z), 0.],
                      [0., 0.,  1.]])
    
    if order[0] == 'x':
        rot_mat = rot_x
    elif order[0] == 'y':
        rot_mat = rot_y
    elif order[0] == 'z':
        rot_mat = rot_z
    
    for i in order[1:]:
        if i == 'x':
            rot_mat = rot_mat @ rot_x
        elif i == 'y':
            rot_mat = rot_mat @ rot_y
        elif i == 'z':
            rot_mat = rot_mat @ rot_z
    
    t = deepcopy(traj)
    t -= origin
    t = (rot_mat @ t.T).T
    # t += origin
    
    return t

def rot2D(traj, degrees, origin=np.array([0., 0.])):
    """
    Rotate a 2D trajectory around a given origin.

    Parameters:
    traj (array): The trajectory to be rotated.
    degrees (float): The rotation angle in degrees.
    origin (array): The origin point for rotation.

    Returns:
    array: The rotated trajectory.
    """
    theta = np.deg2rad(degrees)
    rot_mat = np.array([[cos(theta), -sin(theta)], 
                        [sin(theta), cos(theta)]])
    t = deepcopy(traj)
    t -= origin
    t = (rot_mat @ t.T).T
    t += origin
    return t

def sampling_rmse(y1, y2):
    """
    Calculate the RMSE between two trajectories with different sampling rates.

    Parameters:
    y1 (array): The first trajectory.
    y2 (array): The second trajectory.

    Returns:
    tuple: The RMSE, sampled longer trajectory, and sampled shorter trajectory.
    """
    if y1.shape[0] > y2.shape[0]:
        higher_sampler = y1.shape[0] // y2.shape[0]
        lower_sampler = higher_sampler + 1
        
        used_idx = [i for i in range(0, y1.shape[0], lower_sampler)]
        higher_sampled_idx = np.array([i for i in range(0, y1.shape[0], higher_sampler)])
        
        while len(used_idx) < y2.shape[0]:
            missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
            to_add = y2.shape[0] - len(used_idx)
            
            higher_sampler = len(missing_idx) // to_add
            lower_sampler = higher_sampler + 1
            
            to_add = np.array(missing_idx)[::lower_sampler].tolist()
            for i in to_add:
                used_idx.append(i)
        used_idx = np.array(sorted(used_idx))
        sampled_y1 = y1[used_idx]
        sampled_y2 = deepcopy(y2)
    else:
        higher_sampler = y2.shape[0] // y1.shape[0]
        lower_sampler = higher_sampler + 1
        
        used_idx = [i for i in range(0, y2.shape[0], lower_sampler)]
        higher_sampled_idx = np.array([i for i in range(0, y2.shape[0], higher_sampler)])
        
        while len(used_idx) < y1.shape[0]:
            missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
            to_add = y1.shape[0] - len(used_idx)
            
            higher_sampler = len(missing_idx) // to_add
            lower_sampler = higher_sampler + 1
            
            to_add = np.array(missing_idx)[::lower_sampler].tolist()
            for i in to_add:
                used_idx.append(i)
        used_idx = np.array(sorted(used_idx))
        sampled_y2 = y2[used_idx]
        sampled_y1 = deepcopy(y1)
    
    rmse = np.sqrt((sampled_y1 - sampled_y2)**2).mean()
    return rmse, sampled_y1, sampled_y2

def generate_cutting_image_motion():
    """
    Generate a cutting image motion.

    Returns:
    tuple: The generated image, input image tensor, and segmented DMP trajectory.
    """
    # ...existing code...

def generate_pickplace_image_motion(task, num_object=None):
    """
    Generate a pick and place image motion.

    Parameters:
    task (str): The task type.
    num_object (int): The number of objects.

    Returns:
    tuple: The generated image, input image tensor, y_label, and pickplace generator.
    """
    # ...existing code...

def read_test_data(dataset, test_idx):
    """
    Read test data from the dataset.

    Parameters:
    dataset (Dataset): The dataset to read from.
    test_idx (int): The index of the test data.

    Returns:
    tuple: The image, input image tensor, y_label, rotation degrees, and rotation order.
    """
    # ...existing code...

#%%

if __name__ == '__main__':
    """
    Main function to run the task testing script.
    """
    runcell(0, '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/scripts/task_testing.py')
    # task = 'cutting'
    # task = 'cutting-limited'
    task = 'cutting-real'
    # task = 'pickplace'
    # task = 'pickplace-rand'

    if task == 'cutting':
        dsdnet_name = 'Model_DSDNetV1_2022-09-12_07-23-54'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_07-25-28'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_07-25-46'
    elif task == 'cutting-limited':
        dsdnet_name = 'Model_DSDNetV1_2022-09-11_22-51-01'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-11_22-52-23'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-11_22-54-15'
        task = 'cutting'
    elif task == 'cutting-real':
        dsdnet_name = 'Model_DSDNetV1_2022-09-12_02-23-40'
        # cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_02-24-02'
        # cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_02-24-38'
        task = 'cutting'
    elif task == 'pickplace':
        dsdnet_name = 'Model_DSDNetV1_2022-09-11_00-27-13'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_17-20-24'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_17-21-32'
    elif task == 'pickplace-rand':
        dsdnet_name = 'Model_DSDNetV1_2022-09-11_15-58-06'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_00-52-09'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_17-15-57'
        task = 'pickplace'


    root_model_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'

    dsdnet_dir = join(root_model_dir, dsdnet_name.split('_')[1], dsdnet_name)
    if cimednet_name != None: cimednet_dir = join(root_model_dir, cimednet_name.split('_')[1], cimednet_name)
    if cimednet_L_name != None: cimednet_L_dir = join(root_model_dir, cimednet_L_name.split('_')[1], cimednet_L_name)

    dsdnet_train_param = pkl.load(open(join(dsdnet_dir, 'train-model-dmp_param.pkl'), 'rb'))
    if cimednet_name != None: cimednet_train_param = pkl.load(open(join(cimednet_dir, 'train-model-dmp_param.pkl'), 'rb'))
    if cimednet_L_name != None: cimednet_L_train_param = pkl.load(open(join(cimednet_L_dir, 'train-model-dmp_param.pkl'), 'rb'))

    dsdnet_model_param = dsdnet_train_param.model_param
    if cimednet_name != None: cimednet_model_param = cimednet_train_param.model_param
    if cimednet_L_name != None: cimednet_L_model_param = cimednet_L_train_param.model_param

    dsdnet_best_param = join(dsdnet_dir, 'best_net_parameters')
    if cimednet_name != None: cimednet_best_param = join(cimednet_dir, 'best_net_parameters')
    if cimednet_L_name != None: cimednet_L_best_param = join(cimednet_L_dir, 'best_net_parameters')

    if dsdnet_name.split('_')[1] == 'DSDNetV1':
        dsdnet_model = DSDNetV1(dsdnet_train_param)
    elif dsdnet_name.split('_')[1] == 'DSDNetV2':
        dsdnet_model = DSDNetV2(dsdnet_train_param)
    if cimednet_name != None: cimednet_model = CIMEDNet(cimednet_train_param)
    if cimednet_L_name != None: cimednet_L_model = CIMEDNet(cimednet_L_train_param)

    dsdnet_model.load_state_dict(torch.load(dsdnet_best_param))
    print('DSDNet model loaded')
    if cimednet_name != None: 
        cimednet_model.load_state_dict(torch.load(cimednet_best_param))
        print('CIMEDNet model loaded')
    if cimednet_L_name != None: 
        cimednet_L_model.load_state_dict(torch.load(cimednet_L_best_param))
        print('CIMEDNet_L model loaded')

    dsdnet_model.eval()
    if cimednet_name != None: cimednet_model.eval()
    if cimednet_L_name != None: cimednet_L_model.eval()

    scaler = dsdnet_train_param.scaler
    dsdnet_output_mode = dsdnet_model_param.output_mode
    if cimednet_name != None: cimednet_output_mode = cimednet_model_param.output_mode
    if cimednet_L_name != None: cimednet_L_output_mode = cimednet_model_param.output_mode
    keys_to_normalize = dsdnet_model_param.keys_to_normalize
    #%%
    if task == 'cutting':
        train_loader, val_loader, test_loader = pkl.load(open(join(dsdnet_dir, 'data_loaders.pkl'), 'rb'))
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
        
        used_dataset = test_dataset
        # used_dataset = val_dataset
        to_test = len(used_dataset)
        print('\nDataset length = {}'.format(to_test))

    print('')
    #%%
    test_idx = 3

    fig_size = 9
    ls = 10 
    DSDNetV1_MSEs = []
    CIMEDNet_MSEs = []
    CIMEDNet_L_MSEs = []

    PLOT_STEP = 50
    GRIP_THRESHOLD = 5
    SHOW_SIDE = True
    SHOW_POS = True

    if 1:
    # loop = input('Enter to generate next data')
    # while loop == '':
    # while test_idx < to_test:
        if task == 'cutting':
            original_padding = np.array([50, 20, 23])
            # original_padding = np.array([0, 0, 0])
            padding = deepcopy(original_padding)
            multiplier = np.array([300, 650, -250])
            
            to_plot = 'image'
            # to_plot = 'pos'
            
            img = used_dataset[test_idx][0]['image'].reshape(dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2]).detach().cpu().numpy()
            input_image = used_dataset[test_idx][0]['image'].reshape(1, dsdnet_model_param.image_dim[0], dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2])
            outputs = used_dataset[test_idx][1]

            num_cuts = outputs['segmented_dmp_seg_num']
            num_cuts = (int(scaler['segmented_dmp_seg_num'].denormalize(num_cuts).item()) // 2) + 1
            
            # if 'original_trajectory' in outputs: 
            #     original_traj = str_to_ndarray(outputs['original_trajectory'])
            # if 'rotated_trajectory' in outputs: 
            #     original_traj = str_to_ndarray(outputs['rotated_trajectory'])
            # if 'processed_trajectory' in outputs: 
            #     original_traj = str_to_ndarray(outputs['processed_trajectory'])
            if 'segmented_dmp_trajectory' in outputs: 
                original_traj = str_to_ndarray(outputs['segmented_dmp_trajectory'])
            if 'normal_dmp_trajectory' in outputs: 
                normal_dmp_traj = str_to_ndarray(outputs['normal_dmp_trajectory'])
        # elif task == 'cutting-real':
        #     original_padding = np.array([50, 22, 23])
        #     # original_padding = np.array([0, 0, 0])
        #     padding = deepcopy(original_padding)
        #     multiplier = np.array([300, 1000, 250])
            
        #     # to_plot = 'image'
        #     to_plot = 'pos'
            
        #     img = used_dataset[test_idx][0]['image'].reshape(dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2]).detach().cpu().numpy()
        #     input_image = used_dataset[test_idx][0]['image'].reshape(1, dsdnet_model_param.image_dim[0], dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2])
        #     outputs = used_dataset[test_idx][1]

        #     num_cuts = outputs['segmented_dmp_seg_num']
        #     num_cuts = (int(scaler['segmented_dmp_seg_num'].denormalize(num_cuts).item()) // 2) + 1
            
        #     # if 'original_trajectory' in outputs: 
        #     #     original_traj = str_to_ndarray(outputs['original_trajectory'])
        #     # if 'rotated_trajectory' in outputs: 
        #     #     original_traj = str_to_ndarray(outputs['rotated_trajectory'])
        #     # if 'processed_trajectory' in outputs: 
        #     #     original_traj = str_to_ndarray(outputs['processed_trajectory'])
        #     if 'segmented_dmp_trajectory' in outputs: 
        #         original_traj = str_to_ndarray(outputs['segmented_dmp_trajectory'])
        #     if 'normal_dmp_trajectory' in outputs: 
        #         normal_dmp_traj = str_to_ndarray(outputs['normal_dmp_trajectory'])
            
        elif task == 'pickplace':
            # img = used_dataset[test_idx][0]['image'].reshape(dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2], dsdnet_model_param.image_dim[0]).detach().cpu().numpy()
            root_data_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data'
            task_test_img_dir = join(root_data_dir, 'images/stacking/task_test')
            task_test_pkl_dir = join(root_data_dir, 'images/stacking/task_test')
            pickplacegen = PickAndPlaceGenerator(task_test_img_dir,
                                                 task_test_pkl_dir, 
                                                 # [1],
                                                 [1, 2, 3],
                                                 img_size = (150, 150),
                                                 # permute_block_pos = True,
                                                  permute_block_pos = False,
                                                   # randomize_block_pos = True,
                                                    randomize_block_pos = False,
                                                   # randomize_target_pos = True,
                                                    randomize_target_pos = False,
                                                 pos_noise_magnitude = 0,
                                                 motion_noise_magnitude = 0,
                                                 )
            img = pickplacegen.generateTestData(num_object = 3)
            # pickplacegen.map.plot(show_gripper = False, show_trail = False)
            img = img.reshape(1, img.shape[-1], img.shape[0], img.shape[1]) / 255
            input_image = torch.tensor(img).to(DEVICE).float()
    # #%%        
        # sample_output = test_dataset[0][1]
        # dsdnet_ay = sample_output['segmented_dmp_ay']
        # dsdnet_dt = sample_output['segmented_dmp_dt']
        # if cimednet_name != None: 
        #     cimednet_ay = sample_output['normal_dmp_ay']
        #     cimednet_dt = sample_output['normal_dmp_dt']
        # if cimednet_L_name != None: 
        #     cimednet_L_ay = sample_output['normal_dmp_L_ay']
        #     cimednet_L_dt = sample_output['normal_dmp_dt']
        
        dsdnet_ay = dsdnet_model_param.dmp_param.ay
        dsdnet_dt = dsdnet_model_param.dmp_param.dt
        if cimednet_name != None: 
            cimednet_ay = cimednet_model_param.dmp_param.ay
            cimednet_dt = cimednet_model_param.dmp_param.dt
        if cimednet_L_name != None: 
            cimednet_L_ay = cimednet_L_model_param.dmp_param.ay
            cimednet_L_dt = cimednet_L_model_param.dmp_param.dt
            
        dsdnet_pred = dsdnet_model(input_image)
        if cimednet_name != None: cimednet_pred = cimednet_model(input_image)
        if cimednet_L_name != None: cimednet_L_pred = cimednet_L_model(input_image)
        
        dsdnet_pred_denormalized = []
        if cimednet_name != None: cimednet_pred_denormalized = []
        if cimednet_L_name != None: cimednet_L_pred_denormalized = []
        
        for i, key in enumerate(dsdnet_output_mode):
            if key in keys_to_normalize:
                dsdnet_pred_denormalized.append(scaler[key].denormalize(dsdnet_pred[i]).detach().cpu().numpy())
            else:
                dsdnet_pred_denormalized.append(dsdnet_pred[i].detach().cpu().numpy())
        
        if cimednet_name != None: 
            for i, key in enumerate(cimednet_output_mode):
                if key in keys_to_normalize:
                    cimednet_pred_denormalized.append(scaler[key].denormalize(cimednet_pred[i]).detach().cpu().numpy())
                else:
                    cimednet_pred_denormalized.append(cimednet_pred[i].detach().cpu().numpy())
        
        if cimednet_L_name != None: 
            for i, key in enumerate(cimednet_L_output_mode):
                if key in keys_to_normalize:
                    cimednet_L_pred_denormalized.append(scaler[key].denormalize(cimednet_L_pred[i]).detach().cpu().numpy())
                else:
                    cimednet_L_pred_denormalized.append(cimednet_L_pred[i].detach().cpu().numpy())
        
        if dsdnet_name.split('_')[1] == 'DSDNetV1':
            idx_modifier = 0
        elif dsdnet_name.split('_')[1] == 'DSDNetV2':
            idx_modifier = 1
            
        dsdnet_num_seg = int(np.round(dsdnet_pred_denormalized[0 + idx_modifier][0]))
        dsdnet_y0 = dsdnet_pred_denormalized[1 + idx_modifier][0]
        dsdnet_goal = dsdnet_pred_denormalized[2 + idx_modifier][0]
        dsdnet_w = dsdnet_pred_denormalized[3 + idx_modifier][0]
        dsdnet_tau = dsdnet_pred_denormalized[4 + idx_modifier][0]
        
        if cimednet_name != None: 
            cimednet_y0 = cimednet_pred_denormalized[0][0]
            cimednet_goal = cimednet_pred_denormalized[1][0]
            cimednet_w = cimednet_pred_denormalized[2][0]
            cimednet_tau = cimednet_pred_denormalized[3][0]
        
        if cimednet_L_name != None:
            cimednet_L_y0 = cimednet_L_pred_denormalized[0][0]
            cimednet_L_goal = cimednet_L_pred_denormalized[1][0]
            cimednet_L_w = cimednet_L_pred_denormalized[2][0]
            cimednet_L_tau = cimednet_L_pred_denormalized[3][0]
        
        dsdnet_y = []
        for i in range(min(dsdnet_num_seg, dsdnet_y0.shape[0])):
            dmp = DMPs_discrete(n_dmps = dsdnet_w.shape[-2], n_bfs = dsdnet_w.shape[-1], ay = np.ones(dsdnet_w.shape[-2]) * dsdnet_ay, dt = dsdnet_dt)
            dmp.y0 = dsdnet_y0[i]
            dmp.goal = dsdnet_goal[i]
            dmp.w = dsdnet_w[i]
            y, _, _ = dmp.rollout()
            dsdnet_y.append(y)
        dsdnet_y = np.array(dsdnet_y).reshape(-1, dsdnet_w.shape[-2])
            
        if cimednet_name != None: 
            dmp = DMPs_discrete(n_dmps = cimednet_w.shape[-2], n_bfs = cimednet_w.shape[-1], ay = np.ones(cimednet_w.shape[-2]) * cimednet_ay, dt = cimednet_dt)
            dmp.y0 = cimednet_y0
            dmp.goal = cimednet_goal
            dmp.w = cimednet_w
            cimednet_y, _, _ = dmp.rollout()
            
        if cimednet_L_name != None: 
            dmp = DMPs_discrete(n_dmps = cimednet_L_w.shape[-2], n_bfs = cimednet_L_w.shape[-1], ay = np.ones(cimednet_L_w.shape[-2]) * cimednet_L_ay, dt = cimednet_L_dt)
            dmp.y0 = cimednet_L_y0
            dmp.goal = cimednet_L_goal
            dmp.w = cimednet_L_w
            cimednet_L_y, _, _ = dmp.rollout()
        
        # SAMPLE
        if task == 'cutting' and original_traj.shape[0] > dsdnet_y.shape[0]:
            higher_sampler = original_traj.shape[0] // dsdnet_y.shape[0]
            lower_sampler = higher_sampler + 1
            
            used_idx = [i for i in range(0, original_traj.shape[0], lower_sampler)]
            higher_sampled_idx = np.array([i for i in range(0, original_traj.shape[0], higher_sampler)])
            
            while len(used_idx) < dsdnet_y.shape[0]:
                missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
                to_add = dsdnet_y.shape[0] - len(used_idx)
                
                
                higher_sampler = len(missing_idx) // to_add
                lower_sampler = higher_sampler + 1
                
                to_add = np.array(missing_idx)[::lower_sampler].tolist()
                for i in to_add:
                    used_idx.append(i)
            used_idx = np.array(sorted(used_idx))
            sampled_original_traj = original_traj[used_idx]
            original_traj = deepcopy(sampled_original_traj)
        elif task == 'cutting' and original_traj.shape[0] < dsdnet_y.shape[0]:
            higher_sampler = dsdnet_y.shape[0] // original_traj.shape[0]
            lower_sampler = higher_sampler + 1
            
            used_idx = [i for i in range(0, dsdnet_y.shape[0], lower_sampler)]
            higher_sampled_idx = np.array([i for i in range(0, dsdnet_y.shape[0], higher_sampler)])
            
            while len(used_idx) < original_traj.shape[0]:
                missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
                to_add = original_traj.shape[0] - len(used_idx)
                
                
                higher_sampler = len(missing_idx) // to_add
                lower_sampler = higher_sampler + 1
                
                to_add = np.array(missing_idx)[::lower_sampler].tolist()
                for i in to_add:
                    used_idx.append(i)
            used_idx = np.array(sorted(used_idx))
            sampled_dsdnet_y = dsdnet_y[used_idx]
            dsdnet_y = deepcopy(sampled_dsdnet_y)
            
        if cimednet_name != None and cimednet_y.shape[0] > dsdnet_y.shape[0]:
            higher_sampler = cimednet_y.shape[0] // dsdnet_y.shape[0]
            lower_sampler = higher_sampler + 1
            
            used_idx = [i for i in range(0, cimednet_y.shape[0], lower_sampler)]
            higher_sampled_idx = np.array([i for i in range(0, cimednet_y.shape[0], higher_sampler)])
            
            while len(used_idx) < dsdnet_y.shape[0]:
                missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
                to_add = dsdnet_y.shape[0] - len(used_idx)
                
                higher_sampler = len(missing_idx) // to_add
                lower_sampler = higher_sampler + 1
                
                to_add = np.array(missing_idx)[::lower_sampler].tolist()
                for i in to_add:
                    used_idx.append(i)
            used_idx = np.array(sorted(used_idx))
            sampled_cimednet_y = cimednet_y[used_idx]
            cimednet_y = deepcopy(sampled_cimednet_y)
            
        if cimednet_L_name != None and cimednet_L_y.shape[0] > dsdnet_y.shape[0]:
            higher_sampler = cimednet_L_y.shape[0] // dsdnet_y.shape[0]
            lower_sampler = higher_sampler + 1
            
            used_idx = [i for i in range(0, cimednet_L_y.shape[0], lower_sampler)]
            higher_sampled_idx = np.array([i for i in range(0, cimednet_L_y.shape[0], higher_sampler)])
            
            while len(used_idx) < dsdnet_y.shape[0]:
                missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
                to_add = dsdnet_y.shape[0] - len(used_idx)
                
                
                higher_sampler = len(missing_idx) // to_add
                lower_sampler = higher_sampler + 1
                
                to_add = np.array(missing_idx)[::lower_sampler].tolist()
                for i in to_add:
                    used_idx.append(i)
            used_idx = np.array(sorted(used_idx))
            sampled_cimednet_L_y = cimednet_L_y[used_idx]
            cimednet_L_y = deepcopy(sampled_cimednet_L_y)
            
        # cimednet_y -= cimednet_y.min(axis = 0)
        
        # if dsdnet_y.shape[0] > original_traj.shape[0]: dsdnet_y = dsdnet_y[:original_traj.shape[0]]
        # if cimednet_y.shape[0] > original_traj.shape[0]: cimednet_y = cimednet_y[:original_traj.shape[0]]
        
        # DSDNetV1_MSEs.append(np.sqrt((original_traj - dsdnet_y)**2).mean())
        # CIMEDNet_MSEs.append(np.sqrt((original_traj - cimednet_y)**2).mean())
        # CIMEDNet_L_MSEs.append(np.sqrt((original_traj - cimednet_L_y)**2).mean())
        # test_idx += 1
        # print('DSDNetV1 MSE = {}\nCIMEDNet MSE = {}'.format(DSDNetV1_MSEs[-1], CIMEDNet_MSEs[-1]))
        
    # print('DSDNetV1 MSE = {}\nCIMEDNet MSE = {}\nCIMEDNet-accurate MSE = {}'.format(np.array(DSDNetV1_MSEs).mean(), np.array(CIMEDNet_MSEs).mean(), np.array(CIMEDNet_L_MSEs).mean()))
    #%
        if task == 'cutting':
            
            if dsdnet_y.shape[1] == 2:
                
                dsdnet_y_plot = deepcopy(dsdnet_y)
                if cimednet_name != None: cimednet_y_plot = deepcopy(cimednet_y)
                if cimednet_L_name != None: cimednet_L_y_plot = deepcopy(cimednet_L_y)
                original_traj_plot = deepcopy(original_traj)
                
                SUCCESS_THRESHOLD = 5
                padding = 5
                multiplier = 55
                dsdnet_y_plot *= multiplier
                if cimednet_name != None: cimednet_y_plot *= multiplier
                if cimednet_L_name != None: cimednet_L_y_plot *= multiplier
                original_traj_plot *= multiplier
                dsdnet_y_plot += padding
                if cimednet_name != None: cimednet_y_plot += padding
                if cimednet_L_name != None: cimednet_L_y_plot += padding
                original_traj_plot += padding
                
                # plt.figure(figsize = (fig_size, fig_size))
                # plt.figure(figsize = (fig_size * 3, fig_size))
                # plt.figure(figsize = (fig_size, fig_size * 3))
                plt.figure(figsize = (fig_size * 3, fig_size))
                
                rows = 2 if cimednet_name is None else 3 if cimednet_L_name is None else 4
                
                # plt.subplot(rows, 1, 1)
                plt.subplot(1, rows, 1)
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                plt.scatter(original_traj_plot[:, 0], original_traj_plot[:, 1], c = 'y', s = ls)
                
                # plt.subplot(rows, 1, 2)
                plt.subplot(1, rows, 2)
                # plt.title('DSDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                # plt.title('DSDNet')
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                # plt.figure(figsize = (fig_size, fig_size))
                plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                plt.scatter(dsdnet_y_plot[:, 0], dsdnet_y_plot[:, 1], c = 'g', s = ls)
                plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                
                if cimednet_name != None: 
                    # plt.subplot(rows, 1, 3)
                    plt.subplot(1, rows, 3)
                    # plt.title('CIMEDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                    # plt.title('CIMEDNet')
                    plt.xlim(0, 100)
                    plt.ylim(0, 100)
                    # plt.figure(figsize = (fig_size, fig_size))
                    plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                    plt.scatter(cimednet_y_plot[:, 0], cimednet_y_plot[:, 1], c = 'c', s = ls)
                    plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                    
                if cimednet_L_name != None: 
                    # plt.subplot(rows, 1, 4)
                    plt.subplot(1, rows, 4)
                    # plt.title('CIMEDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                    # plt.title('CIMEDNet')
                    plt.xlim(0, 100)
                    plt.ylim(0, 100)
                    # plt.figure(figsize = (fig_size, fig_size))
                    plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                    plt.scatter(cimednet_L_y_plot[:, 0], cimednet_L_y_plot[:, 1], c = 'c', s = ls)
                    plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
            elif dsdnet_y.shape[1] == 3:
                img = img[:, ::-1]
                # plt.imshow(img, cmap = 'gray')
                # plt.show()
                
                dsdnet_y_plot = deepcopy(dsdnet_y)
                if cimednet_name != None: cimednet_y_plot = deepcopy(cimednet_y)
                if cimednet_L_name != None: cimednet_L_y_plot = deepcopy(cimednet_L_y)
                original_traj_plot = deepcopy(original_traj)
                
                if 'rotation_degrees' in outputs:
                    dsdnet_y_plot = rot3d(np.array([0., 0., 0.]), dsdnet_y_plot, -outputs['rotation_degrees'], order = outputs['rotation_order'][::-1])
                    if cimednet_name != None: cimednet_y_plot = rot3d(np.array([0., 0., 0.]), cimednet_y_plot, -outputs['rotation_degrees'], order = outputs['rotation_order'][::-1])
                    if cimednet_L_name != None: cimednet_L_y_plot = rot3d(np.array([0., 0., 0.]), cimednet_L_y_plot, -outputs['rotation_degrees'], order = outputs['rotation_order'][::-1])
                    original_traj_plot = rot3d(np.array([0., 0., 0.]), original_traj_plot, -outputs['rotation_degrees'], order = outputs['rotation_order'][::-1])
                
                # dsdnet_y_plot[:,2] = dsdnet_y_plot[::-1,2]
                # if cimednet_name != None: cimednet_y_plot[:,2] = cimednet_y_plot[::-1,2]
                # if cimednet_L_name != None: cimednet_L_y_plot[:,2] = cimednet_L_y_plot[::-1,2]
                # original_traj_plot[:,2] = original_traj_plot[::-1,2]
            
                dsdnet_y_plot[:,2] -= dsdnet_y_plot[0,2]
                if cimednet_name != None: cimednet_y_plot[:,2] -= cimednet_y_plot[0,2]
                if cimednet_L_name != None: cimednet_L_y_plot[:,2] -= cimednet_L_y_plot[0,2]
                original_traj_plot[:,2] -= original_traj_plot[0,2]
                
                SUCCESS_THRESHOLD = 22
                dsdnet_y_plot *= multiplier
                if cimednet_name != None: cimednet_y_plot *= multiplier
                if cimednet_L_name != None: cimednet_L_y_plot *= multiplier
                original_traj_plot *= multiplier
                dsdnet_y_plot += padding
                if cimednet_name != None: 
                    cimednet_y_plot += padding
                if cimednet_L_name != None: 
                    cimednet_L_y_plot += padding
                original_traj_plot += padding
                
                if to_plot == 'image':
                    # plt.figure(figsize = (fig_size, fig_size * 4))
                    plt.figure(figsize = (fig_size * 4, fig_size))
                    
                    plt.subplot(1, 4, 1)
                    # plt.title('Original')
                    plt.xlim(0, 100)
                    plt.ylim(0, 100)
                    plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                    plt.scatter(original_traj_plot[:, 2], original_traj_plot[:, 1], c = 'y', s = ls)
                    plt.scatter(original_traj_plot[0, 2], original_traj_plot[0, 1], c = 'r', s = ls*2)
                    # plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                    
                    plt.subplot(1, 4, 2)
                    # plt.title('DSDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                    # plt.title('DSDNet')
                    plt.xlim(0, 100)
                    plt.ylim(0, 100)
                    # plt.figure(figsize = (fig_size, fig_size))
                    plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                    plt.scatter(dsdnet_y_plot[:, 2], dsdnet_y_plot[:, 1], c = 'g', s = ls)
                    plt.scatter(dsdnet_y_plot[0, 2], dsdnet_y_plot[0, 1], c = 'r', s = ls*2)
                    # plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                    
                    if cimednet_name != None: 
                        plt.subplot(1, 4, 3)
                        # plt.title('CIMEDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                        # plt.title('CIMEDNet')
                        plt.xlim(0, 100)
                        plt.ylim(0, 100)
                        # plt.figure(figsize = (fig_size, fig_size))
                        plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                        plt.scatter(cimednet_y_plot[:, 2], cimednet_y_plot[:, 1], c = 'c', s = ls)
                        plt.scatter(cimednet_y_plot[0, 2], cimednet_y_plot[0, 1], c = 'r', s = ls*2)
                        # plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                    
                    if cimednet_L_name != None: 
                        plt.subplot(1, 4, 4)
                        # plt.title('CIMEDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                        # plt.title('CIMEDNet')
                        plt.xlim(0, 100)
                        plt.ylim(0, 100)
                        # plt.figure(figsize = (fig_size, fig_size))
                        plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                        plt.scatter(cimednet_L_y_plot[:, 2], cimednet_L_y_plot[:, 1], c = 'r', s = ls)
                        plt.scatter(cimednet_L_y_plot[0, 2], cimednet_L_y_plot[0, 1], c = 'r', s = ls*2)
                        # plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                    plt.show()
                
                elif to_plot == 'pos':
                    plt.figure(figsize = (fig_size, fig_size/2 * 3))
                    plt.subplot(3, 1, 1)
                    plt.title('Real data motion reconstruction')
                    plt.xlabel('time(s)')
                    plt.ylabel('Forward-backward motion')
                    plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 0], c = 'y', s = ls)
                    plt.scatter(range(original_traj_plot.shape[0]), dsdnet_y_plot[:, 0], c = 'g', s = ls)
                    if cimednet_name != None: plt.scatter(range(original_traj_plot.shape[0]), cimednet_y_plot[:, 0], c = 'c', s = ls)
                    if cimednet_L_name != None: plt.scatter(range(original_traj_plot.shape[0]), cimednet_L_y_plot[:, 0], c = 'r', s = ls)
                    plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
                    
                    plt.subplot(3, 1, 2)
                    plt.xlabel('time(s)')
                    plt.ylabel('Up-down motion')
                    plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 1], c = 'y', s = ls)
                    plt.scatter(range(original_traj_plot.shape[0]), dsdnet_y_plot[:, 1], c = 'g', s = ls)
                    if cimednet_name != None: plt.scatter(range(original_traj_plot.shape[0]), cimednet_y_plot[:, 1], c = 'c', s = ls)
                    if cimednet_L_name != None: plt.scatter(range(original_traj_plot.shape[0]), cimednet_L_y_plot[:, 1], c = 'r', s = ls)
                    plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
                    
                    plt.subplot(3, 1, 3)
                    plt.xlabel('time(s)')
                    plt.ylabel('Right-left motion')
                    plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 2], c = 'y', s = ls)
                    plt.scatter(range(original_traj_plot.shape[0]), dsdnet_y_plot[:, 2], c = 'g', s = ls)
                    if cimednet_name != None: plt.scatter(range(original_traj_plot.shape[0]), cimednet_y_plot[:, 2], c = 'c', s = ls)
                    if cimednet_L_name != None: plt.scatter(range(original_traj_plot.shape[0]), cimednet_L_y_plot[:, 2], c = 'r', s = ls)
                    plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
                    plt.show()
                
                # plt.figure(figsize = (fig_size, fig_size))
                # plt.scatter(original_traj_plot[:, 2], original_traj_plot[:, 1], c = 'y', s = ls)
                # plt.scatter(dsdnet_y_plot[:, 2], dsdnet_y_plot[:, 1], c = 'g', s = ls)
                # plt.scatter(cimednet_y_plot[:, 2], cimednet_y_plot[:, 1], c = 'c', s = ls)
        # elif task == 'cutting-real':
        #     dsdnet_y_plot = deepcopy(dsdnet_y)
        #     cimednet_y_plot = deepcopy(cimednet_y)
        #     cimednet_L_y_plot = deepcopy(cimednet_L_y)
            
            
            
        #     original_traj_plot = deepcopy(original_traj)
            
        #     img_to_plot = deepcopy(img)
        #     # img_to_plot = img_to_plot[:,::-1]
        #     plt.scatter(range(dsdnet_y_plot.shape[0]), dsdnet_y_plot[:, 0])
        #     plt.scatter(range(cimednet_y_plot.shape[0]), cimednet_y_plot[:, 0])
        #     plt.scatter(range(cimednet_L_y_plot.shape[0]), cimednet_L_y_plot[:, 0])
        #     plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 0])
        
        elif task == 'pickplace':
            
            dsdnet_fail = False  if np.abs(dsdnet_y).max() < pickplacegen.map_dim[0] * 1.5 else True
            pass_str = 'DSDNetV1       = {}\n'.format('Fail' if dsdnet_fail else 'Pass')
            map_dsdnet = pickplacegen.map
            
            if cimednet_name != None: 
                cimednet_fail = False if np.abs(cimednet_y).max() < pickplacegen.map_dim[0] * 1.5 else True
                pass_str += 'CIMEDNet({})   = {}\n'.format(cimednet_w.shape[1], 'Fail' if cimednet_fail else 'Pass')
                map_cimednet = copy(map_dsdnet)
                
            if cimednet_L_name != None:
                cimednet_L_fail = False if np.abs(cimednet_L_y).max() < pickplacegen.map_dim[0] * 1.5 else True
                pass_str += 'CIMEDNet({})   = {}\n'.format(cimednet_L_w.shape[1], 'Fail' if cimednet_L_fail else 'Pass')
                map_cimednet_L = copy(map_dsdnet)
            
            print(pass_str)
            
            if not dsdnet_fail:
                # if not cimednet_fail or not cimednet_L_fail: input('enter for dsdnet')
                for i in range(dsdnet_y.shape[0]):
                    # print('{}/{}'.format(i+1, dsdnet_y.shape[0]))
                    
                    if not dsdnet_fail:
                        map_dsdnet.moveGripperXYZ(new_xy = dsdnet_y[i,:2], new_z = dsdnet_y[i,2])
                        map_dsdnet.gripper.distance = dsdnet_y[i,3]
                        map_dsdnet.checkObject(threshold = GRIP_THRESHOLD)                
                    
                    if i%PLOT_STEP == 0 or i == dsdnet_y.shape[0] - 1:
                        map_dsdnet.plot(show_gripper = True, show_trail = False)
                if SHOW_SIDE: map_dsdnet.plot_side()
                #%
            
            if cimednet_name != None and not cimednet_fail:
                input('enter for cimednet')
                for i in range(cimednet_y.shape[0]):
                    # print('{}/{}'.format(i+1, dsdnet_y.shape[0]))
                    
                    if not cimednet_fail:
                        map_cimednet.moveGripperXYZ(new_xy = cimednet_y[i, :2], new_z = cimednet_y[i,2])
                        map_cimednet.gripper.distance = cimednet_y[i,3]
                        map_cimednet.checkObject(threshold = GRIP_THRESHOLD)                
                    
                    if i%PLOT_STEP == 0 or i == dsdnet_y.shape[0] - 1:
                        map_cimednet.plot(show_gripper = True, show_trail = False)
                if SHOW_SIDE: map_cimednet.plot_side()
            
            if cimednet_L_name != None and not cimednet_L_fail:
                input('enter for cimednet-accurate')
                for i in range(cimednet_L_y.shape[0]):
                    # print('{}/{}'.format(i+1, dsdnet_y.shape[0]))
                    
                    if not cimednet_L_fail:
                        map_cimednet_L.moveGripperXYZ(new_xy = cimednet_L_y[i, :2], new_z = cimednet_L_y[i,2])
                        map_cimednet_L.gripper.distance = cimednet_L_y[i,3]
                        map_cimednet_L.checkObject(threshold = GRIP_THRESHOLD)
                    
                    
                    if i%PLOT_STEP == 0 or i == dsdnet_y.shape[0] - 1:
                        map_cimednet_L.plot(show_gripper = True, show_trail = False)
                if SHOW_SIDE: map_cimednet_L.plot_side()
                
            legend = ['DSDNet']
            if cimednet_name != None: legend.append('CIMEDNet({})'.format(cimednet_w.shape[1]))
            if cimednet_L_name != None: legend.append('CIMEDNet({})'.format(cimednet_L_w.shape[1]))
                
            if SHOW_POS:
                traj = dsdnet_y
                plt.figure(figsize = (fig_size*1.25, fig_size * 1.75))
                plt.subplot(4, 1, 1)
                plt.ylim(-10, 110)
                plt.scatter(range(traj.shape[0]), traj[:, 0], c = 'g')
                plt.subplot(4, 1, 2)
                plt.ylim(-10, 110)
                plt.scatter(range(traj.shape[0]), traj[:, 1], c = 'g')
                plt.subplot(4, 1, 3)
                plt.ylim(-10, 110)
                plt.scatter(range(traj.shape[0]), traj[:, 2], c = 'g')
                plt.subplot(4, 1, 4)
                plt.ylim(-10, 110)
                plt.scatter(range(traj.shape[0]), traj[:, 3], c = 'g')
                # plt.show()
                
                if cimednet_name != None:
                    traj = cimednet_y
                    # plt.figure(figsize = (fig_size * 1.5, fig_size * 3))
                    plt.subplot(4, 1, 1)
                    plt.scatter(range(traj.shape[0]), traj[:, 0], c = 'c')
                    plt.subplot(4, 1, 2)
                    plt.scatter(range(traj.shape[0]), traj[:, 1], c = 'c')
                    plt.subplot(4, 1, 3)
                    plt.scatter(range(traj.shape[0]), traj[:, 2], c = 'c')
                    plt.subplot(4, 1, 4)
                    plt.scatter(range(traj.shape[0]), traj[:, 3], c = 'c')
                
                if cimednet_L_name != None:
                    traj = cimednet_L_y
                    # plt.figure(figsize = (fig_size * 1.5, fig_size * 3))
                    plt.subplot(4, 1, 1)
                    plt.scatter(range(traj.shape[0]), traj[:, 0], c = 'r')
                    plt.legend(legend)
                    plt.subplot(4, 1, 2)
                    plt.scatter(range(traj.shape[0]), traj[:, 1], c = 'r')
                    plt.legend(legend)
                    plt.subplot(4, 1, 3)
                    plt.scatter(range(traj.shape[0]), traj[:, 2], c = 'r')
                    plt.legend(legend)
                    plt.subplot(4, 1, 4)
                    plt.scatter(range(traj.shape[0]), traj[:, 3], c = 'r')
                    plt.legend(legend)
                    
                plt.show()
                    
            # plt.imshow(img, cmap = 'gray')
            # plt.show()
            
        
        # new_pad = input('{}. Change padding ({})?'.format(test_idx + 1, padding[2]))
        # if new_pad == '':
        #     test_idx += 1
        #     padding = deepcopy(original_padding)
        # else:
        #     padding[2] = new_pad
        # loop = input('Enter to generate next data')
        test_idx += 1