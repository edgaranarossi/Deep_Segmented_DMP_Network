#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 22:22:00 2022

@author: edgar
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


def rot3d(origin, traj, degrees, order = None):
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

def rot2D(traj, degrees, origin = np.array([0., 0.])):
    
    s = sin(degrees)
    c = cos(degrees)
    
    traj -= origin
    
    t = deepcopy(traj)
    t[:, 0] = traj[:, 0] * c - traj[:, 1] * s
    t[:, 1] = traj[:, 0] * s + traj[:, 1] * c
    
    t += origin
    
    return t

def sampling_rmse(y1, y2):
    # print(y1.shape[0], y2.shape[0])
    if y1.shape[0] > y2.shape[0]:
        y_l = deepcopy(y1)
        y_s = deepcopy(y2)
        pattern = 1
    else:
        y_l = deepcopy(y2)
        y_s = deepcopy(y1)
        pattern = 0
        
    higher_sampler = y_l.shape[0] // y_s.shape[0]
    lower_sampler = higher_sampler + 1
    
    used_idx = [i for i in range(0, y_l.shape[0], lower_sampler)]
    higher_sampled_idx = np.array([i for i in range(0, y_l.shape[0], higher_sampler)])
    
    while len(used_idx) < y_s.shape[0]:
        missing_idx = np.array([i for i in higher_sampled_idx if i not in used_idx])
        to_add = y_s.shape[0] - len(used_idx)
        
        
        higher_sampler = len(missing_idx) // to_add
        lower_sampler = higher_sampler + 1
        
        to_add = np.array(missing_idx)[::lower_sampler].tolist()
        for i in to_add:
            used_idx.append(i)
    used_idx = np.array(sorted(used_idx))
    sampled_y_l = y_l[used_idx]
    
    rmse = np.sqrt(((y_s - sampled_y_l)**2).mean())
    
    if pattern:
        return rmse, sampled_y_l, y_s
    else:
        return rmse, y_s, sampled_y_l
    
def generate_cutting_image_motion():
    IMG_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/images/cutting/task_test'
    
    base_shape = array([[0.25, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.5 + (0.5 * rand())],
                        [1.0, 0.5 + (0.5 * rand())],
                        [1.0, 0.0],
                        [0.75, 0.0]])
    gen = ObjectGenerator(base_shape = base_shape)
    
    save_path = join(IMG_DIR, 'image.jpg')
    x = gen.generate(size_random_magnitude = (1., .5), 
                     shape_random_magnitude = (randint(0, 3)/100, randint(0, 6)/100), 
                     smoothing_magnitude = randint(0, 4),
                     plot_shape = False,
                     plot_save_path = save_path,
                     # plot_target_size = (50, 50))
                     plot_target_size = None)
    img = array(Image.open(save_path).convert("L").resize((100, 100)))/255
    
    
    x_min = x.min(axis = 0)[0]
    x_max = x.max(axis = 0)[0]

    top_left = np.where(x[:, 0] == x_min, x[:, 1], 0).max()
    top_right = np.where(x[:, 0] == x_max, x[:, 1], 0).max()

    ys, segs, dmps, num_cut, dmp_fair, dmp_accurate, original_trajectory, segment_dmp_trajectory = generate_cutting_traj(top_left = (x_min, top_left),
                                                                                                                         top_right = (x_max, top_right),
                                                                                                                         distance = 0.2,
                                                                                                                         top_padding = 0.2,
                                                                                                                         side_padding = 0.05,
                                                                                                                         max_segments = int(1.5 // 0.2),
                                                                                                                         dmp_bf = None,
                                                                                                                         dmp_ay = None,
                                                                                                                         dmp_dt = None, 
                                                                                                                         dmp_L_bf = None,
                                                                                                                         dmp_L_ay = None,
                                                                                                                         dmp_L_dt = None,
                                                                                                                         seg_dmp_bf = 50,
                                                                                                                         seg_dmp_ay = 15,
                                                                                                                         seg_dmp_dt = 0.003)
    
    input_image = torch.tensor(img).to(DEVICE).float().reshape(1, 1, img.shape[0], img.shape[0])
    
    return img, input_image, segment_dmp_trajectory

def generate_pickplace_image_motion(task, num_object = None):
    root_data_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data'
    task_test_img_dir = join(root_data_dir, 'images/stacking/task_test')
    task_test_pkl_dir = join(root_data_dir, 'images/stacking/task_test')
    if task == 'pickplace':
        pickplacegen = PickAndPlaceGenerator(task_test_img_dir, task_test_pkl_dir, [1, 2, 3], img_size = (150, 150), permute_block_pos = False, randomize_block_pos = False, randomize_target_pos = False, pos_noise_magnitude = 0, motion_noise_magnitude = 0)
    elif task == 'pickplace-rand':
        pickplacegen = PickAndPlaceGenerator(task_test_img_dir, task_test_pkl_dir, [1, 2, 3], img_size = (150, 150), permute_block_pos = False, randomize_block_pos = True, randomize_target_pos = True, pos_noise_magnitude = 0, motion_noise_magnitude = 0)
        
    img, y_label = pickplacegen.generateTestData(num_object = num_object)
    
    # pickplacegen.map.plot(show_gripper = False, show_trail = False)
    img = img.reshape(1, img.shape[-1], img.shape[0], img.shape[1]) / 255
    input_image = torch.tensor(img).to(DEVICE).float()
    
    return img, input_image, y_label, pickplacegen

def read_test_data(dataset, test_idx):        
    original_padding = np.array([50, 20, 23])
    # original_padding = np.array([0, 0, 0])
    padding = deepcopy(original_padding)
    multiplier = np.array([300, 650, -250])
    
    to_plot = 'image'
    # to_plot = 'pos'
    
    img = dataset[test_idx][0]['image'].reshape(dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2], dsdnet_model_param.image_dim[0]).detach().cpu().numpy()
    input_image = dataset[test_idx][0]['image'].reshape(1, dsdnet_model_param.image_dim[0], dsdnet_model_param.image_dim[1], dsdnet_model_param.image_dim[2])
    outputs = dataset[test_idx][1]

    num_cuts = outputs['segmented_dmp_seg_num']
    num_cuts = (int(scaler['segmented_dmp_seg_num'].denormalize(num_cuts).item()) // 2) + 1
    
    if 'original_trajectory' in outputs: 
        y_label = str_to_ndarray(outputs['original_trajectory'])
    if 'rotated_trajectory' in outputs: 
        y_label = str_to_ndarray(outputs['rotated_trajectory'])
    # if 'processed_trajectory' in outputs: 
        # y_label = str_to_ndarray(outputs['processed_trajectory'])
    # if 'segmented_dmp_trajectory' in outputs: 
    #     y_label = str_to_ndarray(outputs['segmented_dmp_trajectory'])
    # if 'normal_dmp_trajectory' in outputs: 
    #     normal_dmp_traj = str_to_ndarray(outputs['normal_dmp_trajectory'])
    
    if 'rotation_degrees' in outputs:
        rot_degrees = outputs['rotation_degrees']
        rot_order = outputs['rotation_order']
    elif 'rot2D' in outputs:
        rot_degrees = outputs['rot2D']
        rot_order = None
    else:
        rot_degrees = None
        rot_order = None
    
    return img, input_image, y_label, rot_degrees, rot_order
    #%%
if __name__=='__main__':
    runcell(0, '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/scripts/task_tester.py')
    # task = 'cutting'
    # task = 'cutting-limited'
    # task = 'cutting-real'
    # task = 'pickplace'
    # task = 'pickplace-rand'
    task = 'pepper_shaking'

    if task == 'cutting':
        dsdnet_name = 'Model_DSDNetV1_2022-09-12_07-23-54'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_07-25-28'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_07-25-46'
        to_test = 100
    elif task == 'cutting-limited':
        dsdnet_name = 'Model_DSDNetV1_2022-09-11_22-51-01'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-11_22-52-23'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-11_22-54-15'
        to_test = 100
        # task = 'cutting'
    elif task == 'cutting-real':
        dsdnet_name = 'Model_DSDNetV1_2022-09-12_02-23-40'
        # cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_02-24-02'
        # cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_02-24-38'
        # task = 'cutting'
    elif task == 'pickplace':
        dsdnet_name = 'Model_DSDNetV1_2022-09-11_00-27-13'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_17-20-24'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_17-21-32'
        to_test = 3
        num_objects = [1]*1 + [2]*1 + [3]*1
        
    elif task == 'pickplace-rand':
        dsdnet_name = 'Model_DSDNetV1_2022-09-13_20-31-23'
        cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-09-12_00-52-09'
        cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-09-12_17-15-57'
        to_test = 99
        num_objects = [1]*33 + [2]*33 + [3]*33
        
    elif task == 'pepper_shaking':
        dsdnet_name = 'Model_DSDNetV1_2022-12-02_16-39-11'
        # cimednet_name = None
        cimednet_name = 'Model_CIMEDNet_2022-12-02_16-39-51'
        # cimednet_L_name = None
        cimednet_L_name = 'Model_CIMEDNet_2022-12-02_16-39-55'


    root_model_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'

    dsdnet_dir = join(root_model_dir, dsdnet_name.split('_')[1], dsdnet_name)
    if cimednet_name != None: cimednet_dir = join(root_model_dir, cimednet_name.split('_')[1], cimednet_name)
    if cimednet_L_name != None: cimednet_L_dir = join(root_model_dir, cimednet_L_name.split('_')[1], cimednet_L_name)
    if task == 'cutting-real' or task == 'pepper_shaking':
        train_loader, val_loader, test_loader = pkl.load(open(join(dsdnet_dir, 'data_loaders.pkl'), 'rb'))
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
        
        used_dataset = test_dataset
        # used_dataset = val_dataset
        to_test = len(used_dataset)
        print('\nDataset length = {}\n'.format(to_test))
        # print('')
        
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
    fig_size = 5
    ls = 6
    DSDNetV1_RMSEs = []
    CIMEDNet_RMSEs = []
    CIMEDNet_L_RMSEs = []
    
    DSDNetV1_DTWs = []
    CIMEDNet_DTWs = []
    CIMEDNet_L_DTWs = []

    PLOT_STEP = 10000
    GRIP_THRESHOLD = 5
    SHOW_SIDE = True
    SHOW_POS = True
    
    idx = 2
    connect_segment = False
    # connect_segment = True
    
    # while idx < to_test:
    if 1:
        # INFER
        if task in ['pickplace', 'pickplace-rand']:
            img, input_image, y_label, pickplacegen = generate_pickplace_image_motion(task, num_object=num_objects[idx])
            # img, input_image, y_label = generate_pickplace_image_motion(task, num_object=1)
            # img, input_image, y_label = generate_pickplace_image_motion(task, num_object=2)
            # img, input_image, y_label = generate_pickplace_image_motion(task, num_object=3)
            
        elif task in ['cutting', 'cutting-limited']:
            img, input_image, y_label = generate_cutting_image_motion()
            
        elif task == 'cutting-real':
            img, input_image, y_label, rot_degrees, rot_order = read_test_data(dataset = used_dataset, test_idx = idx)
            # to_plot = 'image'
            to_plot = 'pos'
            original_padding = np.array([50, 20, 23])
            # original_padding = np.array([0, 0, 0])
            padding = deepcopy(original_padding)
            multiplier = np.array([300, 650, -250])
            
        elif task == 'pepper_shaking':
            img, input_image, y_label, rot_degrees, rot_order = read_test_data(dataset = used_dataset, test_idx = idx)
            # to_plot = 'image'
            to_plot = 'pos'
            original_padding = np.array([50, 20])
            # original_padding = np.array([0, 0, 0])
            padding = deepcopy(original_padding)
            multiplier = np.array([300, 650])
            
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
        if connect_segment: dsdnet_goal[:-1] = dsdnet_y0[1:]
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
        
        # if y_label.shape[0] < cimednet_y.shape[0] and y_label.shape[0] < cimednet_L_y.shape[0]:
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
        # elif dsdnet_y.shape[0] < cimednet_y.shape[0] and dsdnet_y.shape[0] < cimednet_L_y.shape[0]:
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
        # elif cimednet_y.shape[0] < dsdnet_y.shape[0] and cimednet_y.shape[0] < cimednet_L_y.shape[0]:
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
        # elif cimednet_L_y.shape[0] < dsdnet_y.shape[0] and cimednet_L_y.shape[0] < cimednet_y.shape[0]:
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
            
        # RECONSTRUCT AND SAMPLE
        dsdnet_y = []
        for i in range(min(dsdnet_num_seg, dsdnet_y0.shape[0])):
            dmp = DMPs_discrete(n_dmps = dsdnet_w.shape[-2], n_bfs = dsdnet_w.shape[-1], ay = np.ones(dsdnet_w.shape[-2]) * dsdnet_ay, dt = dsdnet_dt)
            dmp.y0 = dsdnet_y0[i]
            dmp.goal = dsdnet_goal[i]
            dmp.w = dsdnet_w[i]
            # y, _, _ = dmp.rollout()
            y, _, _ = dmp.rollout(tau = dsdnet_tau[i])
            dsdnet_y.append(y)
        # dsdnet_y = np.array(dsdnet_y).reshape(-1, dsdnet_w.shape[-2])
        full_y = dsdnet_y[0]
        for y in dsdnet_y[1:]:
            full_y = np.append(full_y, y, axis = 0)
        dsdnet_y = full_y
        dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
        DSDNetV1_RMSEs.append(dsdnet_rmse)
        dsdnet_dtw = dtw.distance(y_label[:, 0], dsdnet_y[:, 0]) + dtw.distance(y_label[:, 1], dsdnet_y[:, 1])
        if y_label.shape[1] > 2:
            dsdnet_dtw += dtw.distance(y_label[:, 2], dsdnet_y[:, 2])
        DSDNetV1_DTWs.append(dsdnet_dtw)
        if y_label.shape[1] > 3:
            dsdnet_dtw += dtw.distance(y_label[:, 3], dsdnet_y[:, 3])
        DSDNetV1_DTWs.append(dsdnet_dtw)
        
        if cimednet_name != None: 
            dmp = DMPs_discrete(n_dmps = cimednet_w.shape[-2], n_bfs = cimednet_w.shape[-1], ay = np.ones(cimednet_w.shape[-2]) * cimednet_ay, dt = cimednet_dt)
            dmp.y0 = cimednet_y0
            dmp.goal = cimednet_goal
            dmp.w = cimednet_w
            cimednet_y, _, _ = dmp.rollout()
            cimednet_rmse, sampled_y_label, sampled_cimednet_y = sampling_rmse(y_label, cimednet_y)
            CIMEDNet_RMSEs.append(cimednet_rmse)
            dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(sampled_y_label, sampled_dsdnet_y)
            cimednet_dtw = dtw.distance(y_label[:, 0], cimednet_y[:, 0]) + dtw.distance(y_label[:, 1], cimednet_y[:, 1])
            if y_label.shape[1] > 2:
                cimednet_dtw += dtw.distance(y_label[:, 2], cimednet_y[:, 2])
            DSDNetV1_DTWs.append(dsdnet_dtw)
            if y_label.shape[1] > 3:
                cimednet_dtw += dtw.distance(y_label[:, 3], cimednet_y[:, 3])
            CIMEDNet_DTWs.append(cimednet_dtw)
            
        if cimednet_L_name != None: 
            dmp = DMPs_discrete(n_dmps = cimednet_L_w.shape[-2], n_bfs = cimednet_L_w.shape[-1], ay = np.ones(cimednet_L_w.shape[-2]) * cimednet_L_ay, dt = cimednet_L_dt)
            dmp.y0 = cimednet_L_y0
            dmp.goal = cimednet_L_goal
            dmp.w = cimednet_L_w
            cimednet_L_y, _, _ = dmp.rollout()
            cimednet_L_rmse, sampled_y_label, sampled_cimednet_L_y = sampling_rmse(y_label, cimednet_L_y)
            CIMEDNet_L_RMSEs.append(cimednet_L_rmse)
            dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(sampled_y_label, sampled_dsdnet_y)
            cimednet_rmse, sampled_y_label, sampled_cimednet_y = sampling_rmse(sampled_y_label, sampled_cimednet_y)
            cimednet_L_dtw = dtw.distance(y_label[:, 0], cimednet_L_y[:, 0]) + dtw.distance(y_label[:, 1], cimednet_L_y[:, 1])
            if y_label.shape[1] > 2:
                cimednet_L_dtw += dtw.distance(y_label[:, 2], cimednet_L_y[:, 2])
            DSDNetV1_DTWs.append(dsdnet_dtw)
            if y_label.shape[1] > 3:
                cimednet_L_dtw += dtw.distance(y_label[:, 3], cimednet_L_y[:, 3])
            CIMEDNet_L_DTWs.append(cimednet_L_dtw)
            
        # if dsdnet_y.shape[0] < cimednet_y.shape[0] and dsdnet_y.shape[0] < cimednet_L_y.shape[0]:
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
        #     dsdnet_rmse, sampled_y_label, sampled_dsdnet_y = sampling_rmse(y_label, dsdnet_y)
    
        # TASK EVALUATION
        if task in ['pickplace', 'pickplace-rand']:
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
        elif task in ['cutting', 'cutting-limited']:
            dsdnet_y_plot = deepcopy(dsdnet_y)
            if cimednet_name != None: cimednet_y_plot = deepcopy(cimednet_y)
            if cimednet_L_name != None: cimednet_L_y_plot = deepcopy(cimednet_L_y)
            original_traj_plot = deepcopy(sampled_y_label)
            
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
            # plt.title('{} {}'.format(task, idx+1))
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.tick_params(bottom=False,     
                            top=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)
            plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
            plt.scatter(original_traj_plot[:, 0], original_traj_plot[:, 1], c = 'y', s = ls)
            
            # plt.subplot(rows, 1, 2)
            plt.subplot(1, rows, 2)
            # plt.title('DSDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
            # plt.title('DSDNet')
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.tick_params(bottom=False,     
                            top=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)
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
                plt.tick_params(bottom=False,     
                                top=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)
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
                plt.tick_params(bottom=False,     
                                top=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)
                # plt.figure(figsize = (fig_size, fig_size))
                plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                plt.scatter(cimednet_L_y_plot[:, 0], cimednet_L_y_plot[:, 1], c = 'c', s = ls)
                plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
            
            plt.show()
                
        elif task in ['cutting-real']:
            img = img[:, ::-1]
            # plt.imshow(img, cmap = 'gray')
            # plt.show()
            
            dsdnet_y_plot = deepcopy(sampled_dsdnet_y)
            if cimednet_name != None: cimednet_y_plot = deepcopy(sampled_cimednet_y)
            if cimednet_L_name != None: cimednet_L_y_plot = deepcopy(sampled_cimednet_L_y)
            original_traj_plot = deepcopy(y_label)
            
            if rot_degrees is not None:
                dsdnet_y_plot = rot3d(np.array([0., 0., 0.]), dsdnet_y_plot, -rot_degrees, order = rot_order[::-1])
                if cimednet_name != None: cimednet_y_plot = rot3d(np.array([0., 0., 0.]), cimednet_y_plot, -rot_degrees, order = rot_order[::-1])
                if cimednet_L_name != None: cimednet_L_y_plot = rot3d(np.array([0., 0., 0.]), cimednet_L_y_plot, -rot_degrees, order = rot_order[::-1])
                original_traj_plot = rot3d(np.array([0., 0., 0.]), original_traj_plot, -rot_degrees, order = rot_order[::-1])
            
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
                plt.tick_params(bottom=False,     
                                top=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)
                plt.imshow(img[::-1], cmap = 'gray', origin = 'lower')
                plt.scatter(original_traj_plot[:, 2], original_traj_plot[:, 1], c = 'y', s = ls)
                plt.scatter(original_traj_plot[0, 2], original_traj_plot[0, 1], c = 'r', s = ls*2)
                # plt.plot([0, img.shape[0]], [SUCCESS_THRESHOLD, SUCCESS_THRESHOLD], c = 'r', ls = '--')
                
                plt.subplot(1, 4, 2)
                # plt.title('DSDNet - {} - {} cuts'.format(test_idx + 1, num_cuts))
                # plt.title('DSDNet')
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                plt.tick_params(bottom=False,     
                                top=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)
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
                    plt.tick_params(bottom=False,     
                                    top=False,
                                    left=False,
                                    labelbottom=False,
                                    labelleft=False)
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
                    plt.tick_params(bottom=False,     
                                    top=False,
                                    left=False,
                                    labelbottom=False,
                                    labelleft=False)
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
                plt.scatter(range(dsdnet_y_plot.shape[0]), dsdnet_y_plot[:, 0], c = 'g', s = ls)
                if cimednet_name != None: plt.scatter(range(cimednet_y_plot.shape[0]), cimednet_y_plot[:, 0], c = 'c', s = ls)
                if cimednet_L_name != None: plt.scatter(range(cimednet_L_y_plot.shape[0]), cimednet_L_y_plot[:, 0], c = 'r', s = ls)
                plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
                
                plt.subplot(3, 1, 2)
                plt.xlabel('time(s)')
                plt.ylabel('Up-down motion')
                plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 1], c = 'y', s = ls)
                plt.scatter(range(dsdnet_y_plot.shape[0]), dsdnet_y_plot[:, 1], c = 'g', s = ls)
                if cimednet_name != None: plt.scatter(range(cimednet_y_plot.shape[0]), cimednet_y_plot[:, 1], c = 'c', s = ls)
                if cimednet_L_name != None: plt.scatter(range(cimednet_L_y_plot.shape[0]), cimednet_L_y_plot[:, 1], c = 'r', s = ls)
                plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
                
                plt.subplot(3, 1, 3)
                plt.xlabel('time(s)')
                plt.ylabel('Right-left motion')
                plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 2], c = 'y', s = ls)
                plt.scatter(range(dsdnet_y_plot.shape[0]), dsdnet_y_plot[:, 2], c = 'g', s = ls)
                if cimednet_name != None: plt.scatter(range(cimednet_y_plot.shape[0]), cimednet_y_plot[:, 2], c = 'c', s = ls)
                if cimednet_L_name != None: plt.scatter(range(cimednet_L_y_plot.shape[0]), cimednet_L_y_plot[:, 2], c = 'r', s = ls)
                plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
                plt.show()
                    
        elif task in ['pepper_shaking']:
            img = img[:, ::-1]
            # plt.imshow(img, cmap = 'gray')
            # plt.show()
            
            dsdnet_y_plot = deepcopy(sampled_dsdnet_y)
            if cimednet_name != None: cimednet_y_plot = deepcopy(sampled_cimednet_y)
            if cimednet_L_name != None: cimednet_L_y_plot = deepcopy(sampled_cimednet_L_y)
            original_traj_plot = deepcopy(y_label)
            
            if rot_degrees is not None:
                dsdnet_y_plot = rot2D(dsdnet_y_plot, -rot_degrees)
                if cimednet_name != None: cimednet_y_plot = rot2D(cimednet_y_plot, -rot_degrees)
                if cimednet_L_name != None: cimednet_L_y_plot = rot2D(cimednet_L_y_plot, -rot_degrees)
                original_traj_plot = rot2D(original_traj_plot, -rot_degrees)
            
            # dsdnet_y_plot[:,2] = dsdnet_y_plot[::-1,2]
            # if cimednet_name != None: cimednet_y_plot[:,2] = cimednet_y_plot[::-1,2]
            # if cimednet_L_name != None: cimednet_L_y_plot[:,2] = cimednet_L_y_plot[::-1,2]
            # original_traj_plot[:,2] = original_traj_plot[::-1,2]
        
            # dsdnet_y_plot[:,2] -= dsdnet_y_plot[0,2]
            # if cimednet_name != None: cimednet_y_plot[:,2] -= cimednet_y_plot[0,2]
            # if cimednet_L_name != None: cimednet_L_y_plot[:,2] -= cimednet_L_y_plot[0,2]
            # original_traj_plot[:,2] -= original_traj_plot[0,2]
            
            SUCCESS_THRESHOLD = 22
            # dsdnet_y_plot *= multiplier
            if cimednet_name != None: cimednet_y_plot *= multiplier
            if cimednet_L_name != None: cimednet_L_y_plot *= multiplier
            # original_traj_plot *= multiplier
            # dsdnet_y_plot += padding
            if cimednet_name != None: 
                cimednet_y_plot += padding
            if cimednet_L_name != None: 
                cimednet_L_y_plot += padding
            # original_traj_plot += padding
            
            plt.figure(figsize = (fig_size, fig_size/2 * 3))
            plt.subplot(3, 1, 1)
            plt.title('Real data motion reconstruction')
            plt.xlabel('time(s)')
            plt.ylabel('Forward-backward motion')
            plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 0], c = 'y', s = ls)
            plt.scatter(range(dsdnet_y_plot.shape[0]), dsdnet_y_plot[:, 0], c = 'g', s = ls)
            if cimednet_name != None: plt.scatter(range(cimednet_y_plot.shape[0]), cimednet_y_plot[:, 0], c = 'c', s = ls)
            if cimednet_L_name != None: plt.scatter(range(cimednet_L_y_plot.shape[0]), cimednet_L_y_plot[:, 0], c = 'r', s = ls)
            plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
            
            plt.subplot(3, 1, 2)
            plt.xlabel('time(s)')
            plt.ylabel('Up-down motion')
            plt.scatter(range(original_traj_plot.shape[0]), original_traj_plot[:, 1], c = 'y', s = ls)
            plt.scatter(range(dsdnet_y_plot.shape[0]), dsdnet_y_plot[:, 1], c = 'g', s = ls)
            if cimednet_name != None: plt.scatter(range(cimednet_y_plot.shape[0]), cimednet_y_plot[:, 1], c = 'c', s = ls)
            if cimednet_L_name != None: plt.scatter(range(cimednet_L_y_plot.shape[0]), cimednet_L_y_plot[:, 1], c = 'r', s = ls)
            plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
            
            plt.subplot(3, 1, 3)
            plt.xlabel('Forward-backward motion')
            plt.ylabel('Up-down motion')
            plt.scatter(original_traj_plot[:, 0], original_traj_plot[:, 1], c = 'y', s = ls)
            plt.scatter(dsdnet_y_plot[:, 0], dsdnet_y_plot[:, 1], c = 'g', s = ls)
            if cimednet_name != None: plt.scatter(cimednet_y_plot[:, 0], cimednet_y_plot[:, 1], c = 'c', s = ls)
            if cimednet_L_name != None: plt.scatter(cimednet_L_y_plot[:, 0], cimednet_L_y_plot[:, 1], c = 'r', s = ls)
            plt.legend(['Original motion', 'DSDNet', 'CIMEDNet(250)', 'CIMEDNet(800)'])
            plt.show()
        
        # if idx+1 < to_test:
        #     input('({}/{}) Press Enter to continue'.format(idx+1, to_test))
        # else:
        #     print('\n({}/{})\n'.format(idx+1, to_test))
        
        print('\n({}/{})\n'.format(idx+1, to_test))
        idx += 1
    
    DSDNetV1_RMSEs_mean = array(DSDNetV1_RMSEs).mean()
    if cimednet_name != None: CIMEDNet_RMSEs_mean = array(CIMEDNet_RMSEs).mean()
    if cimednet_L_name != None: CIMEDNet_L_RMSEs_mean = array(CIMEDNet_L_RMSEs).mean()
    
    DSDNetV1_DTWs_mean = array(DSDNetV1_DTWs).mean()
    if cimednet_name != None: CIMEDNet_DTWs_mean = array(CIMEDNet_DTWs).mean()
    if cimednet_L_name != None: CIMEDNet_L_DTWs_mean = array(CIMEDNet_L_DTWs).mean()
    
    print('DSDNet RMSE = {}'.format(DSDNetV1_RMSEs_mean))
    if cimednet_name != None: print('CIMEDNet RMSE = {}'.format(CIMEDNet_RMSEs_mean))
    if cimednet_L_name != None: print('CIMEDNet_L RMSE = {}'.format(CIMEDNet_L_RMSEs_mean))
    
    print('DSDNet DTW = {}'.format(DSDNetV1_DTWs_mean))
    if cimednet_name != None: print('CIMEDNet DTW = {}'.format(CIMEDNet_DTWs_mean))
    if cimednet_L_name != None: print('CIMEDNet_L DTW = {}'.format(CIMEDNet_L_DTWs_mean))