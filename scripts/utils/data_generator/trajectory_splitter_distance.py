# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:13:59 2022

@author: edgar
"""

import bagpy
from bagpy import bagreader
import numpy as np
from numpy import array, floor, ceil, round, loadtxt, diff, concatenate, sort, zeros, ones, cos, sin, tan
import pandas as pd
from matplotlib import pyplot as plt
from os.path import join, isdir
from os import listdir, makedirs
import shutil
from PIL import Image, ImageOps
from pydmps import DMPs_discrete
from datetime import datetime
import pickle as pkl
import json
from copy import deepcopy
from random import shuffle

def trim_ends(df, upper_limit = 5e-4):
    begin_idx = 0
    end_idx = len(df) - 1
    
    while df['pose.velocity.abs_total'][begin_idx] < upper_limit: begin_idx += 1
    while df['pose.velocity.abs_total'][end_idx] < upper_limit: end_idx -= 1
    
    return df.iloc[begin_idx:end_idx + 1]

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

def detect_split(df, key, min_length = 50, lower_limit = 0, upper_limit = 4e-4):
    splits = []
    split = []
    cur_idx = 0
    while cur_idx < len(df):
        if df[key].iloc[cur_idx] > lower_limit and \
           df[key].iloc[cur_idx] < upper_limit:
            split.append(cur_idx)
        elif cur_idx < len(df) - 1 and \
           df[key].iloc[cur_idx + 1] > lower_limit and \
           df[key].iloc[cur_idx + 1] < upper_limit:
            split.append(cur_idx)
        else:
            if len(split) >= min_length:
                # print(split)
                splits.append(split)
            split = []
        cur_idx += 1
    
    if splits[0][0] < min_length:
        splits = splits[1:]
    
    segments = []
    start_idx = 0
    for split in splits:
        segments.append([0, [start_idx, split[0]]])
        segments.append([99, [split[0], split[-1]]])
        start_idx = split[-1]
    segments.append([0, [start_idx, len(df)]])
    return segments

def bag_to_df(bag_path):
    b = bagreader(bag_path, verbose = False)
    bag_csv = b.message_by_topic('/mocap_pose_topic/knife_marker_pose')
    df = pd.read_csv(bag_csv)
    return df

def df_add_velocity(df):
    x = np.array(df['pose.position.x'])
    y = np.array(df['pose.position.y'])
    z = np.array(df['pose.position.z'])
    
    # traj = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis = 1)
    
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    
    dx = np.concatenate(([dx[0]], dx))
    dy = np.concatenate(([dy[0]], dy))
    dz = np.concatenate(([dz[0]], dz))
    
    df['pose.velocity.x'] = dx
    df['pose.velocity.y'] = dy
    df['pose.velocity.z'] = dz
    df['pose.velocity.abs_x'] = np.abs(dx)
    df['pose.velocity.abs_y'] = np.abs(dy)
    df['pose.velocity.abs_z'] = np.abs(dz)
    df['pose.velocity.abs_xy'] = np.abs(dx) + np.abs(dy)
    df['pose.velocity.abs_yz'] = np.abs(dy) + np.abs(dz)
    df['pose.velocity.abs_xz'] = np.abs(dx) + np.abs(dz)
    df['pose.velocity.abs_total'] = np.abs(dx) + np.abs(dy) + np.abs(dz)
    # df_bag['pose.velocity.min'] = np.min([dx.reshape(-1, 1), dy.reshape(-1, 1), dz.reshape(-1, 1)], axis = 0)
    
    return df

# def plot_segments(df, seg_points):
#     start_idx = 0
#     fig, ax = bagpy.create_fig(7)
#     for seg_point in seg_points:
#         if seg_point[0] == 0:  
#             col = 'b' 
#         elif seg_point[0] == 1:  
#             col = 'g'
#         elif seg_point[0] == 2:  
#             col = 'm'
#         else:  
#             col = 'r' 
            
#         ax[0].scatter(range(seg_point[2][0], seg_point[2][1]),  
#                       df['pose.position.x'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col) 
#         ax[1].scatter(range(seg_point[2][0], seg_point[2][1]),  
#                       df['pose.position.y'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col) 
#         ax[2].scatter(range(seg_point[2][0], seg_point[2][1]),  
#                       df['pose.position.z'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col) 
#         ax[3].scatter(range(seg_point[2][0], seg_point[2][1]),  
#                       df['pose.velocity.abs_total'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col) 
#         if seg_point[0] in [0, 1, 2]: 
#             ax[4].scatter(range(start_idx, start_idx + seg_point[2][1] - seg_point[2][0]),  
#                           df['pose.position.x'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col) 
#             ax[5].scatter(range(start_idx, start_idx + seg_point[2][1] - seg_point[2][0]),  
#                           df['pose.position.y'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col) 
#             ax[6].scatter(range(start_idx, start_idx + seg_point[2][1] - seg_point[2][0]),  
#                           df['pose.position.z'].iloc[seg_point[2][0]:seg_point[2][1]], s = 1, c = col)
             
#             start_idx += seg_point[2][1] - seg_point[2][0]
#     plt.show()
    
def df_to_traj(df):
    x = np.array(df['pose.position.x'])
    y = np.array(df['pose.position.y'])
    z = np.array(df['pose.position.z'])
    traj = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis = 1)
    return traj

def decrease_seg_points(seg_points):
    to_decrease = 0
    seg_points[0][1][1] -= to_decrease
    for i in range(1, len(seg_points)):
        seg_points[i][1][0] -= to_decrease
    return seg_points

def split_seg_points_lowest_vel(df, seg_points, num_split, split_min_distance = 30):
    new_seg_points = []
    for seg_point in seg_points:
        df_seg = df.iloc[seg_point[1][0]:seg_point[1][1]].reset_index(drop=False)
        abs_total_vel_sort_idx = array(df_seg['pose.velocity.abs_total']).argsort()
        while abs_total_vel_sort_idx[0] < split_min_distance: abs_total_vel_sort_idx = abs_total_vel_sort_idx[1:]
        low_idx = []
        idx = 1
        while len(low_idx) < num_split:
            # print(len(abs_total_vel_sort_idx), abs_total_vel_sort_idx[idx], abs(len(abs_total_vel_sort_idx) - abs_total_vel_sort_idx[idx]) < split_min_distance)
            print('#', abs_total_vel_sort_idx[idx], (seg_point[1][1] - seg_point[1][0]), len(abs_total_vel_sort_idx))
            if abs_total_vel_sort_idx[idx] > split_min_distance and \
               abs(len(abs_total_vel_sort_idx) - abs_total_vel_sort_idx[idx]) > split_min_distance and \
               abs(abs_total_vel_sort_idx[idx] - (seg_point[1][1] - seg_point[1][0])) > split_min_distance:
                if len(low_idx) > 0:
                    if abs(abs_total_vel_sort_idx[idx] - low_idx[-1]) > split_min_distance:
                        print(abs_total_vel_sort_idx[idx], (seg_point[1][1] - seg_point[1][0]), len(abs_total_vel_sort_idx))
                        low_idx.append(abs_total_vel_sort_idx[idx])
                else:
                    print(abs_total_vel_sort_idx[idx], (seg_point[1][1] - seg_point[1][0]))
                    low_idx.append(abs_total_vel_sort_idx[idx])
            idx += 1
        low_idx = sorted(low_idx)
        # print(seg_point, low_idx)
        new_point = seg_point[1][0] + low_idx[0]
        new_seg_points.append([seg_point[0], [seg_point[1][0], new_point]])
        for to_add in low_idx[1:]:
            new_point = seg_point[1][0] + to_add
            new_seg_points.append([new_seg_points[-1][0] + 1, [new_seg_points[-1][1][1], new_point]])
        new_seg_points.append([new_seg_points[-1][0] + 1, [new_seg_points[-1][1][1], seg_point[1][1]]])
    return new_seg_points

def seg_points_add_pause(seg_points, with_pause, num_split):
    seg_points_pauses = [i for i in with_pause if i[0] == 99]
    # print(seg_points_pauses)
    # print(seg_points)
    new_seg_points = []
    new_seg_points.append(seg_points_pauses[0])
    pause_idx = 1
    for i, seg_point in enumerate(seg_points):
        new_seg_points.append(seg_point)
        if seg_point[0] == num_split and pause_idx != len(seg_points_pauses):
            # print(pause_idx)
            new_seg_points.append(seg_points_pauses[pause_idx])
            pause_idx += 1
    return new_seg_points

def connect_short(seg_points, split_min_distance = 30):
    dist = [i[1][1] - i[1][0] for i in seg_points]
    new_seg_points = [seg_points[0]]
    i = 1
    while i < len(seg_points[:-1]):
    # for i in range(1, len(seg_points[:-1])):
        if dist[i] < split_min_distance:
            new_seg_points[i-1][1][1] = seg_points[i+1][1][1]
            i += 1
        else:
            new_seg_points.append(seg_points[i])
        i += 1
    new_seg_points += [seg_points[-1]]
    return new_seg_points

def generate_no_pause(traj, seg_points):
    idxs = []
    for seg_point in seg_points:
        if seg_point[1][0] != seg_point[1][1]:
            idxs.append(array([i for i in range(seg_point[1][0], seg_point[1][1])]))
        
    traj_no_pause_seg = []
    traj_no_pause = []
    for i, j in enumerate(idxs):
        traj_no_pause_seg.append([seg_points[i][0], traj[j]])
        traj_no_pause.append(traj[j])
    traj_no_pause = [j for i in traj_no_pause for j in i]
    return array(traj_no_pause), traj_no_pause_seg

def split_low_vel(traj, num_split, min_length):
    traj_id = traj[0]
    seg = traj[1]
    sum_abs_vel = np.sum(np.abs(np.diff(seg, axis = 0)), axis = 1).reshape(-1, 1)
    indexed_x = np.append(np.array(range(sum_abs_vel.shape[0])).reshape(-1, 1), sum_abs_vel, axis = 1)
    sorted_x = indexed_x[np.argsort(indexed_x[:, 1])]
    # print(sorted_x)
    # print((min_length < sorted_x[:, 0]) * (sorted_x[:, 0] < seg.shape[0] - min_length))
    # print()
    filtered_x = sorted_x[(min_length < sorted_x[:, 0]) * (sorted_x[:, 0] < seg.shape[0] - min_length)]
    # print(filtered_x)
    # input()
    sorted_idx_x = filtered_x[:, 0].astype(int)
    # print(sorted_x)
    low_idx = [sorted_idx_x[0]]
    idx = 1
    while len(low_idx) < num_split:
        if np.abs(low_idx[-1] - sorted_idx_x[idx]) > min_length:
            low_idx.append(sorted_idx_x[idx])
        idx += 1
    low_idx = sorted(low_idx)
    low_idx = [0] + low_idx + [len(seg)]
    # print(low_idx)
    segs = []
    for i in range(len(low_idx) - 1):
        segs.append([traj_id, seg[low_idx[i]:low_idx[i + 1] + 1], [low_idx[i], low_idx[i + 1] + 1]])
        traj_id += 1
    return segs

def plot_segments(segs):
    fig, ax = bagpy.create_fig(7)
    
    start_idx_with_pause = 0
    start_idx_no_pause = 0
    for seg in segs:
        if seg[0] == 0: col = 'b'
        elif seg[0] == 1: col = 'g'
        elif seg[0] == 2: col = 'm'
        elif seg[0] == 3: col = 'c'
        else: col = 'r'
        size = 10
        
        traj = seg[1]
        dy = np.diff(traj, axis = 0)
        dy = np.append(dy[0].reshape(1, -1), dy, axis = 0)
        sum_abs_dy = np.sum(np.abs(dy), axis = 1)
        
        ax[0].scatter(range(start_idx_with_pause , start_idx_with_pause + traj.shape[0]), traj[:, 0], c = col, s = size)
        ax[1].scatter(range(start_idx_with_pause , start_idx_with_pause + traj.shape[0]), traj[:, 1], c = col, s = size)
        ax[2].scatter(range(start_idx_with_pause , start_idx_with_pause + traj.shape[0]), traj[:, 2], c = col, s = size)
        
        ax[3].scatter(range(start_idx_with_pause , start_idx_with_pause + traj.shape[0]), sum_abs_dy, c = col, s = size)
        
        start_idx_with_pause += traj.shape[0]

        if seg[0] != 99:
            ax[4].scatter(range(start_idx_no_pause , start_idx_no_pause + traj.shape[0]), traj[:, 0], c = col, s = size)
            ax[5].scatter(range(start_idx_no_pause , start_idx_no_pause + traj.shape[0]), traj[:, 1], c = col, s = size)
            ax[6].scatter(range(start_idx_no_pause , start_idx_no_pause + traj.shape[0]), traj[:, 2], c = col, s = size)
            start_idx_no_pause += traj.shape[0]
    plt.show()
       
def shuffle_segment_points(seg_points):
    pass

def plot_traj(traj):
    size = 10
    fig, ax = bagpy.create_fig(3)
    ax[0].scatter(range(traj.shape[0]), traj[:, 0], s = size)
    ax[1].scatter(range(traj.shape[0]), traj[:, 1], s = size)
    ax[2].scatter(range(traj.shape[0]), traj[:, 2], s = size)
    plt.show()
    
def split(bag_path, min_length = 45, upper_limit = 5e-4, plot = True, num_split = 1, shuffle_segments = False, image_path = None):
    if image_path != None:
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path)
        
    df = bag_to_df(bag_path)
    df = df_add_velocity(df)
    df = trim_ends(df, upper_limit = 4e-4)
    traj = df_to_traj(df)
    # plot_traj(traj)
    seg_points = detect_split(df, 'pose.velocity.abs_total', min_length = min_length, upper_limit = upper_limit)
    # print(seg_points)
    # seg_points = connect_short(seg_points, split_min_distance)
    # print(seg_points)
    # seg_points = decrease_seg_points(seg_points)
    seg_points_no_pauses = [i for i in seg_points if i[0] == 0]
    seg_points_pauses = [seg_points[i] for i in range(len(seg_points)) if i != 0 and i != (len(seg_points) - 1) and seg_points[i][0] == 99]
    # print(seg_points_no_pauses, '\n', seg_points_pauses)
    # input()
    
    if shuffle_segments: 
        shuffled_idx = range(len(seg_points_no_pauses))
        shuffle(shuffled_idx)
        shuffled_idx = np.array(shuffled_idx)
    
    traj_no_pause, seg_traj_no_pause = generate_no_pause(traj, seg_points_no_pauses)    
    traj_pause, seg_traj_pause = generate_no_pause(traj, seg_points_pauses)
    
    if num_split > 0:
        split_seg_traj_no_pause = [seg_traj_no_pause[0]]
        for seg in seg_traj_no_pause[1:-1]:
            # print(seg)
            split_seg_traj_no_pause += split_low_vel(seg, num_split, min_length // (num_split))
        split_seg_traj_no_pause += [seg_traj_no_pause[-1]]
        seg_traj_no_pause = split_seg_traj_no_pause
    
    
    combined_seg_traj = []
    idx_pause = 0
    for i, j in enumerate(seg_traj_no_pause):
        if i != 0 and j[0] == 0:
            combined_seg_traj.append(seg_traj_pause[idx_pause])
            idx_pause += 1
        combined_seg_traj.append(j)
        
    if plot: plot_segments(combined_seg_traj)
    shutil.rmtree(bag_path[:-4])
    
    if image_path != None:        
        return [[traj_no_pause, seg_traj_no_pause], [traj_pause, seg_traj_pause], combined_seg_traj, [image_name, image]]
    else:
        return [[traj_no_pause, seg_traj_no_pause], [traj_pause, seg_traj_pause], combined_seg_traj]
    
        
num_split = 0
default_bag_upper_limit = 6e-4
default_bag_min_length = 80

bag_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/recordings/cutting/bag/'
param_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/recordings/cutting/splitter_param/'
data_name = '60_3_4_5'
# data_name = '5_similar'  

IMAGE_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/recordings/cutting/images/'
IMAGE_DIR += data_name

image_names = sorted(listdir(IMAGE_DIR))

bag_dir = join(bag_dir, data_name)
param_dir = join(param_dir, data_name, '{}_split'.format(num_split))
# bag_path = join(bag_dir)
bag_dirs = [i for i in listdir(bag_dir) if isdir(join(bag_dir, i))]
for i in bag_dirs: shutil.rmtree(join(bag_dir, i))
bags = sorted([i for i in listdir(bag_dir) if i[-3:] == 'bag'])

# bag_idx = 2
# segments, traj = split(join(bag_dir, bags[bag_idx]), 60, 6e-4)

if not isdir(param_dir):
    begin_idx = 0
    makedirs(param_dir)
    params = []
    bags_to_process = bags
else:
    processed_bag = sorted([i for i in listdir(param_dir) if i[-4:] == 'json'])
    if len(processed_bag) > 0:
        begin_idx = 0
        params = []
        for f in processed_bag:
            loaded = json.load(open(join(param_dir, f), 'r'))
            params.append([f[:-5], loaded['bag_min_length'], loaded['bag_upper_limit']])
        processed_bag_names = [i[:-5] for i in processed_bag]
        bags_to_process = [i for i in bags if i not in processed_bag_names]
    else:
        begin_idx = 0
        params = []
        bags_to_process = bags
#% Human-supervised
for i in range(len(bags_to_process)):
    try:
        print("\n# {}/{} | Processing '{}'".format(i+1, len(bags_to_process), bags_to_process[i]))
        bag_upper_limit = default_bag_upper_limit
        bag_min_length = default_bag_min_length
        bag_path = join(bag_dir, bags_to_process[i])
        segment, _, _ = split(bag_path, bag_min_length, bag_upper_limit, num_split = num_split)
        j = '.'
        while j != '':
            j = input('    Num segments = {}\n    Options:\n    (enter) Proceed\n    (1)     Change upper limit (Current value = {})\n    (2)     Change minimum length (Current value = {})\n    Input =  '.format(len(segment[1]), bag_upper_limit, bag_min_length))
            if j == '1':
                while 1:
                    try:
                        bag_upper_limit = float(input('        Enter new upper limit (Current value = {}): '.format(bag_upper_limit)))
                        break
                    except ValueError:
                        print('        Wrong input, try again')
                segment, _, _  = split(bag_path, bag_min_length, bag_upper_limit, num_split = num_split)
            elif j == '2':
                while 1:
                    try:
                        bag_min_length = int(input('        Enter new minimum length (Current value = {}): '.format(bag_min_length)))
                        break
                    except ValueError:
                        print('        Wrong input, try again')
                segment, _, _  = split(bag_path, bag_min_length, bag_upper_limit, num_split = num_split)
        params.append([bags_to_process[i], 
                       bag_min_length, 
                       bag_upper_limit])
        to_json = {'bag_min_length': bag_min_length, 
                   'bag_upper_limit': bag_upper_limit}
        json.dump(to_json, open(join(param_dir, '{}.json'.format(bags_to_process[i])), "w"))
    except KeyboardInterrupt:
        break
             
#%%
    
segments = []
for i in range(len(bags)):
    print('({}/{}) Generating segment points from parameters'.format(i+1, len(bags)))
    bag_path = join(bag_dir, params[i][0])
    # split_min_distance = 35
    while 1:
        try:
            [traj_no_pause, seg_traj_no_pause], [traj_pause, seg_traj_pause], combined_seg_traj, [image_name, image] = split(bag_path, params[i][1], params[i][2], plot = 0, num_split = num_split, image_path = join(IMAGE_DIR, image_names[i]))
            
            # seg_no_pause, seg_no_pause_seg = generate_no_pause(traj, seg_points_no_pause)
            segments.append([params[i], 
                             combined_seg_traj,
                             traj_no_pause,
                             [i[1] for i in seg_traj_no_pause],
                             [image_name, image]])
            break
        except IndexError:
            # split_min_distance = int(input('Enter shorter min distance (current = {}): '.format(split_min_distance)))
            # split_min_distance -= 1
            pass
    # input()
            
max_segments = np.array([len(i[3]) for i in segments]).max()

# for i in range(len(segments)):
#     temp_points = []
#     for j in range(len(segments[i][1][0])):
#         if segments[i][1][0][j][1][1] - segments[i][1][0][j][1][0] > 10:
#             temp_points.append(segments[i][1][0][j])
#     segments[i][1][0] = temp_points
#%%
BF = 30
DATA = {'image_name': [],
        'bag_name': [],
        'image': [],
        'image_dim': (1, 100, 100),
        'original_trajectory': [],
        'normal_dmp_seg_num': np.ones(len(bags)).reshape(-1, 1),
        'normal_dmp_y0': [],
        'normal_dmp_goal': [],
        'normal_dmp_w': [],
        'normal_dmp_tau': [],
        'normal_dmp_dt': 0.0001,
        'normal_dmp_bf': max_segments * BF,
        'normal_dmp_ay': 25,
        'segmented_dmp_max_seg_num': max_segments,
        'segmented_dmp_seg_num': [],
        'segmented_dmp_y0': [],
        'segmented_dmp_goal': [],
        'segmented_dmp_w': [],
        'segmented_dmp_tau': [],
        'segmented_dmp_dt': 0.01,
        'segmented_dmp_bf': BF,
        'segmented_dmp_ay': 15,
        'segmented_dmp_trajectory': []
       }

for i, segment in enumerate(segments):
    bag_name = segment[0][0]
    print("({}/{})".format(i + 1, len(segments)), "Processing", bag_name)
    
    # image = Image.open(join(IMAGE_DIR, image_name)).resize(DATA['image_dim'][1:])
    image_name = segment[4][0]
    image = segment[4][1].resize(DATA['image_dim'][1:])
    if DATA['image_dim'][0] == 1: image = ImageOps.grayscale(image)
    image = array(image).reshape(DATA['image_dim'])
    traj = segment[2]
    
    fig, ax = bagpy.create_fig(3)
    
    ax[0].scatter(range(traj.shape[0]), traj[:, 0], c = 'g')
    ax[1].scatter(range(traj.shape[0]), traj[:, 1], c = 'g')
    ax[2].scatter(range(traj.shape[0]), traj[:, 2], c = 'g')
    # start_idx = 0
    # for seg_traj in segment[2]:
    #     col = 'g'
    #     ax[0].scatter(range(start_idx, start_idx + seg_traj.shape[0]), seg_traj[:, 0], c = col)
    #     ax[1].scatter(range(start_idx, start_idx + seg_traj.shape[0]), seg_traj[:, 1], c = col)
    #     ax[2].scatter(range(start_idx, start_idx + seg_traj.shape[0]), seg_traj[:, 2], c = col)
    #     start_idx += seg_traj.shape[0]
    
    normal_dmp = DMPs_discrete(n_dmps   = traj.shape[1], 
                               n_bfs    = DATA['normal_dmp_bf'], 
                               dt       = DATA['normal_dmp_dt'],
                               ay       = np.ones(traj.shape[1]) * DATA['normal_dmp_ay'])
    normal_dmp.imitate_path(traj.T)
    normal_dmp_tau =  (1 / DATA['normal_dmp_dt']) / traj.shape[0]
    # normal_dmp_tau =  1.
    normal_y, normal_dy, normal_ddy = normal_dmp.rollout(tau = normal_dmp_tau)
    
    ax[0].scatter(range(normal_y.shape[0]), normal_y[:, 0], c = 'r')
    ax[1].scatter(range(normal_y.shape[0]), normal_y[:, 1], c = 'r')
    ax[2].scatter(range(normal_y.shape[0]), normal_y[:, 2], c = 'r')
    
    segment_dmp_y0s = []
    segment_dmp_goals = []
    segment_dmp_ws = []
    segment_dmp_taus = []
    segment_ys = []
    for seg in segment[3]:
        segment_dmp = DMPs_discrete(n_dmps   = traj.shape[1], 
                                    n_bfs    = DATA['segmented_dmp_bf'], 
                                    dt       = DATA['segmented_dmp_dt'],
                                    ay       = np.ones(traj.shape[1]) * DATA['segmented_dmp_ay'])
        segment_dmp.imitate_path(seg.T)
        tau =  (1 / DATA['segmented_dmp_dt']) / seg.shape[0]
        # tau =  1.
        segment_y, segment_dy, segment_ddy = segment_dmp.rollout(tau = tau)
        segment_ys.append(segment_y)
        
        segment_dmp_y0s.append(segment_dmp.y0)
        segment_dmp_goals.append(segment_dmp.goal)
        segment_dmp_ws.append(segment_dmp.w)
        segment_dmp_taus.append(tau)
            
    start_idx = 0
    for seg in segment_ys:
        ax[0].scatter(range(start_idx, start_idx + seg.shape[0]), seg[:, 0])
        ax[1].scatter(range(start_idx, start_idx + seg.shape[0]), seg[:, 1])
        ax[2].scatter(range(start_idx, start_idx + seg.shape[0]), seg[:, 2])
        start_idx = start_idx + seg.shape[0]
    plt.show()
    # input()
    
    segmented_dmp_traj = segment_ys[0]
    for j in range(1, len(segment_ys)):
        segmented_dmp_traj = np.append(segmented_dmp_traj, segment_ys[j], axis = 0)
    
    DATA['image_name'].append(image_name)
    DATA['bag_name'].append(bag_name)
    DATA['image'].append(image)
    DATA['original_trajectory'].append(traj)
    DATA['segmented_dmp_trajectory'].append(segmented_dmp_traj)
    DATA['normal_dmp_y0'].append(normal_dmp.y0)
    DATA['normal_dmp_goal'].append(normal_dmp.goal)
    DATA['normal_dmp_w'].append(normal_dmp.w)
    DATA['normal_dmp_tau'].append(normal_dmp_tau)
    DATA['segmented_dmp_seg_num'].append(len(segment[3]))
    DATA['segmented_dmp_y0'].append(segment_dmp_y0s)
    DATA['segmented_dmp_goal'].append(segment_dmp_goals)
    DATA['segmented_dmp_w'].append(segment_dmp_ws)
    DATA['segmented_dmp_tau'].append(segment_dmp_taus)
#%% Pad segmented with average
to_process = deepcopy(DATA)

unique_lengths = []
for i in to_process['segmented_dmp_w']:
    if len(i) not in unique_lengths:
        unique_lengths.append(len(i))
unique_lengths = sorted(unique_lengths)
unique_lengths = [i for i in range(1, unique_lengths[0])] + unique_lengths

all_segments = {'y0': [], 'goal': [], 'w': [], 'tau' : []}
cut_segments = {'y0': [], 'goal': [], 'w': [], 'tau' : []}
end_segments = {'y0': [], 'goal': [], 'w': [], 'tau' : []}
idx_segments = {'y0': [[] for i in range(unique_lengths[-1])],
                'goal': [[] for i in range(unique_lengths[-1])],
                'w': [[] for i in range(unique_lengths[-1])],
                'tau': [[] for i in range(unique_lengths[-1])]}

for i in range(len(to_process['segmented_dmp_y0'])):
    for seg in range(len(to_process['segmented_dmp_y0'][i])):        
        if seg != 0 and seg != len(to_process['segmented_dmp_y0'][i]) - 1:
            cut_segments['y0'].append(to_process['segmented_dmp_y0'][i][seg])
            cut_segments['goal'].append(to_process['segmented_dmp_goal'][i][seg])
            cut_segments['w'].append(to_process['segmented_dmp_w'][i][seg])
            cut_segments['tau'].append(to_process['segmented_dmp_tau'][i][seg])
        
        all_segments['y0'].append(to_process['segmented_dmp_y0'][i][seg])
        all_segments['goal'].append(to_process['segmented_dmp_goal'][i][seg])
        all_segments['w'].append(to_process['segmented_dmp_w'][i][seg])
        all_segments['tau'].append(to_process['segmented_dmp_tau'][i][seg])
        
        idx_segments['y0'][seg].append(to_process['segmented_dmp_y0'][i][seg])
        idx_segments['goal'][seg].append(to_process['segmented_dmp_goal'][i][seg])
        idx_segments['w'][seg].append(to_process['segmented_dmp_w'][i][seg])
        idx_segments['tau'][seg].append(to_process['segmented_dmp_tau'][i][seg])
        
    end_segments['y0'].append(to_process['segmented_dmp_y0'][i][-1])
    end_segments['goal'].append(to_process['segmented_dmp_goal'][i][-1])
    end_segments['w'].append(to_process['segmented_dmp_w'][i][-1])
    end_segments['tau'].append(to_process['segmented_dmp_tau'][i][-1])
    
all_segments['y0'] = array(all_segments['y0'])
all_segments['goal'] = array(all_segments['goal'])
all_segments['w'] = array(all_segments['w'])
all_segments['tau'] = array(all_segments['tau'])
cut_segments['y0'] = array(cut_segments['y0'])
cut_segments['goal'] = array(cut_segments['goal'])
cut_segments['w'] = array(cut_segments['w'])
cut_segments['tau'] = array(cut_segments['tau'])
end_segments['y0'] = array(end_segments['y0'])
end_segments['goal'] = array(end_segments['goal'])
end_segments['w'] = array(end_segments['w'])
end_segments['tau'] = array(end_segments['tau'])
idx_segments['y0'] = [array(i) for i in idx_segments['y0']]
idx_segments['goal'] = [array(i) for i in idx_segments['goal']]
idx_segments['w'] = [array(i) for i in idx_segments['w']]
idx_segments['tau'] = [array(i) for i in idx_segments['tau']]


pads = idx_segments

for i in range(len(to_process['segmented_dmp_y0'])):
    if len(to_process['segmented_dmp_y0'][i]) < unique_lengths[-1]:
        while len(to_process['segmented_dmp_y0'][i]) < unique_lengths[-1]:
            to_process['segmented_dmp_y0'][i].append(pads['y0'][len(to_process['segmented_dmp_y0'][i])].mean(axis = 0))
            to_process['segmented_dmp_goal'][i].append(pads['goal'][len(to_process['segmented_dmp_goal'][i])].mean(axis = 0))
            to_process['segmented_dmp_w'][i].append(pads['w'][len(to_process['segmented_dmp_w'][i])].mean(axis = 0))
            to_process['segmented_dmp_tau'][i].append(pads['tau'][len(to_process['segmented_dmp_tau'][i])].mean(axis = 0))
            
#%%
DATA = to_process

DATA['image']                   = array(DATA['image'])
DATA['normal_dmp_goal']         = array(DATA['normal_dmp_goal']).reshape(len(bags), 1, traj.shape[1])
DATA['normal_dmp_w']            = array(DATA['normal_dmp_w']).reshape(len(bags), 1, traj.shape[1], DATA['normal_dmp_bf'])
DATA['normal_dmp_y0']           = array(DATA['normal_dmp_y0']).reshape(len(bags), 1, traj.shape[1])
DATA['normal_dmp_tau']          = array(DATA['normal_dmp_tau']).reshape(-1, 1)
DATA['segmented_dmp_seg_num']   = array(DATA['segmented_dmp_seg_num']).reshape(-1, 1)
DATA['segmented_dmp_goal']      = array(DATA['segmented_dmp_goal'])
DATA['segmented_dmp_tau']       = array(DATA['segmented_dmp_tau'])
DATA['segmented_dmp_w']         = array(DATA['segmented_dmp_w'])
DATA['segmented_dmp_y0']        = array(DATA['segmented_dmp_y0'])
# DATA['segmented_dmp_y0']        = DATA['segmented_dmp_y0'][:, 0]

# print('Weight variance = {}'. format(DATA['segmented_dmp_w'].var(axis = 0)))
#%%
dist_w = np.abs((DATA['segmented_dmp_w'] - DATA['segmented_dmp_w'].mean(axis = 0)).mean(axis = (1, 2, 3)))
dist_w_sorted = np.sort(dist_w)
dist_w_filtered = dist_w < 10

print('Filtered = {}'.format(len([i for i in dist_w_filtered if i])))
# input()
#%%
DATA_FILTERED = deepcopy(DATA)

DATA_FILTERED['bag_name'] = [j for i,j in enumerate(DATA['bag_name']) if dist_w_filtered[i]]
DATA_FILTERED['image_name'] = [j for i,j in enumerate(DATA['image_name']) if dist_w_filtered[i]]
DATA_FILTERED['original_trajectory'] = [j for i,j in enumerate(DATA['original_trajectory']) if dist_w_filtered[i]]
DATA_FILTERED['image'] = DATA['image'][dist_w_filtered]

DATA_FILTERED['normal_dmp_goal'] = DATA['normal_dmp_goal'][dist_w_filtered]
DATA_FILTERED['normal_dmp_tau'] = DATA['normal_dmp_tau'][dist_w_filtered]
DATA_FILTERED['normal_dmp_w'] = DATA['normal_dmp_w'][dist_w_filtered]
DATA_FILTERED['normal_dmp_y0'] = DATA['normal_dmp_y0'][dist_w_filtered]

DATA_FILTERED['segmented_dmp_seg_num'] = DATA['segmented_dmp_seg_num'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_goal'] = DATA['segmented_dmp_goal'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_tau'] = DATA['segmented_dmp_tau'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_w'] = DATA['segmented_dmp_w'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_y0'] = DATA['segmented_dmp_y0'][dist_w_filtered]

num_seg = DATA_FILTERED['segmented_dmp_seg_num'].reshape(-1).tolist()
y0s = DATA_FILTERED['segmented_dmp_y0'].tolist()
goals = DATA_FILTERED['segmented_dmp_goal'].tolist()

for i in range(len(y0s)):
    if y0s[i][-1] == [0.0, 0.0, 0.0]:
        idx = 0
        while y0s[i][idx] != [0.0, 0.0, 0.0]:
            idx += 1
        y0s[i] = y0s[i][:idx]
        goals[i] = goals[i][:idx]

num_segments = [[num_seg[i]] * len(y0s[i]) for i in range(len(y0s))]
num_segments_unraveled = [j for i in num_segments for j in i]
y0s_unraveled = [j for i in y0s for j in i]
goals_unraveled = [j for i in goals for j in i]

DATA_FILTERED['seg_num'] = np.array(num_segments_unraveled).reshape(-1, 1)
DATA_FILTERED['pos_y0'] = np.array(y0s_unraveled)
DATA_FILTERED['pos_goal'] = np.array(goals_unraveled)

DATA_FILTERED['dmp_y0'] = DATA_FILTERED['segmented_dmp_y0'][:, 0, :]
#%%
# to_pkl = DATA_FILTERED
to_pkl = DATA

ROOT_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
PKL_DIR = 'data/pkl/cutting_traj'
PKL_NAME = 'real_distanced_trajectory'
PKL_NAME += '.num_data_' + str(len(to_pkl['image_name'])) + '_num_seg_' + str(to_pkl['segmented_dmp_max_seg_num'])
PKL_NAME += '.normal_dmp_bf_' + str(to_pkl['normal_dmp_bf']) + '_ay_' + str(to_pkl['normal_dmp_ay']) + '_dt_' + str(to_pkl['normal_dmp_dt'])
PKL_NAME += '.seg_dmp_bf_' + str(to_pkl['segmented_dmp_bf']) + '_ay_' + str(to_pkl['segmented_dmp_ay']) + '_dt_' + str(to_pkl['segmented_dmp_dt'])
PKL_NAME += '.' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
PKL_NAME += '.pkl'
pkl.dump(to_pkl, open(join(ROOT_DIR, PKL_DIR, PKL_NAME), 'wb'))
print('Saved as {}'.format(PKL_NAME))

#%%