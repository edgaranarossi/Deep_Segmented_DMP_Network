#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:40:41 2022

@author: edgar
"""
import numpy as np
from numpy import array, round, abs, floor, ceil, sin, cos, tan
from numpy.random import randint
import pandas as pd
from bagpy import bagreader, create_fig
from PIL import Image, ImageOps
from os.path import join, isdir
from os import listdir, makedirs
import shutil
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from random import shuffle
from pydmps import DMPs_discrete, DMPs_rhythmic
from datetime import datetime
import pickle as pkl

def bag_to_df(bag_path):
    b = bagreader(bag_path, verbose = False)
    bag_csv = b.message_by_topic('/mocap_pose_topic/knife_marker_pose')
    df = pd.read_csv(bag_csv)
    bag_dir = bag_path[:-4]
    shutil.rmtree(bag_dir)
    return df

def import_image_traj(image_path, bag_path):
    traj_df = bag_to_df(bag_path)
    image = Image.open(image_path)
    x = array(traj_df['pose.position.x']).reshape(-1, 1)
    y = array(traj_df['pose.position.y']).reshape(-1, 1)
    z = array(traj_df['pose.position.z']).reshape(-1, 1)
    dx = np.diff(x, axis = 0)
    dy = np.diff(y, axis = 0)
    dz = np.diff(z, axis = 0)
    dx = np.append(dx[0].reshape(1, -1), dx, axis = 0)
    dy = np.append(dy[0].reshape(1, -1), dy, axis = 0)
    dz = np.append(dz[0].reshape(1, -1), dz, axis = 0)
    sum_abs_total_vel = np.sum(np.abs(dx) + np.abs(dy) + np.abs(dz), axis = 1)
    traj = np.append(x, y, axis = 1)
    traj = np.append(traj, z, axis = 1)
    data = {'image_path'        : image_path,
            'image_name'        : image_path.split('/')[-1],
            'bag_path'          : bag_path,
            'bag_name'          : bag_path.split('/')[-1],
            'image'             : image,
            'traj_df'           : traj_df,
            'pos_x'            : x,
            'pos_y'            : y,
            'pos_z'            : z,
            'traj'              : traj,
            'vel_x'                : x,
            'vel_y'                : y,
            'vel_z'                : z,
            'sum_abs_total_vel' : sum_abs_total_vel}
    return data

def plot_segments(data, trim_ends = True, title = None):
    # df = data['traj_df']
    # x = array(df['pose.position.x']).reshape(-1, 1)
    # y = array(df['pose.position.y']).reshape(-1, 1)
    # z = array(df['pose.position.z']).reshape(-1, 1)
    # traj = np.concatenate([x, y, z], axis = 1)
    seg_points = data['seg_points']
    
    fig, ax = create_fig(7)
    if title != None: ax[0].set_title(str(title))
    cur_idx = 0
    cur_idx_with_pause = 0
    for seg_point in seg_points:
        for split in seg_point:
            if split[0] == 0:
                col = 'b'
            elif split[0] == 1:
                col = 'c'
            elif split[0] == 2:
                col = 'm'
            elif split[0] == 3:
                col = 'g'
            elif split[0] == 9999:
                col = 'r'
            size = 2
            
            d_traj = np.diff(split[2], axis = 0)
            d_traj = np.append(d_traj[0].reshape(1, -1), d_traj, axis = 0)
            sum_abs_d_traj = np.sum(np.abs(d_traj), axis = 1)
                
            if split[0] != 9999:
                ax[0].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), split[2][:, 0], c = col, s = size)
                ax[1].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), split[2][:, 1], c = col, s = size)
                ax[2].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), split[2][:, 2], c = col, s = size)
                ax[3].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), sum_abs_d_traj, c = col, s = size)
                ax[4].scatter(range(cur_idx, cur_idx + split[2].shape[0]), split[2][:, 0], c = col, s = size)
                ax[5].scatter(range(cur_idx, cur_idx + split[2].shape[0]), split[2][:, 1], c = col, s = size)
                ax[6].scatter(range(cur_idx, cur_idx + split[2].shape[0]), split[2][:, 2], c = col, s = size)
                cur_idx += split[2].shape[0]
                cur_idx_with_pause += split[2].shape[0]
            else:
                ax[0].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), split[2][:, 0], c = col, s = size)
                ax[1].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), split[2][:, 1], c = col, s = size)
                ax[2].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), split[2][:, 2], c = col, s = size)
                ax[3].scatter(range(cur_idx_with_pause, cur_idx_with_pause + split[2].shape[0]), sum_abs_d_traj, c = col, s = size)
                cur_idx_with_pause += split[2].shape[0]
    plt.show()
    
def split_low_vel(seg_point, num_split, min_length = 20):
    traj_id = seg_point[0]
    seg = seg_point[2]
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
        segs.append([traj_id, [low_idx[i], low_idx[i + 1] + 1], seg[low_idx[i]:low_idx[i + 1] + 1]])
        traj_id += 1
    return segs

def detect_low_vel(data, upper_limit = 6e-4, ignore_limit = 5, min_length = 20, num_split = 0):
    df = data['traj_df']
    x = array(df['pose.position.x']).reshape(-1, 1)
    y = array(df['pose.position.y']).reshape(-1, 1)
    z = array(df['pose.position.z']).reshape(-1, 1)
    
    d_x = np.diff(x, axis = 0)
    d_y = np.diff(y, axis = 0)
    d_z = np.diff(z, axis = 0)
    d_x = np.append(d_x[0].reshape(1, 1), d_x, axis = 0)
    d_y = np.append(d_y[0].reshape(1, 1), d_y, axis = 0)
    d_z = np.append(d_z[0].reshape(1, 1), d_z, axis = 0)
    
    df['pose.velocity.x'] = d_x
    df['pose.velocity.y'] = d_y
    df['pose.velocity.z'] = d_z
    
    df['pose.velocity.sum_abs_velocity'] = abs(d_x) + abs(d_y) + abs(d_z)
    
    below_vel_threshold = ''.join(np.where(array(df['pose.velocity.sum_abs_velocity']) < upper_limit, '1', '0').tolist())
    pauses = [i for i in below_vel_threshold.split('0') if len(i) > 0]
    not_pauses = [i for i in below_vel_threshold.split('1') if len(i) > 0]
    identifier = []
    current = bool(below_vel_threshold[0])
    idx_pause = 0
    idx_not_pause = 0
    # print(len(''.join(pauses)) + len(''.join(not_pauses)))
    while len(identifier) < len(pauses + not_pauses):
        if len(pauses + not_pauses) - len(identifier) == 1:
            if not current:
                identifier.append(not_pauses[-1])
                break
        if current:
            identifier.append(pauses[idx_pause])
            idx_pause += 1
        else:
            identifier.append(not_pauses[idx_not_pause])
            idx_not_pause += 1
        current = not current
    # print(len(''.join(identifier)), ''.join(identifier))
    
    for i in range(1, ignore_limit + 1):
        j = 1
        while j < len(identifier) - 1:
            if j + 1 > len(identifier): break
            if len(identifier[j]) == i:
                prev_val = bool(int(identifier[j - 1][0]))
                cur_val = bool(int(identifier[j][0]))
                next_val = bool(int(identifier[j + 1][0]))
                if prev_val != cur_val != next_val:
                    identifier[j - 1] = identifier[j - 1] + (str(int(not cur_val)) * len(identifier[j])) + identifier[j + 1]
                    identifier[j:] = identifier[j + 2:]
            else:
                j += 1
        
    identifier = ''.join(identifier)
    data['identifier'] = identifier
    identifier = [bool(int(identifier[i])) for i in range(len(identifier))]
    
    seg_points = []
    begin = 0
    prev = identifier[0]
    for i in range(1, len(identifier)):
        if identifier[i] != prev:
            name = int(prev) if int(prev) == 0 else 9999
            seg_points.append([[name, [begin, i], data['traj'][begin:i]]])
            begin = i
            prev = identifier[i]
    seg_points.append([[9999, [begin, len(identifier)], data['traj'][begin:len(identifier)]]])
    
    # print(len(identifier))
    
    if num_split > 0:
        split_seg_points = []
        for i, seg_point in enumerate(seg_points):
            if seg_point[0][0] == 9999 or i in [1, len(seg_points) - 2]:
                split_seg_points.append(seg_point)
            else:
                # splits = split_low_vel(seg_point, num_split = 2, min_length = ignore_limit // 2)
                # for split in splits:
                split_min_div = 1.5
                while 1:
                    try:
                        split_seg_points.append(split_low_vel(seg_point[0], num_split = num_split, min_length = ignore_limit // split_min_div))
                        # split_seg_points.append(split_low_vel(seg_point[0], num_split = 2, min_length = min_length))
                        # print('succeed')
                        break
                    except IndexError:
                        # split_min_div = float(input('Enter new split_min_div = '))
                        split_min_div += 0.1
        seg_points = split_seg_points
            
    df['is_low_vel'] = identifier
    data['traj_df'] = df
    data['seg_points'] = seg_points
    
    return data

def grayscale_image(data):
    if 'processed_image' in data:
        img = data['processed_image']
    else:
        img = data['image']
    data['processed_image'] = ImageOps.grayscale(img)
    return data

def resize_image(data, dim = (100, 100)):
    if 'processed_image' in data:
        img = data['processed_image']
    else:
        img = data['image']
    data['processed_image'] = img.resize(dim)
    return data

def image_to_ndarray(data):
    if 'processed_image' in data:
        img = data['processed_image']
    else:
        img = data['image']
    data['np_image'] = array(img)
    if data['np_image'].max() > 1: data['np_image'] = data['np_image'] / 255.
    return data
    
def crop_image(data, upper_limit = 45, bottom_limit = 82, left_limit = 25, right_limit = 76):
    data['np_image'] = data['np_image'][upper_limit:bottom_limit, left_limit:right_limit]
    return data    

class ImageTrajPair:
    def __init__(self, image, traj, traj_type = None):
        self.image = image
        self.traj = traj
        self.dof = self.traj[0][2].shape[1]
        self.traj_type = traj_type
        
        # print(self.traj)
        
        # if self.traj_type != 'leftover' and self.traj[0][2][0, 2] != 0.0:
            # zs = array([i[2][:, 2].min() for i in self.traj])
            # z_min = zs.min()
            
        # x_min = self.traj[0][2][0, 0] if self.traj_type != 'start' else self.traj[0][2][-1, 0]
        # # x_min = self.traj[0][2][0, 0]
        # y_min = self.traj[0][2][0, 1]
        # z_min = self.traj[0][2][0, 2]
        # for i in range(len(self.traj)):
            # self.traj[i][2][:, 0] -= x_min
            # self.traj[i][2][:, 1] -= y_min
            # self.traj[i][2][:, 2] -= z_min
        
            
def split_image(data, z_end_pos = -0.37202444672584534):
    img = data['np_image']
    # plt.imshow(img)
    # plt.show()
    
    seg_points = deepcopy(data['seg_points'])
    start_seg_points = [i for i in seg_points[:2] if i[0][0] != 9999]
    mid_seg_points = [i for i in seg_points[2:-3] if i[0][0] != 9999]
    end_seg_points = [i for i in seg_points[-3:] if i[0][0] != 9999]
    
    z_seg_points = [i[0][2][0,2] for i in mid_seg_points] + [end_seg_points[0][0][2][0,2]] + [z_end_pos]
    z_seg_points = array(z_seg_points[::-1])
    segment_ratio = (z_seg_points - z_seg_points.min()) / ((z_seg_points - z_seg_points.min())).max()
    width = img.shape[1]
    img_pixel_coords = np.round(segment_ratio * width).astype(int)
    
    # for seg in range(len()):
    img_segments = []
    img_segments.append(ImageTrajPair(img[:, img_pixel_coords[0]:img_pixel_coords[1]], end_seg_points[0], traj_type = 'leftover'))
    # print(img_pixel_coords)
    # print(len(mid_seg_points))
    for x in range(1, len(img_pixel_coords) - 1):
        if x == len(img_pixel_coords) - 2:
            t_type = 'start_cut'
        elif x == 1:
            t_type = 'end_cut'
        else:
            t_type = None
        img_segments.append(ImageTrajPair(img[:, img_pixel_coords[x]:img_pixel_coords[x + 1]], mid_seg_points[::-1][x - 1], traj_type = t_type))
    img_segments.append(ImageTrajPair(img[:, img_pixel_coords[-2]:img_pixel_coords[-1]], start_seg_points[-1], traj_type = 'begin'))
        
    return img_segments

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
#%%
runcell(0, '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/scripts/utils/data_generator/image_traj_shuffler.py')
if __name__=='__main__':
    ROOT_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
    DATA_DIR = join(ROOT_DIR, 'data', 'recordings', 'cutting')
    DATA_NAME = '60_3_4_5'
    BAG_DIR = join(DATA_DIR, 'bag', DATA_NAME)
    IMAGE_DIR = join(DATA_DIR, 'images', DATA_NAME)
    NUM_SPLIT = 0
    PARAM_DIR = join(DATA_DIR, 'splitter_param', DATA_NAME, '{}_split_shuffle'.format(NUM_SPLIT))
    
    bags = sorted(listdir(BAG_DIR))
    images = sorted(listdir(IMAGE_DIR)) 
    bag_paths = [join(BAG_DIR, i) for i in bags if not isdir(join(BAG_DIR, i))]
    image_paths = [join(IMAGE_DIR, i) for i in images if not isdir(join(IMAGE_DIR, i))]
    
    if not isdir(PARAM_DIR): 
        makedirs(PARAM_DIR)
        begin_idx = 0
    else:
        processed = [i[:-5] for i in listdir(PARAM_DIR) if not isdir(join(PARAM_DIR, i)) and i[-4:] == 'json']

#%
    data = []
    params = []
    seg_begin = []
    seg_starts = []
    seg_mids = []
    seg_ends = []
    seg_leftovers = []
    max_segments = 0
    for i in range(len(bag_paths)):
        if bags[i][:-4] in processed:
            print('({}/{}) Loading {}'.format(i+1, len(bag_paths), bags[i]))
            param = json.load(open(join(PARAM_DIR, '{}.json'.format(bags[i][:-4])), 'r'))
            d = import_image_traj(image_paths[i], bag_paths[i])
            d = detect_low_vel(d, upper_limit = param['upper_limit'], ignore_limit = param['ignore_limit'], num_split = NUM_SPLIT)
        else:
            param = {'bag_name': bag_paths[i], 
                     'image_name': image_paths[i],
                     'upper_limit': 6e-4,
                     'ignore_limit': 80,
                     'min_length': 40}
            
            d = import_image_traj(image_paths[i], bag_paths[i])
            while 1:
                d = detect_low_vel(d, upper_limit = param['upper_limit'], ignore_limit = param['ignore_limit'], min_length = param['min_length'], num_split = NUM_SPLIT)
                plot_segments(d, title = i)
                j = input('({}/{}) Processing {}\n  (Enter) Proceed\n  (1)     Change velocity upper limit\n  (2)     Change segment ignore limit\n  (3)     Change segment min length\nInput = '.format(i+1, len(bag_paths), bags[i]))
                if j == '':
                    json.dump(param, open(join(PARAM_DIR, '{}.json'.format(bag_paths[i].split('/')[-1].split('.')[0])), "w"))
                    break
                elif j == '1':
                    param['upper_limit'] = float(input('    Enter new velocity upper limit (current = {}) = '.format(param['upper_limit'])))
                elif j == '2':
                    param['ignore_limit'] = int(input('    Enter new segment ignore limit (current = {}) = '.format(param['ignore_limit'])))
                elif j == '3':
                    param['min_length'] = int(input('    Enter new segment ignore limit (current = {}) = '.format(param['min_length'])))
        params.append(param)
                
        d = grayscale_image(d)
        d = resize_image(d)
        d = image_to_ndarray(d)
        d = crop_image(d)
        seg_img_traj = split_image(d)
        d['segments'] = seg_img_traj[::-1]
        if len(d['segments']) - 2 > max_segments: max_segments = len(d['segments']) - 2
        seg_begin += [i for i in seg_img_traj if i.traj_type == 'begin']
        seg_starts += [i for i in seg_img_traj if i.traj_type == 'start_cut']
        seg_mids += [i for i in seg_img_traj if i.traj_type == None]
        seg_ends += [i for i in seg_img_traj if i.traj_type == 'end_cut']
        seg_leftovers += [i for i in seg_img_traj if i.traj_type == 'leftover']
        
        data.append(d)
        # input('break')

#%
img_dim = (37, 51)
data_per_cut_type = int(1e3)
cuts = [3, 4, 5, 6]
padding = data[59]['np_image'][:, 5:11]

#%%
print('\nGenerating random segment combinations\n')
dataset = []
longest_img_width = 0
max_segments = 0
for cut in cuts:
    for i in range(data_per_cut_type):
        img_width = 0
        segs = []
        segs.append(seg_starts[randint(0, len(seg_starts))])
        for i in range(cut - 2):
            segs.append(seg_mids[randint(0, len(seg_mids))])
        segs.append(seg_ends[randint(0, len(seg_ends))])
        segs.append(seg_leftovers[randint(0, len(seg_leftovers))])
        for seg in segs:
            img_width += seg.image.shape[1]
        if longest_img_width < img_width: longest_img_width = img_width
        dataset.append(segs)
        trajs = [j for i in segs for j in i.traj]
        max_segments = len(trajs) if len(trajs) > max_segments else max_segments
        
padded_img_width = int((np.ceil(longest_img_width / 10)) * 10)

#%%
target_bf = 1000
base_bf = 50
plot_traj = 0
DT = 0.01
AY = 25
# plot_segments(d)
rot_degs = [-45., -45., -45.]
rot_order = ['x', 'y', 'z']

DATA = {'image': [],
        'image_dim': (1, 100, 100),
        'original_trajectory': [],
        'processed_trajectory': [],
        'rotated_trajectory': [],
        'rotation_order': rot_order,
        'rotation_degrees': rot_degs,
        'normal_dmp_seg_num': np.ones(len(data)).reshape(-1, 1),
        'normal_dmp_y0': [],
        'normal_dmp_goal': [],
        'normal_dmp_w': [],
        'normal_dmp_tau': [],
        'normal_dmp_dt': 0.001,
        'normal_dmp_bf': max_segments * base_bf,
        'normal_dmp_target_bf': target_bf,
        'normal_dmp_ay': 25,
        'normal_dmp_trajectory': [],
        'normal_dmp_target_trajectory': [],
        'normal_dmp_L_y0': [],
        'normal_dmp_L_goal': [],
        'normal_dmp_L_w': [],
        'normal_dmp_L_tau': [],
        'normal_dmp_L_bf': 1000,
        'normal_dmp_L_ay': 200,
        'normal_dmp_L_trajectory': [],
        'segmented_dmp_max_seg_num': max_segments,
        'segmented_dmp_seg_num': [],
        'segmented_dmp_y0': [],
        'segmented_dmp_goal': [],
        'segmented_dmp_w': [],
        'segmented_dmp_tau': [],
        'segmented_dmp_dt': 0.001 * max_segments,
        'segmented_dmp_bf': base_bf,
        'segmented_dmp_target_bf': target_bf,
        'segmented_dmp_ay': 15,
        'segmented_dmp_trajectory': [],
        'segmented_dmp_target_trajectory': []
       }

for i in range(len(data)):
    print('Generating {}/{}'.format(i + 1, len(data)))
    d = data[i]
    image = array(d['processed_image']).reshape(DATA['image_dim'])
    segments = d['segments'][1:-1]
    
    processed_traj = segments[0].traj[0][2]
    for seg in segments[1:]:
        processed_traj = np.append(processed_traj, seg.traj[0][2], axis = 0)
    
    origin = segments[0].traj[0][2][0]
    rot_traj = []
    rot_dmps = []
    rot_dmps_tau = []
    rot_dmps_w = []
    rot_dmps_y0 = []
    rot_dmps_goal = []
    rot_y = []
    rot_target_y = []
    rot_dy = []
    rot_ddy = []
    
    for j, seg in enumerate(segments):
        traj = deepcopy(seg.traj[0][2])
        # print(np.sum(np.abs(traj[-1] - traj[0])))
        traj = rot3d(origin, traj, rot_degs, order = rot_order)
        
        # print(np.sum(np.abs(traj[-1] - traj[0])), '\n')
        # traj = rot3d(origin, traj, [45., 45., 0.], order = ['z', 'y', 'x'])
        rot_traj.append(traj)
        dmp = DMPs_discrete(n_dmps = traj.shape[1], 
                            n_bfs = DATA['segmented_dmp_bf'], 
                            dt = DATA['segmented_dmp_dt'], 
                            ay = np.ones(traj.shape[1]) * DATA['segmented_dmp_ay'])
        
        dmp_target = DMPs_discrete(n_dmps = traj.shape[1], 
                            n_bfs = DATA['segmented_dmp_target_bf'], 
                            dt = DATA['segmented_dmp_dt'], 
                            ay = np.ones(traj.shape[1]) * 25)
        
        dmp.imitate_path(traj.T)
        dmp_target.imitate_path(traj.T)
        # tau = (1 / DATA['segmented_dmp_dt']) / traj.shape[0]
        tau = 1.
        y, dy, ddy = dmp.rollout(tau = tau)
        target_y, _, _ = dmp_target.rollout(tau = tau)
        
        rot_y.append(y)
        rot_target_y.append(target_y)
        rot_dy.append(dy)
        rot_ddy.append(ddy)
        rot_dmps.append(dmp)
        rot_dmps_w.append(dmp.w)
        rot_dmps_tau.append(tau)
        rot_dmps_y0.append(dmp.y0)
        rot_dmps_goal.append(dmp.goal)
        
    rot_target_y_full = rot_target_y[0]
    for j in rot_target_y[1:]:
        rot_target_y_full = np.append(rot_target_y_full, j, axis = 0)
    
    rot_traj_full = rot_traj[0]
    for j in rot_traj[1:]:
        rot_traj_full = np.append(rot_traj_full, j, axis = 0)
    
    rot_dmp_traj = rot_y[0]
    for j in rot_y[1:]:
        rot_dmp_traj = np.append(rot_dmp_traj, j, axis = 0)
    
    dmp = DMPs_discrete(n_dmps = traj.shape[1], 
                        n_bfs = DATA['normal_dmp_bf'], 
                        dt = DATA['normal_dmp_dt'], 
                        ay = np.ones(traj.shape[1]) * DATA['normal_dmp_ay'])
    dmp_accurate = DMPs_discrete(n_dmps = traj.shape[1], 
                        n_bfs = DATA['normal_dmp_L_bf'], 
                        dt = DATA['normal_dmp_dt'], 
                        ay = np.ones(traj.shape[1]) * DATA['normal_dmp_L_ay'])
    dmp.imitate_path(rot_traj_full.T)
    dmp_accurate.imitate_path(rot_traj_full.T)
    # tau = (1 / 0.001) / rot_traj_full.shape[0]
    tau = 1.
    y, dy, ddy = dmp.rollout(tau = tau)
    y_accurate, _, _ = dmp_accurate.rollout(tau = tau)
    
    DATA['image'].append(image)
    DATA['original_trajectory'].append(d['traj'])
    DATA['processed_trajectory'].append(processed_traj)
    DATA['rotated_trajectory'].append(rot_traj_full)
    DATA['normal_dmp_y0'].append(dmp.y0)
    DATA['normal_dmp_goal'].append(dmp.goal)
    DATA['normal_dmp_w'].append(dmp.w)
    DATA['normal_dmp_tau'].append(tau)
    DATA['normal_dmp_trajectory'].append(y)
    DATA['normal_dmp_L_y0'].append(dmp_accurate.y0)
    DATA['normal_dmp_L_goal'].append(dmp_accurate.goal)
    DATA['normal_dmp_L_w'].append(dmp_accurate.w)
    DATA['normal_dmp_L_tau'].append(tau)
    DATA['normal_dmp_L_trajectory'].append(y_accurate)
    DATA['normal_dmp_target_trajectory'].append(y_accurate)
    DATA['segmented_dmp_seg_num'].append(len(rot_traj))
    DATA['segmented_dmp_y0'].append(rot_dmps_y0)
    DATA['segmented_dmp_goal'].append(rot_dmps_goal)
    DATA['segmented_dmp_w'].append(rot_dmps_w)
    DATA['segmented_dmp_tau'].append(rot_dmps_tau)
    DATA['segmented_dmp_trajectory'].append(rot_dmp_traj)
    DATA['segmented_dmp_target_trajectory'].append(rot_target_y_full)
    
    # fig, ax = create_fig(3)
    # ax[0].scatter(range(y.shape[0]), y[:, 0], c = 'c')
    # ax[1].scatter(range(y.shape[0]), y[:, 1], c = 'c')
    # ax[2].scatter(range(y.shape[0]), y[:, 2], c = 'c')
    
    # ax[0].scatter(range(y_accurate.shape[0]), y_accurate[:, 0], c = 'b')
    # ax[1].scatter(range(y_accurate.shape[0]), y_accurate[:, 1], c = 'b')
    # ax[2].scatter(range(y_accurate.shape[0]), y_accurate[:, 2], c = 'b')
    
    # ax[0].scatter(range(rot_target_y_full.shape[0]), rot_target_y_full[:, 0], c = 'r')
    # ax[1].scatter(range(rot_target_y_full.shape[0]), rot_target_y_full[:, 1], c = 'r')
    # ax[2].scatter(range(rot_target_y_full.shape[0]), rot_target_y_full[:, 2], c = 'r')
    
    # start_x = 0
    # for y in rot_y:
    #     ax[0].scatter(range(start_x, start_x + y.shape[0]), y[:, 0], c = 'g')
    #     ax[1].scatter(range(start_x, start_x + y.shape[0]), y[:, 1], c = 'g')
    #     ax[2].scatter(range(start_x, start_x + y.shape[0]), y[:, 2], c = 'g')
    #     start_x += y.shape[0]
    # plt.show()
    # input()
    
    # data[i]['rot_traj'] = rot_traj
    # data[i]['rot_origin'] = origin
    # data[i]['rot_dmps_tau'] = 
    
    # traj -= origin
    # traj = (rot_x @ rot_y @ rot_z @ traj.T).T
    
    
    # traj = (rot_x @ traj.T).T
    # traj += origin
    # rotated_traj.append(traj)
    
# #%%
# traj = rotated_traj[2]
# # rotated_traj = segments[3].traj[0][2]
# plt.figure(figsize = (16, 16))
# plt.axis('equal')
# plt.plot(traj[:, 2], traj[:, 1])
    
#%%

for i in range(len(DATA['image'])):
    original_traj = DATA['original_trajectory'][i]
    rot_traj = DATA['rotated_trajectory'][i]
    dmp_traj = DATA['normal_dmp_trajectory'][i]
    dmp_L_traj = DATA['normal_dmp_L_trajectory'][i]
    seg_dmp_traj = DATA['segmented_dmp_trajectory'][i]
    
    fig, ax = create_fig(3)
    
    # start_x = 0
    # ax[0].scatter(range(start_x, start_x + original_traj.shape[0]), original_traj[:, 0])
    # ax[1].scatter(range(start_x, start_x + original_traj.shape[0]), original_traj[:, 1])
    # ax[2].scatter(range(start_x, start_x + original_traj.shape[0]), original_traj[:, 2])
    # start_x += original_traj.shape[0]
    
    start_x = 0
    ax[0].scatter(range(start_x, start_x + rot_traj.shape[0]), rot_traj[:, 0], c = 'g')
    ax[1].scatter(range(start_x, start_x + rot_traj.shape[0]), rot_traj[:, 1], c = 'g')
    ax[2].scatter(range(start_x, start_x + rot_traj.shape[0]), rot_traj[:, 2], c = 'g')
    start_x += rot_traj.shape[0]
    
    start_x = 0
    ax[0].scatter(range(start_x, start_x + dmp_traj.shape[0]), dmp_traj[:, 0], c = 'r')
    ax[1].scatter(range(start_x, start_x + dmp_traj.shape[0]), dmp_traj[:, 1], c = 'r')
    ax[2].scatter(range(start_x, start_x + dmp_traj.shape[0]), dmp_traj[:, 2], c = 'r')
    start_x += dmp_traj.shape[0]
    
    start_x = 0
    ax[0].scatter(range(start_x, start_x + dmp_L_traj.shape[0]), dmp_L_traj[:, 0], c = 'b')
    ax[1].scatter(range(start_x, start_x + dmp_L_traj.shape[0]), dmp_L_traj[:, 1], c = 'b')
    ax[2].scatter(range(start_x, start_x + dmp_L_traj.shape[0]), dmp_L_traj[:, 2], c = 'b')
    start_x += dmp_L_traj.shape[0]
    
    start_x = 0
    ax[0].scatter(range(start_x, start_x + seg_dmp_traj.shape[0]), seg_dmp_traj[:, 0], c = 'c')
    ax[1].scatter(range(start_x, start_x + seg_dmp_traj.shape[0]), seg_dmp_traj[:, 1], c = 'c')
    ax[2].scatter(range(start_x, start_x + seg_dmp_traj.shape[0]), seg_dmp_traj[:, 2], c = 'c')
    start_x += seg_dmp_traj.shape[0]
    
    print(i)
    print('CIMEDNet w interval = {}'.format(DATA['normal_dmp_w'][i].max() - DATA['normal_dmp_w'][i].min()))
    print('CIMEDNet_L w interval = {}'.format(DATA['normal_dmp_L_w'][i].max() - DATA['normal_dmp_L_w'][i].min()))
    print('DSDNet w interval = {}'.format(np.array(DATA['segmented_dmp_w'][i]).max() - np.array(DATA['segmented_dmp_w'][i]).min()))
    
    plt.show()
    input()
#%%
# dmps[2][1].y0 = np.array([0.0])
# dmps[2][1].goal = np.array([0.0])
# # dmps[2][0].w[0, -1] = -dmps[2][0].w[0, -2]

# ys = []
# for i, dmp in enumerate(dmps):
#     ax_ys = []
#     for j, ax_dmp in enumerate(dmp):
#         y, dy, ddy = ax_dmp.rollout(tau = taus[i][j])
#         ax_ys.append(y)
#     comb_ys = ax_ys[0]
#     comb_ys = np.append(comb_ys, ax_ys[1], axis = 1)
#     comb_ys = np.append(comb_ys, ax_ys[2], axis = 1)
#     ys.append(comb_ys)

# #%%
# fig, ax = create_fig(3)
# start_x = 0
# for y in ys:
#     ax[0].scatter(range(start_x, start_x + y.shape[0]), y[:, 0])
#     ax[1].scatter(range(start_x, start_x + y.shape[0]), y[:, 1])
#     ax[2].scatter(range(start_x, start_x + y.shape[0]), y[:, 2])
#     start_x += y.shape[0]
    
#%%
"""d = dataset[randint(0, len(dataset))]
# d = dataset[-1]
gen_img = np.zeros((img_dim[0], padded_img_width))
gen_traj = None
img_x_start = 0
traj_last_value = np.zeros(3)
# last_z_value = 0
trajs = []
cols = []
for i in range(len(d)):
    gen_img[:, img_x_start:img_x_start + d[i].image.shape[1]] = d[i].image[:, ::-1]
    img_x_start += d[i].image.shape[1]
    
    for j, t in enumerate(d[i].traj):
        if gen_traj is None:
            gen_traj = t[2]
            trajs.append(t[2])
        else:
            gen_traj = np.append(gen_traj, traj_last_value + t[2], axis = 0)
            trajs.append(traj_last_value + t[2])
        if j == 0:
            cols.append('b')
        if j == 1:
            cols.append('c')
        if j == 2:
            cols.append('m')
    # last_z_value = gen_traj[-1, 2]
    traj_last_value = deepcopy(gen_traj[-1, :])
    traj_last_value[0] = 0.
    traj_last_value[1] = 0.

# Pad image
while img_x_start < padded_img_width:
    pad_x_limit = (padded_img_width - img_x_start) if (padded_img_width - img_x_start) < padding.shape[1] else None
    gen_img[:, img_x_start:img_x_start + padding[:, :pad_x_limit].shape[1]] = padding[:, :pad_x_limit]
    img_x_start += padding[:, :pad_x_limit].shape[1]
gen_img = gen_img[:, ::-1]
plt.imshow(gen_img)
plt.show()

fig, ax = create_fig(3)
# ax[0].scatter(range(gen_traj.shape[0]), gen_traj[:, 0])
# ax[1].scatter(range(gen_traj.shape[0]), gen_traj[:, 1])
# ax[2].scatter(range(gen_traj.shape[0]), gen_traj[:, 2])

tstep = 0
for i, traj in enumerate(trajs):
    ax[0].scatter(range(tstep, tstep + traj.shape[0]), traj[:, 0], c = cols[i])
    ax[1].scatter(range(tstep, tstep + traj.shape[0]), traj[:, 1], c = cols[i])
    ax[2].scatter(range(tstep, tstep + traj.shape[0]), traj[:, 2], c = cols[i])
    tstep += traj.shape[0]
plt.show()

# max_segments = """
#%%
base_bf = 20
plot_traj = 0

DATA = {'image': [],
        'image_dim': (1, img_dim[0], padded_img_width),
        'original_trajectory': [],
        'normal_dmp_seg_num': np.ones(len(dataset)).reshape(-1, 1),
        'normal_dmp_y0': [],
        'normal_dmp_goal': [],
        'normal_dmp_w': [],
        'normal_dmp_tau': [],
        'normal_dmp_dt': 0.0001,
        'normal_dmp_bf': max_segments * base_bf,
        'normal_dmp_ay': 25,
        'segmented_dmp_max_seg_num': max_segments,
        'segmented_dmp_seg_num': [],
        'segmented_dmp_y0': [],
        'segmented_dmp_goal': [],
        'segmented_dmp_w': [],
        'segmented_dmp_tau': [],
        'segmented_dmp_dt': 0.01,
        'segmented_dmp_bf': base_bf,
        'segmented_dmp_ay': 15,
        'segmented_dmp_trajectory': []
       }

for d_idx, d in enumerate(dataset):
    print('({}/{}) Generating DMPs parameters'.format(d_idx + 1, len(dataset)))
    
    if plot_traj: fig, ax = create_fig(3)
    
    gen_img = np.zeros((img_dim[0], padded_img_width))
    gen_traj = None
    img_x_start = 0
    traj_last_value = np.zeros(3)
    # last_z_value = 0
    trajs = []
    cols = []
    for i in range(len(d)):
        gen_img[:, img_x_start:img_x_start + d[i].image.shape[1]] = d[i].image[:, ::-1]
        img_x_start += d[i].image.shape[1]
        
        for j, t in enumerate(d[i].traj):
            if gen_traj is None:
                gen_traj = t[2]
                trajs.append(t[2])
            else:
                gen_traj = np.append(gen_traj, traj_last_value + t[2], axis = 0)
                trajs.append(traj_last_value + t[2])
            if j == 0:
                cols.append('b')
            if j == 1:
                cols.append('c')
            if j == 2:
                cols.append('m')
        # last_z_value = gen_traj[-1, 2]
        traj_last_value = deepcopy(gen_traj[-1, :])
        # traj_last_value[0] = 0.
        traj_last_value[1] = 0.
        
    if plot_traj: 
        ax[0].scatter(range(gen_traj.shape[0]), gen_traj[:, 0], c = 'r')
        ax[1].scatter(range(gen_traj.shape[0]), gen_traj[:, 1], c = 'r')
        ax[2].scatter(range(gen_traj.shape[0]), gen_traj[:, 2], c = 'r')
    
    # Pad image
    while img_x_start < padded_img_width:
        pad_x_limit = (padded_img_width - img_x_start) if (padded_img_width - img_x_start) < padding.shape[1] else None
        gen_img[:, img_x_start:img_x_start + padding[:, :pad_x_limit].shape[1]] = padding[:, :pad_x_limit]
        img_x_start += padding[:, :pad_x_limit].shape[1]
    gen_img = gen_img[:, ::-1].reshape(DATA['image_dim'])
    
    normal_dmp = DMPs_discrete(n_dmps   = gen_traj.shape[1], 
                               n_bfs    = DATA['normal_dmp_bf'], 
                               dt       = DATA['segmented_dmp_dt'],
                               ay       = np.ones(gen_traj.shape[1]) * DATA['segmented_dmp_ay'])
    normal_dmp.imitate_path(gen_traj.T)
    normal_dmp_tau = (1 / DATA['segmented_dmp_dt']) / gen_traj.shape[0]
    normal_y, normal_dy, normal_ddy = normal_dmp.rollout(tau = normal_dmp_tau)
    
    if plot_traj: 
        ax[0].scatter(range(normal_y.shape[0]), normal_y[:, 0], c = 'b')
        ax[1].scatter(range(normal_y.shape[0]), normal_y[:, 1], c = 'b')
        ax[2].scatter(range(normal_y.shape[0]), normal_y[:, 2], c = 'b')
    
    DATA['image'].append(gen_img)
    DATA['original_trajectory'].append(gen_traj)
    DATA['normal_dmp_y0'].append(normal_dmp.y0)
    DATA['normal_dmp_goal'].append(normal_dmp.goal)
    DATA['normal_dmp_w'].append(normal_dmp.w)
    DATA['normal_dmp_tau'].append(normal_dmp_tau)
    
    segment_dmp_y0s = []
    segment_dmp_goals = []
    segment_dmp_ws = []
    segment_dmp_taus = []
    segment_ys = []
    
    for i, seg in enumerate(trajs):
        segment_dmp = DMPs_discrete(n_dmps   = seg.shape[1], 
                                    n_bfs    = DATA['segmented_dmp_bf'], 
                                    dt       = DATA['segmented_dmp_dt'],
                                    ay       = np.ones(seg.shape[1]) * DATA['segmented_dmp_ay'])
        # if i > 0:
        #     seg_appended = np.append(trajs[i - 1][0, :].reshape(1, -1), seg, axis = 0)
        segment_dmp.imitate_path(seg.T)
        tau =  (1 / DATA['segmented_dmp_dt']) / seg.shape[0]
        # tau =  1.
        segment_y, segment_dy, segment_ddy = segment_dmp.rollout(tau = tau)
        segment_ys.append(segment_y)
        
        segment_dmp_y0s.append(segment_dmp.y0)
        segment_dmp_goals.append(segment_dmp.goal)
        segment_dmp_ws.append(segment_dmp.w)
        segment_dmp_taus.append(tau)
    
    if plot_traj:
        start_idx = 0
        for seg in segment_ys:
            ax[0].scatter(range(start_idx, start_idx + seg.shape[0]), seg[:, 0], c = 'g')
            ax[1].scatter(range(start_idx, start_idx + seg.shape[0]), seg[:, 1], c = 'g')
            ax[2].scatter(range(start_idx, start_idx + seg.shape[0]), seg[:, 2], c = 'g')
            start_idx = start_idx + seg.shape[0]
        plt.show()
    
    segmented_dmp_traj = segment_ys[0]
    for j in range(1, len(segment_ys)):
        segmented_dmp_traj = np.append(segmented_dmp_traj, segment_ys[j], axis = 0)
    
    DATA['segmented_dmp_trajectory'].append(segmented_dmp_traj)
    DATA['segmented_dmp_seg_num'].append(len(trajs))
    DATA['segmented_dmp_y0'].append(segment_dmp_y0s)
    DATA['segmented_dmp_goal'].append(segment_dmp_goals)
    DATA['segmented_dmp_w'].append(segment_dmp_ws)
    DATA['segmented_dmp_tau'].append(segment_dmp_taus)
    
    if plot_traj: input('{}/{}'.format(d_idx + 1, len(dataset)))
    
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
unique_lengths = []
for i in to_process['segmented_dmp_target_trajectory']:
    if len(i) not in unique_lengths:
        unique_lengths.append(len(i))
unique_lengths = sorted(unique_lengths)
ranges = [[unique_lengths[i], unique_lengths[i + 1]] for i in range(len(unique_lengths) - 1)]
40
traj_pads = [[] for i in ranges]
for i in range(len(to_process['segmented_dmp_target_trajectory'])):
    traj = to_process['segmented_dmp_target_trajectory'][i]
    for j, r in enumerate(ranges):
        if r[1] <= traj.shape[0]:
            traj_pads[j].append(traj[r[0]:r[1]])
traj_pads = [array(i).mean(axis = 0) for i in traj_pads]

for i in range(len(to_process['segmented_dmp_target_trajectory'])):
    traj = to_process['segmented_dmp_target_trajectory'][i]
    for j, t in enumerate(ranges):
        if traj.shape[0] == t[0]:
            traj = np.append(traj, traj_pads[j], axis = 0)
    to_process['segmented_dmp_target_trajectory'][i] = traj

#%%
DATA = to_process
data_len = len(DATA['image'])
dof = DATA['original_trajectory'][0].shape[1]

DATA['image']                   = array(DATA['image'])
DATA['normal_dmp_goal']         = array(DATA['normal_dmp_goal']).reshape(data_len, 1, dof)
DATA['normal_dmp_w']            = array(DATA['normal_dmp_w']).reshape(data_len, 1, dof, DATA['normal_dmp_bf'])
DATA['normal_dmp_y0']           = array(DATA['normal_dmp_y0']).reshape(data_len, 1, dof)
DATA['normal_dmp_tau']          = array(DATA['normal_dmp_tau']).reshape(-1, 1)
DATA['normal_dmp_trajectory']   = array(DATA['normal_dmp_trajectory'])
DATA['normal_dmp_L_goal']         = array(DATA['normal_dmp_L_goal']).reshape(data_len, 1, dof)
DATA['normal_dmp_L_w']            = array(DATA['normal_dmp_L_w']).reshape(data_len, 1, dof, DATA['normal_dmp_L_bf'])
DATA['normal_dmp_L_y0']           = array(DATA['normal_dmp_L_y0']).reshape(data_len, 1, dof)
DATA['normal_dmp_L_tau']          = array(DATA['normal_dmp_L_tau']).reshape(-1, 1)
DATA['normal_dmp_L_trajectory']   = array(DATA['normal_dmp_L_trajectory'])
DATA['normal_dmp_target_trajectory']   = array(DATA['normal_dmp_target_trajectory'])
DATA['normal_dmp_tau']          = array(DATA['normal_dmp_tau']).reshape(-1, 1)
DATA['segmented_dmp_seg_num']   = array(DATA['segmented_dmp_seg_num']).reshape(-1, 1)
DATA['segmented_dmp_goal']      = array(DATA['segmented_dmp_goal'])
DATA['segmented_dmp_tau']       = array(DATA['segmented_dmp_tau'])
DATA['segmented_dmp_w']         = array(DATA['segmented_dmp_w'])
DATA['segmented_dmp_y0']        = array(DATA['segmented_dmp_y0'])
DATA['segmented_dmp_target_trajectory']        = array(DATA['segmented_dmp_target_trajectory'])
if 'rotation_degrees' in DATA:  DATA['rotation_degrees'] = array(DATA['rotation_degrees'])
#%%
# dist_w = np.abs((DATA['segmented_dmp_w'] - DATA['segmented_dmp_w'].mean(axis = 0)).mean(axis = (1, 2, 3)))
# dist_w_sorted = np.sort(dist_w)
# dist_w_filtered = dist_w < 20

w_intervals = np.array([(DATA['segmented_dmp_w'][i].max() - DATA['segmented_dmp_w'][i].min()) for i in range(DATA['segmented_dmp_w'].shape[0])])
w_intervals_sorted = np.sort(w_intervals)
dist_w_filtered = w_intervals < 5000

print('Passed = {}'.format(len([i for i in dist_w_filtered if i])))
# input()
#%%
DATA_FILTERED = deepcopy(DATA)

# DATA_FILTERED['bag_name'] = [j for i,j in enumerate(DATA['bag_name']) if dist_w_filtered[i]]
# DATA_FILTERED['image_name'] = [j for i,j in enumerate(DATA['image_name']) if dist_w_filtered[i]]
DATA_FILTERED['original_trajectory'] = [j for i,j in enumerate(DATA['original_trajectory']) if dist_w_filtered[i]]
DATA_FILTERED['processed_trajectory'] = [j for i,j in enumerate(DATA['processed_trajectory']) if dist_w_filtered[i]]
DATA_FILTERED['rotated_trajectory'] = [j for i,j in enumerate(DATA['rotated_trajectory']) if dist_w_filtered[i]]
DATA_FILTERED['image'] = DATA['image'][dist_w_filtered]

DATA_FILTERED['normal_dmp_seg_num'] = DATA['normal_dmp_seg_num'][dist_w_filtered]
DATA_FILTERED['normal_dmp_goal'] = DATA['normal_dmp_goal'][dist_w_filtered]
DATA_FILTERED['normal_dmp_tau'] = DATA['normal_dmp_tau'][dist_w_filtered]
DATA_FILTERED['normal_dmp_w'] = DATA['normal_dmp_w'][dist_w_filtered]
DATA_FILTERED['normal_dmp_y0'] = DATA['normal_dmp_y0'][dist_w_filtered]
DATA_FILTERED['normal_dmp_trajectory'] = DATA['normal_dmp_trajectory'][dist_w_filtered]

DATA_FILTERED['normal_dmp_L_seg_num'] = DATA['normal_dmp_seg_num'][dist_w_filtered]
DATA_FILTERED['normal_dmp_L_goal'] = DATA['normal_dmp_L_goal'][dist_w_filtered]
DATA_FILTERED['normal_dmp_L_tau'] = DATA['normal_dmp_L_tau'][dist_w_filtered]
DATA_FILTERED['normal_dmp_L_w'] = DATA['normal_dmp_L_w'][dist_w_filtered]
DATA_FILTERED['normal_dmp_L_y0'] = DATA['normal_dmp_L_y0'][dist_w_filtered]
DATA_FILTERED['normal_dmp_L_trajectory'] = DATA['normal_dmp_L_trajectory'][dist_w_filtered]

DATA_FILTERED['normal_dmp_target_trajectory'] = DATA['normal_dmp_target_trajectory'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_target_trajectory'] = DATA['segmented_dmp_target_trajectory'][dist_w_filtered]

DATA_FILTERED['segmented_dmp_seg_num'] = DATA['segmented_dmp_seg_num'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_goal'] = DATA['segmented_dmp_goal'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_tau'] = DATA['segmented_dmp_tau'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_w'] = DATA['segmented_dmp_w'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_y0'] = DATA['segmented_dmp_y0'][dist_w_filtered]
DATA_FILTERED['segmented_dmp_trajectory'] = [j for i, j in enumerate(DATA['segmented_dmp_trajectory']) if dist_w_filtered[i]]

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
to_pkl = DATA_FILTERED
# to_pkl = DATA

ROOT_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
PKL_DIR = 'data/pkl/cutting'
PKL_NAME = 'rotated_real_distanced_trajectory'
PKL_NAME += '.num_data_' + str(len(to_pkl['image'])) + '_num_seg_' + str(to_pkl['segmented_dmp_max_seg_num'])
PKL_NAME += '.normal_dmp_bf_' + str(to_pkl['normal_dmp_bf']) + '_ay_' + str(to_pkl['normal_dmp_ay']) + '_dt_' + str(to_pkl['normal_dmp_dt'])
PKL_NAME += '.seg_dmp_bf_' + str(to_pkl['segmented_dmp_bf']) + '_ay_' + str(to_pkl['segmented_dmp_ay']) + '_dt_' + str(to_pkl['segmented_dmp_dt'])
PKL_NAME += '.' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
PKL_NAME += '.pkl'
pkl.dump(to_pkl, open(join(ROOT_DIR, PKL_DIR, PKL_NAME), 'wb'))
print('Saved as {}'.format(PKL_NAME))