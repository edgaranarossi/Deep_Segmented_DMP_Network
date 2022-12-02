# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:13:59 2022

@author: edgar
"""

import bagpy
from bagpy import bagreader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os.path import join, isdir
from os import listdir
import shutil

def trim_ends(df, upper_limit = 5e-4):
    begin_idx = 0
    end_idx = len(df) - 1
    
    while df_bag['pose.velocity.abs_total'][begin_idx] < upper_limit: begin_idx += 1
    while df_bag['pose.velocity.abs_total'][end_idx] < upper_limit: end_idx -= 1
    
    return df.iloc[begin_idx:end_idx + 1]

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
                splits.append(split)
            split = []
        cur_idx += 1
    
    segments = []
    start_idx = 0
    for split in splits:
        segments.append([0, [start_idx, split[0]]])
        segments.append([1, [split[0], split[-1]]])
        start_idx = split[-1]
    segments.append([0, [start_idx, len(df)]])
    return segments

bag_dir = 'C:\\cutting_motion_recordings'
bag_path = join(bag_dir)
bag_dirs = [i for i in listdir(bag_path) if isdir(join(bag_path, i))]
for i in bag_dirs: shutil.rmtree(join(bag_path, i))
bags = [i for i in listdir(bag_path) if i[-3:] == 'bag']

bag_idx = -1
b = bagreader(join(bag_path, bags[bag_idx]), verbose = False)

bag_csv = b.message_by_topic('/mocap_pose_topic/knife_marker_pose')
df_bag = pd.read_csv(bag_csv)

x = np.array(df_bag['pose.position.x'])
y = np.array(df_bag['pose.position.y'])
z = np.array(df_bag['pose.position.z'])

traj = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis = 1)

dx = np.diff(x)
dy = np.diff(y)
dz = np.diff(z)

dx = np.concatenate(([dx[0]], dx))
dy = np.concatenate(([dy[0]], dy))
dz = np.concatenate(([dz[0]], dz))

df_bag['pose.velocity.x'] = dx
df_bag['pose.velocity.y'] = dy
df_bag['pose.velocity.z'] = dz
df_bag['pose.velocity.abs_x'] = np.abs(dx)
df_bag['pose.velocity.abs_y'] = np.abs(dy)
df_bag['pose.velocity.abs_z'] = np.abs(dz)
df_bag['pose.velocity.abs_xy'] = np.abs(dx) + np.abs(dy)
df_bag['pose.velocity.abs_yz'] = np.abs(dy) + np.abs(dz)
df_bag['pose.velocity.abs_xz'] = np.abs(dx) + np.abs(dz)
df_bag['pose.velocity.abs_total'] = np.abs(dx) + np.abs(dy) + np.abs(dz)

df = trim_ends(df_bag, upper_limit = 4e-4)
# df_bag['pose.velocity.min'] = np.min([dx.reshape(-1, 1), dy.reshape(-1, 1), dz.reshape(-1, 1)], axis = 0)
#%
# splits = detect_split(df, 'pose.velocity.abs_total', upper_limit = 4e-4)
# # splits = detect_split(df, 'pose.velocity.abs_x')
# splits_flatten = [j for i in splits for j in i]

# to_plot = np.array(df['pose.velocity.abs_total'])
# # to_plot = np.array(df['pose.velocity.abs_x'])
# plt.figure(figsize = (56, 10))
# for i in range(len(df)):
#     if i not in splits_flatten:
#         plt.scatter(i, to_plot[i], c = 'b')
#     else:
#         plt.scatter(i, to_plot[i], c = 'r')
# plt.show()


segments = detect_split(df, 'pose.velocity.abs_total', min_length = 50, upper_limit = 5e-4)
# segments = detect_split(df, 'pose.velocity.abs_xy', min_length = 40, upper_limit = 5e-4)
fig, ax = bagpy.create_fig(7)
for segment in segments:
    if segment[0] == 0: 
        col = 'b'
    else: 
        col = 'r'
    
    ax[0].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.position.x'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)
    ax[1].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.position.y'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)
    ax[2].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.position.z'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)
    ax[3].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.velocity.x'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)
    ax[4].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.velocity.y'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)
    ax[5].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.velocity.z'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)
    ax[6].scatter(range(segment[1][0], segment[1][1]), 
                  df['pose.velocity.abs_total'].iloc[segment[1][0]:segment[1][1]], s = 1, c = col)