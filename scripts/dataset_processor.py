#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 23:17:28 2022

@author: edgar
"""

import numpy as np
from numpy import array, zeros_like, sin, cos
from numpy.random import rand, randint
from matplotlib import pyplot as plt
from pydmps import DMPs_discrete
from datetime import datetime
from os.path import join, isdir
from os import listdir, makedirs
from PIL import Image
import pickle as pkl
from copy import copy, deepcopy

def ndarray_to_str(arr):
    assert len(arr.shape) == 2
    s = ''
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            s += str(arr[i, j])
            s += ','
        s = s[:-1]
        s += ';'
    return s
    
def generateDMP(x, dmp_param):
    dmp = DMPs_discrete(n_dmps = x.shape[1],
                        n_bfs = dmp_param.n_bfs,
                        ay = np.ones(x.shape[1]) * dmp_param.ay,
                        dt = dmp_param.dt)
    dmp.imitate_path(x.T)
    return dmp

def rot2D(traj, degrees, origin = np.array([0., 0.])):
    
    s = sin(degrees)
    c = cos(degrees)
    
    # traj -= origin
    
    t = deepcopy(traj)
    
    t -= origin
    
    t[:, 0] = traj[:, 0] * c - traj[:, 1] * s
    t[:, 1] = traj[:, 0] * s + traj[:, 1] * c
    
    t += origin
    
    return t

def rot3D(traj, degrees, origin = None, order = None):
    deg_x, deg_y, deg_z = degrees
    deg_x = np.deg2rad(deg_x)
    deg_y = np.deg2rad(deg_y)
    deg_z = np.deg2rad(deg_z)
    
    if origin is None: origin = copy(traj[0, :])
    if order is None: order = ['x', 'y', 'z']
    
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
    t += origin
    
    return t

class DMPParameters:
    def __init__(self, n_dmps, n_bfs, ay, dt, tau = 1):
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.ay = ay
        self.dt = dt
        self.tau = tau

class DatasetGenerator:
    def __init__(self, 
                 dataset_path,
                 img_path, 
                 pkl_path, 
                 img_dim, 
                 used_dim,
                 remove_start_end = True,
                 rotation_degrees = None,
                 rotation_order = ['x', 'y', 'z']):
        self.dataset_path = dataset_path
        if not isdir(self.dataset_path): makedirs(self.dataset_path)
        
        self.image_names = sorted(listdir(img_path))
        self.pkl_names = sorted(listdir(pkl_path))
        
        self.images = [np.array(Image.open(join(img_path, i)).resize((img_dim[1], img_dim[2]))).reshape(img_dim) for i in self.image_names]
        self.motions = [pkl.load(open(join(pkl_path, i), 'rb')) for i in self.pkl_names]
        
        self.data_num  = len(self.images)
        self.used_dim = sorted(used_dim)
        self.dof = len(used_dim)
        
        self.DATA = {'image': np.array(self.images),
                     'image_dim': img_dim,
                     'original_trajectory': [],
                     'rotation_degrees': rotation_degrees,
                     'rotation_order': rotation_order,
                     'normal_dmp_seg_num': None,
                     'normal_dmp_y0': [],
                     'normal_dmp_goal': [],
                     'normal_dmp_w': [],
                     'normal_dmp_tau': [],
                     'normal_dmp_bf': None,
                     'normal_dmp_ay': None,
                     'normal_dmp_trajectory': [],
                     'normal_dmp_L_y0': [],
                     'normal_dmp_L_goal': [],
                     'normal_dmp_L_w': [],
                     'normal_dmp_L_tau': [],
                     'normal_dmp_L_bf': None,
                     'normal_dmp_L_ay': None,
                     'normal_dmp_L_trajectory': [],
                     'normal_dmp_dt': None,
                     'segmented_dmp_max_seg_num': None,
                     'segmented_dmp_seg_num': [],
                     'segmented_dmp_y0': [],
                     'segmented_dmp_goal': [],
                     'segmented_dmp_w': [],     
                     'segmented_dmp_tau': [],
                     'segmented_dmp_bf': None,
                     'segmented_dmp_ay': None,
                     'segmented_dmp_dt': None,
                     'segmented_dmp_trajectory': []
                    }
                    
        self.parseMotion(remove_start_end)
        
    def parseMotion(self, remove_start_end):
        print('Parsing motion...')
        self.segmented_motions = []
        
        segment_template = deepcopy(self.motions[0])
        for i in segment_template:
            segment_template[i] = []
        
        self.max_segment = 0
        for motion in self.motions:
            prev_segment = -1
            segments = []
            for t in range(len(motion['timestamp'])):
                if motion['segment'][t] != prev_segment:
                    segments.append(deepcopy(segment_template))
                for i in motion:
                    segments[-1][i].append(motion[i][t])
                prev_segment = motion['segment'][t]
            if remove_start_end: segments = segments[1:-1]
            if len(segments) > self.max_segment: self.max_segment = len(segments)
            for seg in segments:
                for i in seg:
                    seg[i] = np.array(seg[i])
                    
            trimmed_segments = []
            for seg in segments:
                new_seg = deepcopy(seg)
                new_seg['ee_pose'] = seg['ee_pose'][:, self.used_dim[0]].reshape(-1, 1)
                for dim in self.used_dim[1:]:
                    new_seg['ee_pose'] = np.append(new_seg['ee_pose'], seg['ee_pose'][:, dim].reshape(-1, 1), axis = 1)
                trimmed_segments.append(new_seg)
                
            if self.DATA['rotation_degrees'] is not None:
                if trimmed_segments[0]['ee_pose'].shape[1] == 2:
                    origin_pos = trimmed_segments[0]['ee_pose'][0, :]
                    for seg in trimmed_segments:
                        seg['ee_pose'] = rot2D(seg['ee_pose'], self.DATA['rotation_degrees'], origin_pos)
                        
                elif trimmed_segments[0]['ee_pose'].shape[1] == 3:
                    origin_pos = trimmed_segments[0]['ee_pose'][0, :]
                    for seg in trimmed_segments:
                        seg['ee_pose'] = rot3D(seg['ee_pose'], self.DATA['rotation_degrees'], origin_pos, self.DATA['rotation_order'])
                
            full_motion = deepcopy(trimmed_segments[0]['ee_pose'])
            for seg in trimmed_segments[1:]:
                full_motion = np.append(full_motion, seg['ee_pose'], axis = 0)
                
            # plt.scatter(range(len(full_motion)), full_motion[:, 1])
            # plt.show()
            # input()
                
            self.DATA['original_trajectory'].append(full_motion)
            
            self.segmented_motions.append(trimmed_segments)
        
        self.DATA['segmented_dmp_max_seg_num'] = self.max_segment
        
    def generateName(self):
        self.generation_time = datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        self.dataset_name += '_num.{}_dof.{}'.format(self.data_num, self.dof)
        self.dataset_name += '_dsdnet[seg.{}-bf.{}-ay.{}-dt.{}]'.format(self.max_segment,
                                                                        self.dsdnet_dmp_param.n_bfs,
                                                                        self.dsdnet_dmp_param.ay,
                                                                        self.dsdnet_dmp_param.dt)
        self.dataset_name += '_cimednet[bf.{}-ay.{}-dt.{}]'.format(self.cimednet_dmp_param.n_bfs,
                                                                   self.cimednet_dmp_param.ay,
                                                                   self.cimednet_dmp_param.dt)
        self.dataset_name += '_cimednet_L[bf.{}-ay.{}-dt.{}]'.format(self.cimednet_dmp_L_param.n_bfs,
                                                                     self.cimednet_dmp_L_param.ay,
                                                                     self.cimednet_dmp_L_param.dt)
        self.dataset_name += self.generation_time
        self.dataset_name += '.pkl'
        
            
    def generate(self,
                 dataset_name,
                 dsdnet_dmp_param,
                 cimednet_dmp_param,
                 cimednet_dmp_L_param,
                 w_mean_filter = None):
        self.dataset_name = dataset_name
        self.dsdnet_dmp_param = dsdnet_dmp_param
        self.cimednet_dmp_param = cimednet_dmp_param
        self.cimednet_dmp_L_param = cimednet_dmp_L_param
        
        self.DATA['segmented_dmp_bf'] = self.dsdnet_dmp_param.n_bfs
        self.DATA['segmented_dmp_ay'] = self.dsdnet_dmp_param.ay
        self.DATA['segmented_dmp_dt'] = self.dsdnet_dmp_param.dt
        self.DATA['normal_dmp_bf'] = self.cimednet_dmp_param.n_bfs
        self.DATA['normal_dmp_ay'] = self.cimednet_dmp_param.ay
        self.DATA['normal_dmp_L_bf'] = self.cimednet_dmp_L_param.n_bfs
        self.DATA['normal_dmp_L_ay'] = self.cimednet_dmp_L_param.ay
        self.DATA['normal_dmp_dt'] = self.cimednet_dmp_L_param.dt
        
        self.generateDMPParameters()
        self.pad(w_mean_filter)
        self.listToStr()
        self.generateName()
        self.pklData()
        # return self.final_DATA
        
    def generateDMPParameters(self):
        print('Generating DMP parameters...')
        for i in range(self.data_num):
            dsdnet_y0s = []
            dsdnet_goals = []
            dsdnet_ws = []
            dsdnet_taus = []
            dsdnet_ys = []
            for j in self.segmented_motions[i]:
                traj = j['ee_pose']
                dsdnet_dmp = generateDMP(traj, self.dsdnet_dmp_param)
                dsdnet_y0s.append(dsdnet_dmp.y0)
                dsdnet_goals.append(dsdnet_dmp.goal)
                dsdnet_ws.append(dsdnet_dmp.w)
                dsdnet_taus.append((1 / self.DATA['segmented_dmp_dt']) / traj.shape[0])
                y, _, _ = dsdnet_dmp.rollout()
                dsdnet_ys.append(y)
            dsdnet_ys = np.array(dsdnet_ys).reshape(-1, self.dof)
            
            self.DATA['segmented_dmp_seg_num'].append(len(self.segmented_motions[i]))
            self.DATA['segmented_dmp_y0'].append(dsdnet_y0s)
            self.DATA['segmented_dmp_goal'].append(dsdnet_goals)
            self.DATA['segmented_dmp_w'].append(dsdnet_ws)
            self.DATA['segmented_dmp_tau'].append(dsdnet_taus)
            self.DATA['segmented_dmp_trajectory'].append(dsdnet_ys)
            
            traj = self.DATA['original_trajectory'][i]
            cimednet_dmp = generateDMP(traj, self.cimednet_dmp_param)
            self.DATA['normal_dmp_y0'].append(cimednet_dmp.y0)
            self.DATA['normal_dmp_goal'].append(cimednet_dmp.goal)
            self.DATA['normal_dmp_w'].append(cimednet_dmp.w)
            self.DATA['normal_dmp_tau'].append((1 / self.DATA['normal_dmp_dt']) / traj.shape[0])
            y, _, _ = cimednet_dmp.rollout()
            self.DATA['normal_dmp_trajectory'].append(y)
            
            cimednet_dmp_L = generateDMP(self.DATA['original_trajectory'][i], self.cimednet_dmp_L_param)
            self.DATA['normal_dmp_L_y0'].append(cimednet_dmp_L.y0)
            self.DATA['normal_dmp_L_goal'].append(cimednet_dmp_L.goal)
            self.DATA['normal_dmp_L_w'].append(cimednet_dmp_L.w)
            self.DATA['normal_dmp_L_tau'].append((1 / self.DATA['normal_dmp_dt']) / traj.shape[0])
            y, _, _ = cimednet_dmp_L.rollout()
            self.DATA['normal_dmp_L_trajectory'].append(y)
        
            
    def pad(self, w_mean_filter):
        to_process = deepcopy(self.DATA)

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
                    
        DATA = to_process

        DATA['image']                   = array(DATA['image'])
        DATA['normal_dmp_goal']         = array(DATA['normal_dmp_goal']).reshape(self.data_num, 1, self.dof)
        DATA['normal_dmp_w']            = array(DATA['normal_dmp_w']).reshape(self.data_num, 1, self.dof, DATA['normal_dmp_bf'])
        DATA['normal_dmp_y0']           = array(DATA['normal_dmp_y0']).reshape(self.data_num, 1, self.dof)
        DATA['normal_dmp_tau']          = array(DATA['normal_dmp_tau']).reshape(-1, 1)
        DATA['normal_dmp_L_goal']         = array(DATA['normal_dmp_L_goal']).reshape(self.data_num, 1, self.dof)
        DATA['normal_dmp_L_w']            = array(DATA['normal_dmp_L_w']).reshape(self.data_num, 1, self.dof, DATA['normal_dmp_L_bf'])
        DATA['normal_dmp_L_y0']           = array(DATA['normal_dmp_L_y0']).reshape(self.data_num, 1, self.dof)
        DATA['normal_dmp_L_tau']          = array(DATA['normal_dmp_tau']).reshape(-1, 1)
        DATA['segmented_dmp_seg_num']   = array(DATA['segmented_dmp_seg_num']).reshape(-1, 1)
        DATA['segmented_dmp_goal']      = array(DATA['segmented_dmp_goal'])
        DATA['segmented_dmp_tau']       = array(DATA['segmented_dmp_tau'])
        DATA['segmented_dmp_w']         = array(DATA['segmented_dmp_w'])
        DATA['segmented_dmp_y0']        = array(DATA['segmented_dmp_y0'])
        
        if w_mean_filter is not None:
            dist_w = np.abs((np.abs(DATA['segmented_dmp_w']) - np.abs(DATA['segmented_dmp_w']).mean(axis = 0)).mean(axis = (1, 2, 3)))
            dist_w_sorted = np.sort(dist_w)
            dist_w_filtered = dist_w < w_mean_filter
    
            print('Filtered = {}'.format(len([i for i in dist_w_filtered if i])))
            
            DATA_FILTERED = deepcopy(DATA)
            
            DATA_FILTERED['original_trajectory'] = [j for i,j in enumerate(DATA['original_trajectory']) if dist_w_filtered[i]]
            DATA_FILTERED['segmented_dmp_trajectory'] = [j for i,j in enumerate(DATA['segmented_dmp_trajectory']) if dist_w_filtered[i]]
            DATA_FILTERED['normal_dmp_trajectory'] = [j for i,j in enumerate(DATA['normal_dmp_trajectory']) if dist_w_filtered[i]]
            DATA_FILTERED['normal_dmp_L_trajectory'] = [j for i,j in enumerate(DATA['normal_dmp_L_trajectory']) if dist_w_filtered[i]]
            DATA_FILTERED['image'] = DATA['image'][dist_w_filtered]

            DATA_FILTERED['normal_dmp_y0'] = DATA['normal_dmp_y0'][dist_w_filtered]
            DATA_FILTERED['normal_dmp_goal'] = DATA['normal_dmp_goal'][dist_w_filtered]
            DATA_FILTERED['normal_dmp_w'] = DATA['normal_dmp_w'][dist_w_filtered]
            DATA_FILTERED['normal_dmp_tau'] = DATA['normal_dmp_tau'][dist_w_filtered]

            DATA_FILTERED['normal_dmp_L_y0'] = DATA['normal_dmp_L_y0'][dist_w_filtered]
            DATA_FILTERED['normal_dmp_L_goal'] = DATA['normal_dmp_L_goal'][dist_w_filtered]
            DATA_FILTERED['normal_dmp_L_w'] = DATA['normal_dmp_L_w'][dist_w_filtered]
            DATA_FILTERED['normal_dmp_L_tau'] = DATA['normal_dmp_L_tau'][dist_w_filtered]

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
            
            self.data_num = len(DATA_FILTERED['image'])
            self.padded_DATA = DATA_FILTERED
        else:
            self.padded_DATA = DATA
        self.final_DATA = self.padded_DATA
        DATA_FILTERED['normal_dmp_seg_num'] = np.ones(self.data_num).reshape(-1, 1)
    
    def listToStr(self):
        DATA = deepcopy(self.padded_DATA)
        for i in range(self.data_num):
            DATA['original_trajectory'][i] = ndarray_to_str(DATA['original_trajectory'][i])
            DATA['normal_dmp_trajectory'][i] = ndarray_to_str(DATA['normal_dmp_trajectory'][i])
            DATA['normal_dmp_L_trajectory'][i] = ndarray_to_str(DATA['normal_dmp_L_trajectory'][i])
            DATA['segmented_dmp_trajectory'][i] = ndarray_to_str(DATA['segmented_dmp_trajectory'][i])
        self.final_DATA = DATA
    
    def pklData(self):
        pkl.dump(self.final_DATA, open(join(self.dataset_path, self.dataset_name), 'wb'))
        print('Saved as\n{}'.format(self.dataset_name))

if __name__=='__main__':
    data_dir = '/home/edgar/rllab/data'
    data_name = 'pepper_shaking_6_target'
    img_dir = 'image/start'
    pkl_dir = 'motion'
    dataset_save_dir = join('/home/edgar/rllab/scripts/Segmented_Deep_DMPs/data/pkl/', data_name)
    generator = DatasetGenerator(dataset_path = dataset_save_dir,
                                 img_path = join(data_dir, data_name, img_dir), 
                                 pkl_path = join(data_dir, data_name, pkl_dir), 
                                 img_dim = (3, 100 ,100),
                                 used_dim = [0, 1, 2],
                                 rotation_degrees = [45, 45, 45],
                                 rotation_order = ['x', 'y', 'z'])
    
    n_dmps = 3
    base_bf = 50
    base_dt = 1e-3
    
    dsdnet_dmp_param = DMPParameters(n_dmps = n_dmps,
                                      n_bfs = base_bf,
                                      ay = 15,
                                      dt = base_dt * generator.max_segment)
                                     
    cimednet_dmp_param = DMPParameters(n_dmps = n_dmps,
                                        n_bfs = base_bf * generator.max_segment,
                                        ay = 25,
                                        dt = base_dt)
                                       
    cimednet_dmp_L_param = DMPParameters(n_dmps = n_dmps,
                                        n_bfs = 1000,
                                        ay = 25,
                                        dt = base_dt)
    
    generator.generate(dataset_name = data_name,
                       dsdnet_dmp_param = dsdnet_dmp_param,
                       cimednet_dmp_param = cimednet_dmp_param,
                       cimednet_dmp_L_param = cimednet_dmp_L_param,
                       w_mean_filter = 50)
    
    print('\nMin w = {}\nMax w = {}'.format(generator.final_DATA['segmented_dmp_w'].min(), generator.final_DATA['segmented_dmp_w'].max()))
    
    