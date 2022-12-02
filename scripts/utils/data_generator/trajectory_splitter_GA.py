#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:23:28 2022

@author: edgar
"""
import numpy as np
from numpy import array
from deap import base, creator, benchmarks, tools, algorithms, cma
import bagpy
from bagpy import bagreader
from os.path import join
from os import listdir
import pandas as pd
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmps import DMPs_discrete
from datetime import datetime
from statistics import variance, stdev
from scipy.spatial.distance import cdist
import copy

def dmp_weight_generator(traj, n_bf, dt = None, ay = None):
    pass
    
def trajectory_cleaner(df):
    t = np.array(df['Time'])
    x = np.array(df['pose.position.x'])
    y = np.array(df['pose.position.y'])
    z = np.array(df['pose.position.z'])
    
    dt = np.diff(t)
    dt_rank = np.argsort(dt)[::-1]
    
    idx = 0
    while dt[dt_rank[idx]] > 0.5:
        fig, ax = bagpy.create_fig(3)
        ax[0].scatter(t, x)
        ax[1].scatter(t, y)
        ax[2].scatter(t, z)
        
        ax[0].plot([t[dt_rank[idx] + 1]] * 2, [x.min(), x.max()], c = 'r')
        ax[1].plot([t[dt_rank[idx] + 1]] * 2, [y.min(), y.max()], c = 'r')
        ax[2].plot([t[dt_rank[idx] + 1]] * 2, [z.min(), z.max()], c = 'r')
        plt.show()
        
        ans = input("Check trajectory trimming ['Enter' to proceed | 'n' for skip]")
        if ans == '':
            print('Trimmed')
            df = df[dt_rank[idx] + 1:]
        idx += 1
    
    df = df.reset_index(drop = True)
    return df

def plot_traj(df, minimum_points, pred_minimum_idx = None):
    fig, ax = bagpy.create_fig(7)
    
    for i in range(len(minimum_points['index']) - 1):
        ax[0].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.position.x'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[1].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.position.y'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[2].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.position.z'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[3].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.velocity.x'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[4].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.velocity.y'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[5].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.velocity.z'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[6].scatter(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.velocity.total'][minimum_points['index'][i]:minimum_points['index'][i + 1]], s = 1)
        ax[6].plot(df.index[minimum_points['index'][i]:minimum_points['index'][i + 1]], 
                      df['pose.velocity.total'][minimum_points['index'][i]:minimum_points['index'][i + 1]], lw = 0.25)
    
    for i in range(len(minimum_points['index'])):
        if pred_minimum_idx != None and i not in pred_minimum_idx: continue
        ax[0].plot([minimum_points['index'][i]] * 2, [df['pose.position.x'].min(), df['pose.position.x'].max()], c = 'r')
        ax[1].plot([minimum_points['index'][i]] * 2, [df['pose.position.y'].min(), df['pose.position.y'].max()], c = 'r')
        ax[2].plot([minimum_points['index'][i]] * 2, [df['pose.position.z'].min(), df['pose.position.z'].max()], c = 'r')
        ax[3].plot([minimum_points['index'][i]] * 2, [df['pose.velocity.x'].min(), df['pose.velocity.x'].max()], c = 'r')
        ax[4].plot([minimum_points['index'][i]] * 2, [df['pose.velocity.y'].min(), df['pose.velocity.y'].max()], c = 'r')
        ax[5].plot([minimum_points['index'][i]] * 2, [df['pose.velocity.z'].min(), df['pose.velocity.z'].max()], c = 'r')
        ax[6].plot([minimum_points['index'][i]] * 2, [df['pose.velocity.total'].min(), df['pose.velocity.total'].max()], c = 'r')

    ylabel = ['position.x', 'position.y', 'position.z', 'velocity.x', 'velocity.y', 'velocity.z', 'velocity.total']
    for i, axis in enumerate(ax):
        # axis.legend()
        axis.set_xlabel('Time')
        axis.set_ylabel(ylabel[i])

    plt.show()

def detect_lowest_velocity_sum(df, num_segments, point_min_dist):
    x = np.array(df_cleaned['pose.position.x'])
    y = np.array(df_cleaned['pose.position.y'])
    z = np.array(df_cleaned['pose.position.z'])

    traj = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis = 1)
    
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    dx = np.concatenate(([dx[0]], dx))
    dy = np.concatenate(([dy[0]], dy))
    dz = np.concatenate(([dz[0]], dz))

    df['pose.velocity.x'] = dx
    df['pose.velocity.y'] = dy
    df['pose.velocity.z'] = dz
    df['pose.velocity.total'] = np.abs(dx + dy + dz)
    df['pose.velocity.min'] = np.min([dx.reshape(-1, 1), dy.reshape(-1, 1), dz.reshape(-1, 1)], axis = 0)
    
    df_sorted_vel = df.sort_values(by = 'pose.velocity.total')
    points_to_find = num_segments - 1
    minimum_points = {'index': [],
                      'velocity': [],
                      'time': []}

    i = 0
    # points_found = 0
    first_is_lowest = False
    last_is_lowest = False
    while len(minimum_points['index']) < points_to_find and i < len(df_sorted_vel):
        
        if df_sorted_vel.index[i] == 0:
            first_is_lowest = True
            points_to_find += 1
                
        if df_sorted_vel.index[i] == len(df) - 1:
            last_is_lowest = True
            points_to_find += 1
            
        in_window = False
        for j in minimum_points['index']:
            if df_sorted_vel.index[i] > j - point_min_dist and df_sorted_vel.index[i] < j + point_min_dist:
                # print(df_sorted_vel['pose.velocity.total'][i])
                in_window = True
                break
        if not in_window:
            minimum_points['index'].append(df_sorted_vel.index[i])
            minimum_points['velocity'].append(df_sorted_vel['pose.velocity.total'].iloc[i])
            minimum_points['time'].append(df_sorted_vel['Time'].iloc[i])
        i += 1

    if not first_is_lowest:
        minimum_points['index'].insert(0, 0)
        minimum_points['velocity'].insert(0, 0)
        minimum_points['time'].insert(0, df['Time'].iloc[0] - (df['Time'].iloc[1] - df['Time'].iloc[0]))

    if not last_is_lowest:
        minimum_points['index'].append(len(df_sorted_vel) - 1)
        minimum_points['velocity'].append(0.)
        minimum_points['time'].append(df['Time'].iloc[-1] + (df['Time'].iloc[-1] - df['Time'].iloc[-2]))

    minimum_points['index'] = np.array(minimum_points['index'])
    minimum_points['velocity'] = np.array(minimum_points['velocity'])
    minimum_points['time'] = np.array(minimum_points['time'])

    idx_sort = np.argsort(minimum_points['index'])
    minimum_points['index'] = minimum_points['index'][idx_sort]
    minimum_points['velocity'] = minimum_points['velocity'][idx_sort]
    minimum_points['time'] = minimum_points['time'][idx_sort]
    
    minimum_points['index'][-1] += 1
    
    # print(minimum_points['index'])
    plot_traj(df, minimum_points)
    
    
    return traj, minimum_points, np.array(df['pose.velocity.total'])

def mse(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert a.shape == b.shape
    return ((a - b)**2).mean()
  
class ReferenceSplitter:
    def __init__(self, num_segments, max_reference):
        pass

    def generate_gaussian_centers(self, traj, method, num_centers):
        pass

    def add_reference(self, reference, method = 'low_vel', min_dist = 10, min_length = 10):
        """
        method options:
        1. low_vel
        2. rand_low_vel
        3. eq_dist
        """
        pass

    def check_reference(self):
        pass

    def segment(self, traj):
        pass
    
class CMAESReferenceSplitter:
    def __init__(self, max_segments, ref_traj = None, point_min_dist = 10, method = 'ref_region_dmd_modes'):
        self.max_segments = max_segments
        self.point_min_dist = point_min_dist
        self.dof = None
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        
        if method == 'ref_traj':
            assert ref_traj is not None
            self.dof = ref_traj.shape[1]
            self.ref_traj = ref_traj - ref_traj.min(axis = 0)
            self.ref_dmds = [DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True) for i in range(self.dof)]
            for i in range(self.dof):
                self.ref_dmds[i].fit([i for i in self.ref_traj[:, i].reshape(-1, 1)])
            # [print(dmd.modes.shape) for dmd in self.ref_dmds]
            # self.ref_dmd = DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True)
            # self.ref_dmd.fit([i for i in self.ref_traj])
        
            self.toolbox.register("evaluate", self.fitness_func)
            
        elif method == 'ref_region_dmd_modes':
            assert ref_traj is not None
            self.dof = ref_traj.shape[1]
            self.ref_traj = ref_traj
            # self.ref_traj = ref_traj - ref_traj.min(axis = 0)
            self.compare_length = ref_traj.shape[0]
            # print(self.ref_traj)
            self.ref_dmds = [DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True).fit(list(self.ref_traj[i])) for i in range(self.compare_length)]
        
            self.toolbox.register("evaluate", self.low_vel_seg_points_region_dmd_modes_fitness_func)
            
        elif method == 'ref_region_pos_profile':
            assert ref_traj is not None
            self.dof = ref_traj.shape[1]
            self.compare_length = ref_traj.shape[0]
            self.ref_pos = ref_traj - ref_traj.min(axis = 0)
            self.toolbox.register("evaluate", self.region_pos_profile_fitness_func)
            
        elif method == 'ref_region_vel_profile':
            assert ref_traj is not None
            self.dof = ref_traj.shape[1]
            self.compare_length = ref_traj.shape[0]
            self.ref_vel = np.diff(ref_traj, axis = 0)
            self.toolbox.register("evaluate", self.region_vel_profile_fitness_func)
            
        elif method == 'ref_region_pos_vel_profile':
            assert ref_traj is not None
            self.dof = ref_traj.shape[1]
            self.compare_length = ref_traj.shape[0]
            self.ref_pos = ref_traj - ref_traj.min(axis = 0)
            self.ref_vel = np.diff(self.ref_pos, axis = 0)
            self.toolbox.register("evaluate", self.region_pos_vel_profile_fitness_func)
            
        else:
            self.toolbox.register("evaluate", self.unsupervised_fitness_func)
        
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
    def preprocess(self, individual):
        # num_segments = int(np.clip(np.round(individual[0]), a_min = 3, a_max = self.max_segments))
        
        # Apply absolute to individual and filter values larger than trajectory length
        seg_points = [np.abs(i) for i in individual if np.abs(i) < self.traj_to_segment.shape[0]] #and np.abs(i) > 0
        # Add start and end points
        seg_points = [0] + seg_points + [self.traj_to_segment.shape[0]]
        # Sort segment points
        seg_points = sorted([int(np.round(i)) for i in seg_points])
        # Filter points that are below point_min_dist
        seg_points = [0] + [seg_points[i] for i in range(1, len(seg_points)) if seg_points[i] - seg_points[i - 1] > self.point_min_dist]
        # print(seg_points)
        # if seg_points[0] != 0:
        #     if seg_points[0] >= 2:
        #         seg_points = [0] + seg_points
        #     elif seg_points[0] < 2:
        #         seg_points[0] = 0
        return seg_points
    
    def individual_to_idx(self, individual):
        individual = np.round(np.clip(individual, a_min = 1, a_max = self.max_segments // 2))
        # print(individual)
        total_seg = 0
        idx = [0]
        cur_idx = 0
        while len(idx) < self.max_segments and total_seg < len(self.low_vel_segments) :
            total_seg += individual[cur_idx]
            if total_seg < len(self.low_vel_segments):
                idx.append(int(total_seg))
            else:
                idx.append(len(self.low_vel_segments))
            cur_idx += 1
        return idx
        
    def fitness_func(self, individual):
        seg_points = self.preprocess(individual)
        segments = [self.traj_to_segment[seg_points[i]:seg_points[i + 1]] for i in range(len(seg_points) - 1)]
        # print([int(np.round(i)) for i in seg_points])
        # print([seg.shape for seg in segments], '\n')
        total_error = 0
        if len(segments) <= 3: total_error+= 1000
        
        dmd_mse = 0
        for seg in segments:
            # seg = seg - seg.min(axis = 0)
            # print(seg)
            
            for axis in range(self.dof):
                # print(seg[axis])
                # input()
                dmd = DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True)
                dmd.fit([i for i in seg[axis].reshape(-1, 1)])
                dmd_mse += ((dmd.modes - self.ref_dmds[axis].modes)**2).mean()
                
            # dmd = DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True)
            # dmd.fit([i for i in seg])
            # dmd_mse += ((dmd.modes - self.ref_dmd.modes)**2).mean()
        
        # var = variance([len(i) for i in segments[1:]])
        if len(segments[1:]) > 1: std = stdev([len(i) for i in segments[1:]])
        # print(var)
                
        total_error += dmd_mse
        # if len(segments[1:]) > 1: total_error += std * 0.15
        # total_error = total_error + dmd_mse
        # print(seg_points, total_error)
        return (total_error,)
    
    def unsupervised_fitness_func(self, individual):
        individual = np.round(np.clip(individual, a_min = 1, a_max = self.max_segments // 2))
        print(individual)
        total_seg = 0
        idx = [0]
        cur_idx = 0
        while len(idx) < self.max_segments and total_seg < len(self.low_vel_segments) :
            total_seg += individual[cur_idx]
            if total_seg < len(self.low_vel_segments):
                idx.append(int(total_seg))
            else:
                idx.append(len(self.low_vel_segments))
            cur_idx += 1
        print(idx, len(self.low_vel_segments))
        combined_seg = []
        for i in range(1, len(idx)):
            to_combine = []
            for j in range(idx[i - 1], idx[i]):
                to_combine += self.low_vel_segments[j]
            combined_seg.append(to_combine)
        print([len(i) for i in combined_seg])
        # combined_seg = [self.low_vel_segments[idx[i - 1]:idx[i]] for i in range(1, idx)]
        combined_seg_dmd_modes = [DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True).fit([i for i in seg]).modes for seg in combined_seg]
        dist = cdist(combined_seg_dmd_modes, combined_seg_dmd_modes, 'euclidean')
        mse = 0
        print(dist.shape)
        for i in range(dist.shape[0] - 1):
            for j in range(i + 1, dist.shape[1]):
                mse += dist[i, j]
        
        return mse
    
    def unsupervised_dmp_w_similarity_fitness_func(self, individual):
        idx = self.individual_to_idx(individual)
        total_error = 0 if len(idx) > 3 else 1000
        
        for i in range(1, len(idx) - 1):
            
            
        print(idx, total_error)
        return (total_error,)
    
    def low_vel_seg_points_region_dmd_modes_fitness_func(self, individual):
        idx = self.individual_to_idx(individual)
        total_error = 0 if len(idx) > 3 else 1000
        
        for i in range(1, len(idx) - 1):
            for j in range(self.compare_length):
                seg_region_dmd = DMD(svd_rank = 1.0, tlsq_rank = 0, exact = True, opt = True).fit(self.low_vel_segments[idx[i]][j])
                total_error += mse(seg_region_dmd.modes, self.ref_dmds[j].modes)
        
        print(idx, total_error)
        return (total_error,)
    
    def region_pos_profile_fitness_func(self, individual):
        idx = self.individual_to_idx(individual)
        total_error = 0 if len(idx) > 3 else 1000
        
        for i in range(1, len(idx) - 1):
            pred_seg = self.low_vel_segments[idx[i]][:self.compare_length]
            total_error += mse(self.ref_pos, pred_seg - pred_seg.min(axis = 0))
            
        print(idx, total_error)
        return (total_error,)
    
    def region_vel_profile_fitness_func(self, individual):
        idx = self.individual_to_idx(individual)
        total_error = 0 if len(idx) > 3 else 1000
        
        for i in range(1, len(idx) - 1):
            pred_seg = self.low_vel_segments[idx[i]][:self.compare_length]
            pred_vel = np.diff(pred_seg, axis = 0)
            total_error += mse(self.ref_vel, pred_vel)
            
        print(idx, total_error)
        return (total_error,)
    
    def region_pos_vel_profile_fitness_func(self, individual):
        idx = self.individual_to_idx(individual)
        total_error = 0 if len(idx) > 3 else 1000
        
        for i in range(1, len(idx) - 1):
            pred_seg = self.low_vel_segments[idx[i]][:self.compare_length]
            pred_vel = np.diff(pred_seg, axis = 0)
            total_error += mse(self.ref_pos, pred_seg - pred_seg.min(axis = 0))
            total_error += mse(self.ref_vel, pred_vel)
            
        print(idx, total_error)
        return (total_error,)
        
    def register_trajectory(self, traj, minimum_points):
        if self.dof == None: self.dof = traj.shape[1]
        self.traj_to_segment = traj
        self.minimum_points = minimum_points
        self.low_vel_segments = [self.traj_to_segment[self.minimum_points['index'][i-1]:self.minimum_points['index'][i]] for i in range(1, len(self.minimum_points['index']))]
        # print([len(i) for i in self.low_vel_segments])
        # search_space = list(np.linspace(0, traj.shape[0], num = self.max_segments + 1)[1:-1])
        # cma_es = cma.Strategy(centroid=search_space, sigma=25.0, lambda_=5*self.max_segments)
        search_space = list(np.ones(self.max_segments))
        cma_es = cma.Strategy(centroid=search_space, sigma=2.0, lambda_=10*self.max_segments)
        self.toolbox.register("generate", cma_es.generate, creator.Individual)
        self.toolbox.register("update", cma_es.update)
    
    def segment(self):
        pop, logbook = algorithms.eaGenerateUpdate(self.toolbox, ngen=100, stats=self.stats, 
                                                   halloffame=self.hof, verbose=True)
            
        print("Best individual is %s, fitness: %s" % (self.hof[0], self.hof[0].fitness.values))
        # print("Segment points: {}".format(self.preprocess(self.hof[0])))
        # return self.preprocess(self.hof[0])
        print("Segment points: {}".format(self.individual_to_idx(self.hof[0])))
        return self.individual_to_idx(self.hof[0])
    
def compare_segments(traj, low_vel_idx, ref_seg, slide_range = 10, dist_threshold = 0.5):
    ref_length = ref_seg.shape[0]
    point_mse = []
    for center in low_vel_idx[1:-1]:
        start = center - slide_range
        end = center + slide_range + 1
        lowest_mse = np.inf
        for window_start in range(start, end):
            compared_seg = traj[window_start:window_start + ref_length]
            ref_traj_temp = copy.deepcopy(ref_seg)
            if compared_seg.shape[0] < ref_seg.shape[0]: ref_traj_temp = ref_traj_temp[ref_seg.shape[0] - compared_seg.shape[0]:]
            compared_seg = compared_seg - compared_seg.min(axis = 0)
            loss = mse(compared_seg, ref_traj_temp)
            if loss < lowest_mse: lowest_mse = loss
        point_mse.append(lowest_mse)
        # print(center, lowest_mse)
    return point_mse

def compare_ratio(traj, minimum_points):
    traj_ratio = []
    temp_traj = np.array(traj[1:-1])
    temp_traj -= temp_traj.min(axis = 0)
    for dof in range(traj.shape[1]):
        dof_ratio = []
        dof_min = traj[:, dof].min()
        dof_max = traj[:, dof].max()
        for i in minimum_points['index'][1:-1]:
            ratio = temp_traj[i, dof] / (dof_max - dof_min)
            dof_ratio.append(ratio)
        traj_ratio.append([0] + dof_ratio + [0])
    return traj_ratio

def threshold_ratio(ratios, dof_threshold):
    pass
#%%
if __name__=='__main__':
    bag_root = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/recordings/'
    bag_type = 'cutting'
    bag_path = join(bag_root, bag_type, 'bag')
    bags = [i for i in listdir(bag_path) if i[-3:] == 'bag']
    #%
    ref_bag_idx = 127
    b = bagreader(join(bag_path, bags[ref_bag_idx]), verbose = False)
    
    bag_csv = b.message_by_topic('/mocap_pose_topic/knife_marker_pose')
    df_bag = pd.read_csv(bag_csv)
    df_cleaned = trajectory_cleaner(df_bag)
    ref_traj, ref_minimum_points, ref_total_vel = detect_lowest_velocity_sum(df_cleaned, num_segments = 30, point_min_dist = 15)
    #%%
    ref_segment_center = 151
    ref_segment_start = ref_segment_center - 5
    ref_segment_end = ref_segment_center + 5
    ref_seg = ref_traj[ref_segment_start:ref_segment_end]
    # ref_seg = ref_seg[::-1]
    ref_seg = (ref_seg - ref_seg.min())[:, :2]
    # ref_seg = ref_seg[:10]
    #%
    
    fig, ax = bagpy.create_fig(2)
    ax[0].set_title('Reference Trajectory | bag:{}'.format(bag_idx))
    ax[0].plot(ref_seg[:, 0])
    ax[1].plot(ref_seg[:, 1])
    # ax[2].plot(ref_seg[:, 2])
    #%%
    runcell(0, '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/scripts/utils/data_generator/trajectory_splitter.py')
    
    bag_idx = 131
    b = bagreader(join(bag_path, bags[bag_idx]), verbose = False)
    
    bag_csv = b.message_by_topic('/mocap_pose_topic/knife_marker_pose')
    df_bag = pd.read_csv(bag_csv)
    df_cleaned = trajectory_cleaner(df_bag)
    #%
    traj, minimum_points, total_vel = detect_lowest_velocity_sum(df_cleaned, num_segments = 30, point_min_dist = 10)
    traj = traj - traj.min(axis = 0)
    #%%
    check = compare_ratio(traj[:, :], minimum_points)
    c = np.round(array(check[0]) / array(check[1]))
    check_2 = [minimum_points['index'][0]] + np.array(minimum_points['index'])[(c >= 3) * (c <= 6)].tolist() + [minimum_points['index'][-1]]
    # check_2 = [minimum_points['index'][0]] + (np.array(minimum_points['index'][1:-1])[np.array(check[0]) > 0.3]).tolist() + [minimum_points['index'][-1]]
    check_2_idx = [i for i in range(len(minimum_points['index'])) if minimum_points['index'][i] in check_2]
    plot_traj(df_cleaned, minimum_points, check_2_idx)
    # check_2 = threshold_ratio(check, [0.6])
    #%%
    check = compare_segments(traj, minimum_points['index'], ref_seg)
    lowest_idx = np.argsort(check)
    #%%
    splitter = CMAESReferenceSplitter(max_segments = 10, ref_traj = ref_seg)
    splitter.register_trajectory(traj[:, :2], minimum_points)
    start_time = datetime.now()
    pred = splitter.segment()
    runtime = (datetime.now() - start_time).seconds
    print('{} | Runtime: {}s'.format(bag_idx, runtime))
    
    plot_traj(df_cleaned, minimum_points, pred)
    #%%
    fig, ax = bagpy.create_fig(4)
    
    ax[0].set_title('{} | Runtime: {}s\nX-axis'.format(bag_idx, runtime))
    ax[1].set_title('Y-axis')
    ax[2].set_title('Z-axis')
    ax[3].set_title('Total velocity')
    
    ax[0].plot(range(traj[:, 0].shape[0]), traj[:, 0])
    ax[0].scatter(range(traj[:, 0].shape[0]), traj[:, 0])
    
    ax[1].plot(range(traj[:, 1].shape[0]), traj[:, 1])
    ax[1].scatter(range(traj[:, 1].shape[0]), traj[:, 1])
    
    ax[2].plot(range(traj[:, 2].shape[0]), traj[:, 2])
    ax[2].scatter(range(traj[:, 2].shape[0]), traj[:, 2])
    
    ax[3].plot(range(total_vel.shape[0]), total_vel)
    ax[3].scatter(range(total_vel.shape[0]), total_vel)
    
    for i in range(len(pred) - 1):
        ax[0].plot([pred[i]] * 2, [traj[:, 0].min(), traj[:, 0].max()], c = 'r')
        ax[1].plot([pred[i]] * 2, [traj[:, 1].min(), traj[:, 1].max()], c = 'r')
        ax[2].plot([pred[i]] * 2, [traj[:, 2].min(), traj[:, 2].max()], c = 'r')
        ax[3].plot([pred[i]] * 2, [total_vel.min(), total_vel.max()], c = 'r')
        
    ax[0].plot([traj.shape[0]] * 2, [traj[:, 0].min(), traj[:, 0].max()], c = 'r')
    ax[1].plot([traj.shape[0]] * 2, [traj[:, 1].min(), traj[:, 1].max()], c = 'r')
    ax[2].plot([traj.shape[0]] * 2, [traj[:, 2].min(), traj[:, 2].max()], c = 'r')
    ax[3].plot([total_vel.shape[0]] * 2, [total_vel.min(), total_vel.max()], c = 'r')
    
    plt.show()
    
    # plt.figure(figsize = (15, 5))
    # plt.title('{} | Runtime: {}s'.format(bag_idx, runtime))
    # plt.plot(range(traj[:, axis].shape[0]), traj[:, axis])
    # plt.scatter(range(traj[:, axis].shape[0]), traj[:, axis])
    # for i in range(len(pred) - 1):
    #     plt.plot([pred[i]] * 2, [traj[:, axis].min(), traj[:, axis].max()], c = 'r')
    # plt.plot([traj.shape[0]] * 2, [traj[:, axis].min(), traj[:, axis].max()], c = 'r')
