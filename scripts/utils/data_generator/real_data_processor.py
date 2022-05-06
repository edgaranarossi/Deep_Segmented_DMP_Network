import bagpy
from bagpy import bagreader
import numpy as np
from numpy import array, floor, ceil, round, loadtxt, diff, concatenate, sort, zeros, ones
from os import listdir
from os.path import join, isdir
import pickle as pkl
from PIL import Image, ImageOps
from pydmps import DMPs_discrete
from datetime import datetime

def lowestVelocityTrajectorySplitter(trajectory, num_segments, point_min_dist):
    
    points_to_find = num_segments - 1
    
    velocity = diff(trajectory, axis = 0)
    velocity = concatenate([zeros((1, 3)), velocity], axis = 0)
    total_velocity = velocity.sum(axis = 1).reshape(-1, 1)
    
    indexed_table = concatenate([array(range(trajectory.shape[0])).reshape(-1, 1), trajectory, total_velocity], axis = 1)
    velocity_sorted_indexed_table = indexed_table[np.argsort(indexed_table[:, -1]), :]
    
    minimum_points = {'index': [],
                      'velocity': []
                      }

    i = 0
    first_is_lowest = False
    last_is_lowest = False
    while len(minimum_points['index']) < points_to_find and i < indexed_table.shape[0]:
        
        if not first_is_lowest and velocity_sorted_indexed_table[i, 0] == 0:
            first_is_lowest = True
            points_to_find += 1
        elif not last_is_lowest and velocity_sorted_indexed_table[i, 0] == indexed_table.shape[0] - 1:
            last_is_lowest = True
            points_to_find += 1
            
        in_window = False
        for j in minimum_points['index']:
            if velocity_sorted_indexed_table[i, 0] > j - point_min_dist and velocity_sorted_indexed_table[i, 0] < j + point_min_dist:
                in_window = True
                break
                
        if not in_window:
            minimum_points['index'].append(velocity_sorted_indexed_table[i, 0])
            minimum_points['velocity'].append(velocity_sorted_indexed_table[i, -1])
            
            if velocity_sorted_indexed_table[i, 0] == 1:
                first_is_lowest = True
                points_to_find += 1
            elif velocity_sorted_indexed_table[i, 0] == indexed_table.shape[0] - 2:
                last_is_lowest = True
                points_to_find += 1
                
        i += 1

    if not first_is_lowest:
        minimum_points['index'].insert(0, 0)
        minimum_points['velocity'].insert(0, 0)

    if not last_is_lowest:
        minimum_points['index'].append(indexed_table.shape[0])
        minimum_points['velocity'].append(0.)

    minimum_points['index'] = np.array(minimum_points['index'], dtype = 'uint32')
    minimum_points['velocity'] = np.array(minimum_points['velocity'])

    idx_sort = np.argsort(minimum_points['index'])
    minimum_points['index'] = minimum_points['index'][idx_sort]
    minimum_points['velocity'] = minimum_points['velocity'][idx_sort]
    
    segments = []
    for i in range(len(minimum_points['index'])-1):
        segments.append(trajectory[minimum_points['index'][i]:minimum_points['index'][i+1]])
    
    return segments, minimum_points

ROOT_DIR = 'D:\\rllab\\scripts\\dmp\\Segmented_Deep_DMPs'
DATA_DIR = 'data\\recordings\\cutting'
IMAGE_DIR = join(ROOT_DIR, DATA_DIR, 'images')
BAG_DIR = join(ROOT_DIR, DATA_DIR, 'bag')

images = listdir(IMAGE_DIR)
bags = [i for i in listdir(BAG_DIR) if i[-3:] == 'bag']

DATA = {'image_name': [],
        'bag_name': [],
        'image': [],
        'image_dim': (1, 100, 100),
        'original_trajectory': [],
        'normal_dmp_y0': [],
        'normal_dmp_goal': [],
        'normal_dmp_w': [],
        'normal_dmp_tau': 1.,
        'normal_dmp_dt': 0.001,
        'normal_dmp_bf': 300,
        'normal_dmp_ay': 25,
        'segmented_dmp_num': 10,
        'segmented_dmp_y0': [],
        'segmented_dmp_goal': [],
        'segmented_dmp_w': [],
        'segmented_dmp_tau': [],
        'segmented_dmp_dt': 0.015,
        'segmented_dmp_bf': 30,
        'segmented_dmp_ay': 10,
        'splitter_min_distance': 10
        }

for i, (image_name, bag_name) in enumerate(zip(images, bags)):
    print("({}/{})".format(i + 1, len(images)), "Processing", image_name)
    image = Image.open(join(IMAGE_DIR, image_name)).resize(DATA['image_dim'][1:])
    if DATA['image_dim'][0] == 1: image = ImageOps.grayscale(image)
    image = array(image).reshape(DATA['image_dim'])
    
    bag = bagreader(join(BAG_DIR, bag_name), verbose = False)
    bag_csv = bag.message_by_topic('/mocap_pose_topic/knife_marker_pose')
    trajectory = loadtxt(open(join(BAG_DIR, bag_csv), 'rb'), delimiter = ',', skiprows = 1, usecols = (5, 6, 7))
    
    normal_dmp = DMPs_discrete(n_dmps   = trajectory.shape[1], 
                               n_bfs    = DATA['normal_dmp_bf'], 
                               dt       = DATA['normal_dmp_dt'],
                               ay       = np.ones(trajectory.shape[1]) * DATA['normal_dmp_ay'])
    normal_dmp.imitate_path(trajectory.T)
    normal_y, normal_dy, normal_ddy = normal_dmp.rollout()
    
    segments, minimum_points = lowestVelocityTrajectorySplitter(trajectory = trajectory,
                                                num_segments = DATA['segmented_dmp_num'],
                                                point_min_dist = DATA['splitter_min_distance'])
    
    segment_dmp_y0s = []
    segment_dmp_goals = []
    segment_dmp_ws = []
    segment_dmp_taus = []
    for segment in segments:
        segment_dmp = DMPs_discrete(n_dmps   = trajectory.shape[1], 
                                    n_bfs    = DATA['segmented_dmp_bf'], 
                                    dt       = DATA['segmented_dmp_dt'],
                                    ay       = np.ones(trajectory.shape[1]) * DATA['segmented_dmp_ay'])
        segment_dmp.imitate_path(segment.T)
        segment_y, segment_dy, segment_ddy = segment_dmp.rollout()
        
        tau =  (1 / DATA['segmented_dmp_dt']) / segment.shape[0]
        
        segment_dmp_y0s.append(segment_dmp.y0)
        segment_dmp_goals.append(segment_dmp.goal)
        segment_dmp_ws.append(segment_dmp.w)
        segment_dmp_taus.append(tau)
    
    DATA['image_name'].append(image_name)
    DATA['bag_name'].append(bag_name)
    DATA['image'].append(image)
    DATA['original_trajectory'].append(trajectory)
    DATA['normal_dmp_y0'].append(normal_dmp.y0)
    DATA['normal_dmp_goal'].append(normal_dmp.goal)
    DATA['normal_dmp_w'].append(normal_dmp.w)
    DATA['segmented_dmp_y0'].append(segment_dmp_y0s)
    DATA['segmented_dmp_goal'].append(segment_dmp_goals)
    DATA['segmented_dmp_w'].append(segment_dmp_ws)
    DATA['segmented_dmp_tau'].append(segment_dmp_taus)
    
DATA['image']               = array(DATA['image'])
DATA['normal_dmp_goal']     = array(DATA['normal_dmp_goal'])
DATA['normal_dmp_w']        = array(DATA['normal_dmp_w'])
DATA['normal_dmp_y0']       = array(DATA['normal_dmp_y0'])
DATA['segmented_dmp_goal']  = array(DATA['segmented_dmp_goal'])
DATA['segmented_dmp_tau']   = array(DATA['segmented_dmp_tau'])
DATA['segmented_dmp_w']     = array(DATA['segmented_dmp_w'])
DATA['segmented_dmp_y0']    = array(DATA['segmented_dmp_y0'])

PKL_DIR = 'data\\pkl\\cutting_traj'
PKL_NAME = 'real_trajectory'
PKL_NAME += '.num_data_' + str(len(DATA['image_name'])) + '_num_seg_' + str(DATA['segmented_dmp_num'])
PKL_NAME += '.normal_dmp_bf_' + str(DATA['normal_dmp_bf']) + '_ay_' + str(DATA['normal_dmp_ay']) + '_dt_' + str(DATA['normal_dmp_dt'])
PKL_NAME += '.seg_dmp_bf_' + str(DATA['segmented_dmp_bf']) + '_ay_' + str(DATA['segmented_dmp_ay']) + '_dt_' + str(DATA['segmented_dmp_dt'])
PKL_NAME += '.' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
PKL_NAME += '.pkl'
pkl.dump(DATA, open(join(ROOT_DIR, PKL_DIR, PKL_NAME), 'wb'))

print("Generated in {}".format(join(ROOT_DIR, PKL_DIR, PKL_NAME)))
