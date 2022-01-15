#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:06:01 2021

@author: edgar
"""
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import numpy as np
from numpy import cos,sin,deg2rad,pi
from numpy.random import rand, randint, RandomState
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import figure
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
from PIL import Image
import matplotlib.image as mpimg
# from pympler import muppy, summary
import pandas as pd
# import gc
import pydmps
import pydmps.dmp_discrete
from os.path import join

TEX_ROOT_DIR = '/home/edgar/rllab/scripts/dmp/'

def Sum_points(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return x1+x2, y1+y2

def Multiply_point(multiplier, P):
    x, y = P
    return float(x)*float(multiplier), float(y)*float(multiplier)

def Check_if_object_is_polygon(Cartesian_coords_list):
    if Cartesian_coords_list[0] == Cartesian_coords_list[len(Cartesian_coords_list)-1]:
        return True
    else:
        return False

class Chaikin():

    def __init__(self, Cartesian_coords_list):
        self.Cartesian_coords_list = Cartesian_coords_list

    def Find_Q_point_position(self, P1, P2):
        Summand1 = Multiply_point(float(3)/float(4), P1)
        Summand2 = Multiply_point(float(1)/float(4), P2)
        Q = Sum_points(Summand1, Summand2) 
        return Q

    def Find_R_point_position(self, P1, P2):
        Summand1 = Multiply_point(float(1)/float(4), P1)
        Summand2 = Multiply_point(float(3)/float(4), P2)        
        R = Sum_points(Summand1, Summand2)
        return R

    def Smooth_by_Chaikin(self, number_of_refinements):
        refinement = 1
        copy_first_coord = Check_if_object_is_polygon(self.Cartesian_coords_list)
        while refinement <= number_of_refinements:
            self.New_cartesian_coords_list = []

            for num, tuple in enumerate(self.Cartesian_coords_list):
                if num+1 == len(self.Cartesian_coords_list):
                    pass
                else:
                    P1, P2 = (tuple, self.Cartesian_coords_list[num+1])
                    Q = self.Find_Q_point_position(P1, P2)
                    R = self.Find_R_point_position(P1, P2)
                    self.New_cartesian_coords_list.append(Q)
                    self.New_cartesian_coords_list.append(R)

            if copy_first_coord:
                self.New_cartesian_coords_list.append(self.New_cartesian_coords_list[0])

            self.Cartesian_coords_list = self.New_cartesian_coords_list
            refinement += 1
        return self.Cartesian_coords_list
    
base_shape = [
    [0      ,  2.5],
    [0.5    ,    0],
    [5      ,  0.4],
    [10     ,  0.8],
    [15     , 1.25],
    [18.5   ,  1.5],
    [20     ,  2.5],
    [18.5   ,  3.5],
    [15     , 3.75],
    [10     , 4.15],
    [5      ,  4.6],
    [0.5    ,    5]
    ]

scale = 10
base_shape = (np.array(base_shape)*scale).tolist()

def rotateCoord(point, center, orientation):
    px, py = point
    cx, cy = center
    new_x = cos(deg2rad(orientation)) * (px-cx) - sin(deg2rad(orientation)) * (py-cy) + cx
    new_y = sin(deg2rad(orientation)) * (px-cx) + cos(deg2rad(orientation)) * (py-cy) + cy
    # print(new_x, new_y)
    return new_x, new_y

def getOffset(orientation):
    diff_limit = 1e-4
    step = 1e-4
    multiplier = 0
    test_center = (1, 1)
    if orientation < 0:
        while rotateCoord(((1 + multiplier * step), 0), test_center, orientation)[1] > diff_limit:
            multiplier += 1
    elif orientation > 0:
        while rotateCoord(((1 + multiplier * step), 0), test_center, orientation)[1] > diff_limit:
            multiplier -= 1
    else:
        return (1, 0)
        
    displacement = rotateCoord(((1 + multiplier * step), 0), test_center, orientation)[0]
    return 1 + multiplier * step, displacement

def closestValueMaxHeight(carrot, traj_x):
    # closest = 0
    max_height = 0
    # diff = 1e9
    carrot_np = np.array(carrot)
    carrot_np_sub_x = np.append([carrot_np[:,0] - traj_x], [carrot_np[:,1]], axis=0).T
    carrot_np_sub_x = carrot_np_sub_x[np.abs(carrot_np_sub_x)[:,0].argsort()]
    
    for i in range(carrot_np_sub_x.shape[0]):
        if carrot_np_sub_x[i, 0] > 0 and carrot_np_sub_x[i, 1] > 0.1:
            if carrot_np_sub_x[i, 1] > max_height:
                max_height = carrot_np_sub_x[i, 1]
            else:
                break
    # print("START")
    # for i in range(start_idx, -1, -1):
    #     # if carrot[i][1] > 0.3 * np.array(carrot).max(axis = 0)[1]:
    #         # print(abs(carrot[i][0] - traj_x), diff)
    #     if abs(carrot[i][0] - traj_x) < diff:
    #         diff = abs(carrot[i][0] - traj_x)
    #         max_height = carrot[i][1]
    #         closest = i
    #     elif abs(carrot[i][0] - traj_x) > diff + 1:
    #         break
            # print(abs(carrot[i][0] - traj_x))
    # print(closest)
    return max_height#, closest

def travel(traj_list, cur_pos, next_pos, step, timestep):
    tstep = timestep
    cur_pos = np.array(cur_pos)
    next_pos = np.array(next_pos)
    
    diff_x = next_pos[0] - cur_pos[0]
    diff_y = next_pos[1] - cur_pos[1]
    
    # direction = np.sign(diff_x) if diff_x != 0 else np.sign(diff_y)
    
    if diff_x == 0:
        direction = np.sign(diff_y)
        pos_step = np.array([0, direction * step])
    elif diff_y == 0:
        direction = np.sign(diff_x)
        pos_step = np.array([direction * step, 0])
    else:
        direction_x = np.sign(diff_x)
        direction_y = np.sign(diff_y)
        pos_step = np.array([direction_x * step, direction_y * abs(diff_y / int(diff_x/step))])
    # step *= direction if direction != 0 else 1
    cur_pos += pos_step
    # cur_pos = start + step
    # prev_t = tstep
    # check = False
    dist = np.sqrt((next_pos[0] - cur_pos[0])**2 + (next_pos[1] - cur_pos[1])**2)
    while abs(dist) > abs(step):
        # check = True
        tstep += abs(step)//2
        traj_list.append([cur_pos[0], cur_pos[1], tstep])
        cur_pos += pos_step
        dist = np.sqrt((next_pos[0] - cur_pos[0])**2 + (next_pos[1] - cur_pos[1])**2)
        # print(dist, step)
        # if prev_t == tstep:
        #     print("# here")
        #     input()
        # else:
        #     prev_t = tstep
    tstep += abs(step)//2
    traj_list.append([cur_pos[0], next_pos[1], tstep])
    # if prev_t == tstep:
    #     print("* here", prev_t, tstep, check, direction)
    #     input()
    # else:
    #     prev_t = tstep
    return traj_list, tstep

def divideTraj(traj, step_size):
    tstep = 0
    new_traj = []
    for i in range(len(traj)-1):
        # print(traj[i][0])
        new_traj, tstep = travel(new_traj, traj[i], traj[i+1], step_size, tstep)
    new_traj_t = list(np.array(new_traj)[:,-1])
    new_traj = list(np.array(new_traj)[:,:-1])
    return new_traj, new_traj_t

def trajGenerator(distance = 10, 
                  add_height = 5, 
                  orientation = 0, 
                  carrot = None,
                  traj_step = 10, 
                  min_offset = None,
                  flip = False):
    traj = []
    direct_traj = []
    cut_displacements = []
    cut_number = 0
    sign = np.sign(orientation)
    # timestep = 0
    direct_timestep = 0
    # if sign == 0: sign = 1
    height_offset, displacement = getOffset(orientation)
    displacement *= sign
    
    carrot_leftmost = np.min(np.array(carrot)[:,0])
    carrot_rightmost = np.max(np.array(carrot)[:,0])
    num_cuts = int((carrot_rightmost - 2 * (min_offset if min_offset != None else 0)) / distance)
    tip_leftover = carrot_rightmost - (distance * num_cuts) - (2 * (min_offset if min_offset != None else 0))
    # carrot_start = carrot[0][0]
    
    traj_x = carrot_leftmost + (min_offset if min_offset != None else 0) + (tip_leftover / 2)
    # traj_start += displacement * cur_height
    # if sign > 0: traj_start += distance * 2
    
    # closest = len(carrot)-1
    cur_height = closestValueMaxHeight(carrot, traj_x)
    
    height = cur_height + add_height
    cut_displacements.append((sign-displacement) * height * 2)
    direct_traj.append([traj_x , height])
    # while traj_start + distance * (5 if sign > 0 else 4) < end:
    # while traj_start + (10 if end_offset == None else end_offset) < carrot_length:
    for i in range(num_cuts):        
        direct_timestep += 10
        direct_traj.append([traj_x, 0])
        direct_timestep += 10
        direct_traj.append([traj_x, height])
        
        traj_x += distance
        cur_height = closestValueMaxHeight(carrot, traj_x)
        height = cur_height + add_height
        cut_displacements.append((sign-displacement) * height * 2)
        direct_timestep += 3
        direct_traj.append([traj_x, height])
    direct_timestep += 10
    direct_traj.append([traj_x, 0])
    # print(len(traj), len(cut_displacements))
    
    displacement_idx = 0
    for i in direct_traj:
        i[0] -= sign*distance*abs(orientation)/10
        if i[1]==0:
            cut_number += 1
            if sign < 0:
                i[0] += cut_displacements[displacement_idx]
            else:
                i[0] -= cut_displacements[displacement_idx]
            displacement_idx += 1
            
    traj, traj_t = divideTraj(direct_traj, traj_step)
    
    return traj, traj_t, direct_traj, cut_number

def carrotRandomizer(length = 200, flip = False):
    global base_shape
    base_shape_np = np.array(base_shape)
    
    # Change length
    base_shape_np = base_shape_np * length / base_shape_np.max(axis=0)[0]
    
    # Randomize shape
    for ridge in base_shape_np:
        ridge[1] += rand() * 5
        
    carrot = base_shape_np.tolist()
    carrot.append(carrot[0])
    
    rotated_carrot = []
    rot = np.deg2rad( 5 )
    for i in carrot:
        rotated_carrot.append([i[0]*np.cos(rot)+i[1]*np.sin(rot), -i[0]*np.sin(rot)+i[1]*np.cos(rot)])
        
    chaikin = Chaikin(rotated_carrot)
    smoothed_carrot = chaikin.Smooth_by_Chaikin(number_of_refinements = 5)
    if flip:
        smoothed_carrot = np.array(smoothed_carrot)
        smoothed_carrot[:,0] *= -1
        x_min = abs(np.min(smoothed_carrot[:,0]))
        smoothed_carrot[:,0] += x_min
    
    return smoothed_carrot

def captionGenerator(distance, orientation):
    cap = 'Cut the carrot '
    if orientation == 0:
        cap += 'straight '
    else:
        cap += (str(orientation) + ' degrees ')
    cap += 'with a distance of '
    cap += str(distance)
    return cap

def generateDataset(dataset_size = 4e4, 
                    cur_num = 0, 
                    distance = 10,
                    orientation = 0,
                    add_height = 5,
                    traj_step = 10,
                    min_offset = 10,
                    flipped = False,
                    save_fig = True):  
    
    traj_name       = 'distance_'+str(distance)+\
                      '_orientation_'+str(orientation)+\
                      '_height_'+str(add_height)+\
                      '_step_'+str(traj_step)+'/'
    traj_dir        = join(ROOT_DIR, traj_name, 'trajectories/')
    img_dir         = join(ROOT_DIR, traj_name, 'image/')
    img_50_dir      = join(img_dir, '50x50/')
    img_500_dir     = join(img_dir, '500x500/')
    img_500_traj_dir= join(img_dir, '500x500_traj/')
    
    if not isdir(traj_dir):
        makedirs(traj_dir)
    if not isdir(img_dir):
        makedirs(img_dir)
    if not isdir(img_50_dir):
        makedirs(img_50_dir)
    if not isdir(img_500_dir):
        makedirs(img_500_dir)
    if not isdir(img_500_traj_dir):
        makedirs(img_500_traj_dir)
    
    # for i in range(dataset_size):
    im = np.asarray(Image.open(join(TEX_ROOT_DIR, "carrot_texture_long.jpg")).resize((randint(200,500), randint(50,100)), Image.LANCZOS))
    seed = cur_num
    rng = RandomState(seed)
    length_modifier = rng.randint(0, 500)/10
    length = 200 - length_modifier
    carrot = carrotRandomizer(length, flip = flipped)
    
    
    traj, traj_t, direct_traj, cut_number = trajGenerator(
                                                distance = distance,
                                                add_height = add_height,
                                                orientation = orientation,
                                                carrot = carrot,
                                                traj_step = traj_step,
                                                min_offset = min_offset,
                                                flip = flipped)
    
    # Generate DMP
    y_des = np.array(traj).T

    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=len(y_des.shape), n_bfs=200, ay=np.ones(2) * 25.0)
    y_track = []
    dy_track = []
    ddy_track = []

    dmp.imitate_path(y_des=y_des, plot=False)
    w_dmps = dmp.w.T
    y_track, dy_track, ddy_track = dmp.rollout(tau = 0.01)

    # Lower cuts
    y_track_sorted = y_track[y_track[:,1].argsort()]
    int_y_track_sorted = np.append([np.int32(y_track_sorted[:,0])],[y_track_sorted[:,1]], axis=0).T
    lowest_x = []
    lowest_y = []
    for i in range(len(int_y_track_sorted)):
        if len(lowest_y) == cut_number:
            break
        if int_y_track_sorted[i][0] not in lowest_x:
            lowest_x.append(int_y_track_sorted[i][0]-1)
            lowest_x.append(int_y_track_sorted[i][0])
            lowest_x.append(int_y_track_sorted[i][0]+1)
            lowest_y.append(int_y_track_sorted[i][1])
        else:
            continue

    # y_track_zeroed = np.copy(y_track)
    # lowered = 0
    # for i in range(len(y_track_zeroed)):
    #     if lowered == cut_number:
    #         break
    #     if y_track_zeroed[i][1] in lowest_y:
    #         y_track_zeroed[i][1] = 0
    #         lowered += 1

    # lowest_y_sorted = np.sort(lowest_y)
    # lowering_offset = lowest_y_sorted[-1]
    
    if save_fig:
        traj_x = [tr[0] for tr in traj]
        traj_y = [tr[1] for tr in traj]
        
        direct_traj_x = [tr[0] for tr in direct_traj]
        direct_traj_y = [tr[1] for tr in direct_traj]
        # traj_t = [tr[2] for tr in traj]
        
        fig = plt.figure(figsize=(10, 10))
        # fig = figure.Figure(figsize=(10, 10))
        p = Polygon(np.array(carrot), closed = False)
        p.set_color('none')
        ax = plt.gca()
        # ax = fig.subplots(1)
        ax.add_patch(p)
        ax.imshow(im, clip_path = p, clip_on=True)
        
        # ax.plot(traj_x, traj_y, linewidth=0.75, color='c')
        # ax.scatter(traj_x, traj_y, s=10, c='c')
        
        plt.xlim(0-length_modifier/2, 200-length_modifier/2)
        plt.ylim(-75, 125)
        plt.tick_params(
            bottom=False,     
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False)
        ax.set_facecolor('black')
        
        dpi_size = 0.33149999999999996
        plt.savefig(join(img_50_dir, 'Image_'+str(cur_num)+'.jpg'), dpi=dpi_size * 50, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(join(img_500_dir, 'Image_'+str(cur_num)+'.jpg'), dpi=dpi_size * 500, bbox_inches='tight', pad_inches=0.0)
        
        # Plot real trajectory
        ax.plot(direct_traj_x, direct_traj_y, linewidth=1, color='g')
        ax.scatter(traj_x, traj_y, s=3, c='g')
        
        # Plot DMP
        # ax.plot(y_track_zeroed[:, 0], y_track_zeroed[:, 1], linewidth=1, color='b', ls='solid')
        ax.plot(y_track[:, 0], y_track[:, 1], linewidth=2, color='w', ls=':')
        
        plt.savefig(join(img_500_traj_dir, 'Image_'+str(cur_num)+'.jpg'), dpi=66.3, bbox_inches='tight', pad_inches=0.0)
        
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close(fig)
    
    # if len(traj) > pad_to:
    #     print(">> Not enough padding ::", cur_num)
    #     return
    
    # while len(traj) < pad_to:
    #     traj.append([traj[-1][0], traj[-1][1], traj[-1][2] + pad_step])
    
    traj_str = ""
    direct_traj_str = ""
    dmp_traj_str = ""
    # zeroed_dmp_traj_str = ""
    dmp_w_str = ""
    dmp_y0_str = ""
    dmp_goal_str = ""

    for i in traj:
        traj_str += '['
        traj_str += str(i[0])
        traj_str += ', '
        traj_str += str(i[1])
        traj_str += '], '
    traj_str = traj_str[:-2]

    for i in direct_traj:
        direct_traj_str += '['
        direct_traj_str += str(i[0])
        direct_traj_str += ', '
        direct_traj_str += str(i[1])
        direct_traj_str += '], '
    direct_traj_str = direct_traj_str[:-2]

    for i in y_track:
        dmp_traj_str += '['
        dmp_traj_str += str(i[0])
        dmp_traj_str += ', '
        dmp_traj_str += str(i[1])
        dmp_traj_str += '], '
    dmp_traj_str = dmp_traj_str[:-2]

    # for i in y_track_zeroed:
    #     zeroed_dmp_traj_str += '['
    #     zeroed_dmp_traj_str += str(i[0])
    #     zeroed_dmp_traj_str += ', '
    #     zeroed_dmp_traj_str += str(i[1])
    #     zeroed_dmp_traj_str += '], '
    # zeroed_dmp_traj_str = zeroed_dmp_traj_str[:-2]

    for i in w_dmps:
        dmp_w_str += '['
        dmp_w_str += str(i[0])
        dmp_w_str += ', '
        dmp_w_str += str(i[1])
        dmp_w_str += '], '
    dmp_w_str = dmp_w_str[:-2]
    
    dmp_y0_str += '['
    dmp_y0_str += str(dmp.y0[0])
    dmp_y0_str += ', '
    dmp_y0_str += str(dmp.y0[1])
    dmp_y0_str += ']'
    
    dmp_goal_str += '['
    dmp_goal_str += str(dmp.goal[0])
    dmp_goal_str += ', '
    dmp_goal_str += str(dmp.goal[1])
    dmp_goal_str += ']'

    with open(join(traj_dir, 'Image_'+str(cur_num)+'.json'), "w") as f:
        to_json = {
            "Image number"      : cur_num,
            "Carrot length"     : length,
            "Cut number"        : cut_number,
            "Additional height" : add_height,
            "Cut distance"      : distance,
            "Orientation"       : orientation,
            "Trajectory step"   : traj_step,
            "Trajectory"        : traj_str,
            # "Direct trajectory" : direct_traj_str,
            "DMP trajectory"    : dmp_traj_str,
            # "Lowered DMP traj"  : zeroed_dmp_traj_str,
            "DMP weight"        : dmp_w_str,
            "DMP tau"           : dmp.timesteps,
            "DMP dt"            : dmp.dt,
            "DMP y0"            : dmp_y0_str,
            "DMP goal"          : dmp_goal_str,
            "Caption"           : captionGenerator(distance, orientation)
            }
        json.dump(to_json, f)
        to_json.clear()
        
#%% Normal plot
# x = [x[0] for x in base_shape]
# y = [y[1] for y in base_shape]

# plt.figure(figsize=(5,5))
# p = Polygon(np.array(base_shape), closed = False)
# ax = plt.gca()
# ax.add_patch(p)
# plt.xlim(0, 21)
# plt.ylim(0, 21)
# plt.show()

#%% Test trajectory generation
im = np.asarray(Image.open(join(TEX_ROOT_DIR, "carrot_texture_long.jpg")).resize((randint(200,500), randint(50,100)), Image.LANCZOS))
seed = None
rng = RandomState(seed)
length_modifier = rng.randint(0, 500)/10
length = 200 - length_modifier
flipped = 1
carrot = carrotRandomizer(length, flip = flipped)

traj, traj_t, direct_traj, cut_number = trajGenerator(
    distance = 10,
    add_height = 25,
    orientation = 0,
    carrot = carrot,
    traj_step = 5,
    min_offset = 10,
    flip = flipped)
print("Cuts =", cut_number)

traj_x = [tr[0] for tr in traj]
traj_y = [tr[1] for tr in traj]
# traj_t = [tr[2] for tr in traj]

direct_traj_x = [tr[0] for tr in direct_traj]
direct_traj_y = [tr[1] for tr in direct_traj]
# direct_traj_t = [tr[2] for tr in direct_traj]

y_des = np.array(traj).T

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=len(y_des.shape), n_bfs=300,  ay=np.ones(2) * 25.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=False)
w_dmps = dmp.w
y_track, dy_track, ddy_track = dmp.rollout(tau = 0.01)

# Lower cuts
y_track_sorted = y_track[y_track[:,1].argsort()]
int_y_track_sorted = np.append([np.int32(y_track_sorted[:,0])],[y_track_sorted[:,1]], axis=0).T
lowest_x = []
lowest_y = []
for i in range(len(int_y_track_sorted)):
    if len(lowest_y) == cut_number:
        break
    if int_y_track_sorted[i][0] not in lowest_x:
        lowest_x.append(int_y_track_sorted[i][0]-1)
        lowest_x.append(int_y_track_sorted[i][0])
        lowest_x.append(int_y_track_sorted[i][0]+1)
        lowest_y.append(int_y_track_sorted[i][1])
    else:
        continue

# y_track_zeroed = np.copy(y_track)
# lowered = 0
# for i in range(len(y_track_zeroed)):
#     if lowered == cut_number:
#         break
#     if y_track_zeroed[i][1] in lowest_y:
#         y_track_zeroed[i][1] = 0
#         lowered += 1

lowest_y_sorted = np.sort(lowest_y)
lowering_offset = lowest_y_sorted[-1]

# plt.figure(figsize=(10, 10))
fig = figure.Figure(figsize=(10, 10))
p = Polygon(np.array(carrot), closed = False)
p.set_color('none')
ax = plt.gca()
ax.add_patch(p)
ax.imshow(im, clip_path = p, clip_on=True)

# Plot real trajectory
ax.plot(direct_traj_x, direct_traj_y, linewidth=1, color='g')
ax.scatter(traj_x, traj_y, s=3, c='g')

# Plot DMP
# ax.plot(y_track_zeroed[:, 0], y_track_zeroed[:, 1], linewidth=1, color='b', ls='solid')
ax.plot(y_track[:, 0], y_track[:, 1], linewidth=2, color='w', ls=':')

plt.xlim(0-length_modifier/2, 200-length_modifier/2)
plt.ylim(-75, 125)
plt.tick_params(
    bottom=False,     
    top=False,
    left=False,
    labelbottom=False,
    labelleft=False)
ax.set_facecolor('black')
dpi_size = 0.33149999999999996
plt.savefig(join(TEX_ROOT_DIR, 'carrot50px.jpg'), dpi=dpi_size * 50, bbox_inches='tight', pad_inches=0.0)
plt.savefig(join(TEX_ROOT_DIR, 'carrot500px.jpg'), dpi=dpi_size * 500, bbox_inches='tight', pad_inches=0.0)
plt.show()
plt.axis("equal")

plt.cla()
plt.clf()
plt.close('all')

traj_str = ""
direct_traj_str = ""

for i in traj:
    traj_str += str(i)
    traj_str += ', '
traj_str = traj_str[:-2]

#%% Generate dataset
from os.path import join, isdir
from os import makedirs
import json
from multiprocessing import Process
from tqdm import tqdm

ROOT_DIR        = '/home/edgar/rllab/scripts/dmp/data/img_json/'

dataset_size    = int(4e4)
start_num       = 0
distance        = 10
add_height      = 25
orientation     = 0
traj_step       = 5
min_offset      = 10
flipped         = True
# pad_to          = 150
# pad_step        = 0.01
save_fig        = True

cur_num         = start_num
pbar = tqdm(total = dataset_size, ncols = 65)
pbar.update(start_num)
for i in range(dataset_size - cur_num):
    # generateDataset(dataset_size = dataset_size, start_num = cur_num, distance = distance, orientation = orientation, add_height = add_height, traj_step = traj_step, pad_to = pad_to, pad_step = pad_step, save_fig = save_fig)
    
    # Using multiprocessing.Process to handle matplotlib memory leakage
    p = Process(target = generateDataset, args = (dataset_size, cur_num, distance,  orientation, add_height, traj_step, min_offset, flipped, save_fig))
    p.start()
    p.join()
    pbar.update(1)    
    if cur_num + 1 == dataset_size:
        break
    else:
        cur_num += 1
pbar.close()
