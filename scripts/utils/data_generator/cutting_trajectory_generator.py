import numpy as np
from numpy import cos,sin,deg2rad,pi
from matplotlib import pyplot as plt

def plot_trajectory(traj):
    traj_x = [tr[0] for tr in traj['subdivided']]
    traj_y = [tr[1] for tr in traj['subdivided']]
    direct_traj_x = [tr[0] for tr in traj['direct']]
    direct_traj_y = [tr[1] for tr in traj['direct']]
    
    plt.plot(direct_traj_x, direct_traj_y, linewidth=0.5, color='g')
    plt.scatter(traj_x, traj_y, s=1, c='g')

def generate_cutting_trajectory(obj, dist_btw_cut, lift_height, orientation, traj_length, margin):
    """
    Generate cutting motion trajectory from a given object shape outline
    """

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
    
    obj_bottommost = np.array(obj).min(axis = 0)[1]
    obj_leftmost = np.array(obj).min(axis = 0)[0]
    obj_rightmost = np.array(obj).max(axis = 0)[0]
    num_cuts = int((obj_rightmost - 2 * (margin if margin != None else 0)) / dist_btw_cut)
    tip_leftover = obj_rightmost - (dist_btw_cut * num_cuts) - (2 * (margin if margin != None else 0))
    # obj_start = obj[0][0]
    
    traj_x = obj_leftmost + (margin if margin != None else 0) + (tip_leftover / 2)
    # traj_start += displacement * cur_height
    # if sign > 0: traj_start += dist_btw_cut * 2
    roof = []
    for i in range(obj.shape[0]):
        if obj[i, 0] > obj_leftmost + (margin / 2) and \
        obj[i, 0] < obj_rightmost - (margin / 2) and \
        obj[i, 1] > obj_bottommost + margin :
            # plt.scatter(obj[i, 0], obj[i, 1], c = 'r')
            roof.append([obj[i, 0], obj[i, 1]])
    roof = np.array(roof)
    roof = roof[roof[:,0].argsort()]
    # print(roof)
    cur_height = closestValueMaxHeight(roof, traj_x)
    
    height = cur_height + lift_height
    cut_displacements.append((sign-displacement) * height * 2)
    direct_traj.append([traj_x , height])
    # while traj_start + dist_btw_cut * (5 if sign > 0 else 4) < end:
    # while traj_start + (10 if end_offset == None else end_offset) < obj_length:
    for i in range(num_cuts):        
        direct_timestep += 10
        direct_traj.append([traj_x, 0])
        direct_timestep += 10
        direct_traj.append([traj_x, height])
        
        traj_x += dist_btw_cut
        cur_height = closestValueMaxHeight(roof, traj_x)
        height = cur_height + lift_height
        cut_displacements.append((sign-displacement) * height * 2)
        direct_timestep += 3
        direct_traj.append([traj_x, height])
    direct_timestep += 10
    direct_traj.append([traj_x, 0])
    # print(len(traj), len(cut_displacements))
    
    displacement_idx = 0
    for i in direct_traj:
        i[0] -= sign*dist_btw_cut*abs(orientation)/10
        if i[1]==0:
            cut_number += 1
            if sign < 0:
                i[0] += cut_displacements[displacement_idx]
            else:
                i[0] -= cut_displacements[displacement_idx]
            displacement_idx += 1
            
    subdiv_traj = subDivideTraj(direct_traj, traj_length, cut_number)
    traj = {'direct':direct_traj, 'subdivided':subdiv_traj, 'num_cut':cut_number}
    return traj

def closestValueMaxHeight(roof, traj_x):
    idx_further = np.where((roof[:, 0] >= traj_x) == True)[0][0]
    max_height = roof[idx_further, 1] if np.abs(roof[idx_further, 0] - traj_x) < np.abs(roof[idx_further - 1, 0] - traj_x) else roof[idx_further - 1, 1]
    return max_height

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

def euclideanDistance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculateAngle(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return np.arctan2(y, x)

def subDivideTraj(traj, traj_length, cut_number):
    # print(len(traj))
    total_length = 0
    for idx_p, p in enumerate(traj[:-1]):
        total_length += euclideanDistance(p, traj[idx_p+1])
    step_size = total_length / traj_length
    # print(total_length, step_size)
    
    subdivided_traj = []
    edge_distances = []
    
    for idx_p, p in enumerate(traj[:-1]):
        edge_distance = euclideanDistance(p, traj[idx_p+1])
        edge_distances.append([edge_distance, idx_p])
    # edge_distances = sorted(edge_distances, key=lambda i: i[0])
    sorted_edge_distance = np.array(edge_distances)[::-1] #[:((cut_number * 2) - 1)]
    edge_cuts = np.zeros_like(sorted_edge_distance[:,0], dtype = 'int32')
    remaining_points = traj_length - len(traj)
    # need = traj_length - len(traj)
    if remaining_points < 0:
        raise ValueError('Direct Traj ' + str(len(traj)) + ' longer than total length')
    elif remaining_points == 0:
        subdivided_traj = traj
    else:
        while np.sum(edge_cuts) < remaining_points:
            divided_length = sorted_edge_distance[:,0] / (edge_cuts + 1)
            max_edge_length = np.max(divided_length)
            for idx_dist, dist in enumerate(divided_length):
                if dist == max_edge_length:
                    edge_cuts[idx_dist] += 1
                    break
                
        for idx_p, p in enumerate(sorted_edge_distance):
            edge_distance = p[0]
            edge_idx = int(p[1])
            cur_point = traj[edge_idx]
            next_point = traj[edge_idx + 1]
            edge_angle = calculateAngle(cur_point, next_point)
            edge_cut = edge_cuts[idx_p] + 1
            edge_step_size = edge_distance / edge_cut
            subdivided_edge = []
            for i in range(edge_cut):
                subdivide_point = [cur_point[0] + i * edge_step_size * np.cos(edge_angle),
                                   cur_point[1] + i * edge_step_size * np.sin(edge_angle)]
                subdivided_edge.append(subdivide_point)
                # subdivided_traj.append([subdivide_point, edge_idx])
            subdivided_traj.append([subdivided_edge, edge_idx])
            # subdivided_traj.append(next_point)
        subdivided_traj.append([[traj[-1]], np.max(sorted_edge_distance[:,1])+1])
        subdivided_traj = sorted(subdivided_traj, key=lambda i: i[1])
        
    subdivided_traj = [i for j in subdivided_traj for i in j[0]]
    # print(subdivided_traj)
            
    subdivided_traj_np = np.array(subdivided_traj)
    # print(len(traj), subdivided_traj_np.shape)
    # plt.scatter(subdivided_traj_np[:,0], subdivided_traj_np[:,1])
    return subdivided_traj