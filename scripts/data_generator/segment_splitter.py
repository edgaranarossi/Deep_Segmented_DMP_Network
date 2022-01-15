import numpy as np
from numpy.random import randint

def split_traj_into_segment(traj, num_segment):
    traj = np.array(traj)
    length_distribution = np.ones(num_segment) * (traj.shape[0] // num_segment)
    rmndr = traj.shape[0] - (num_segment * (traj.shape[0] // num_segment))
    to_add = np.zeros(num_segment)
    to_add[:rmndr] = 1
    length_distribution += to_add
    np.random.shuffle(length_distribution)
    idx_list = get_index(traj, length_distribution)
    print(idx_list)
    segments = []
    for i in range(1, len(idx_list)):
        segments.append(traj[idx_list[i - 1]:idx_list[i], :])
    return segments

def get_index(traj, length_distribution):
    idx_list = [0]
    for i in range(len(length_distribution)):
        idx_list.append(int(idx_list[-1] + length_distribution[i]))
    
    for i in range(1, len(idx_list) - 1):
        if traj[idx_list[i - 1], 0] == traj[idx_list[i] - 1, 0] and traj[idx_list[i - 1], 1] == traj[idx_list[i] - 1, 1]:
            # print(traj[idx_list[i - 1], :], traj[idx_list[i] - 1, :])
            idx_list[i] -= 1
            # print(traj[idx_list[i - 1], :], traj[idx_list[i] - 1, :], '\n')
    return idx_list