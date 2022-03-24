import pydmps
from pydmps.dmp_discrete import DMPs_discrete
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand, randint
from .utils import smooth
from .utils import rotateCoordNp

def generate_dmps(trajs, n_bf, ay, dt, segmented):
    if not segmented:
        y_des = np.array(trajs).T
        dof = y_des.shape[0]
        dmps = pydmps.dmp_discrete.DMPs_discrete(n_dmps = dof, n_bfs=n_bf, ay = np.ones(dof) * ay, dt = dt)
        dmps.imitate_path(y_des = y_des, plot = False)
    elif segmented:
        dmps = []
        for traj in trajs:
            y_des = np.array(traj).T
            # print(y_des)
            dof = y_des.shape[0]
            dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps = dof, n_bfs=n_bf, ay = np.ones(dof) * ay, dt = dt)
            dmp.imitate_path(y_des = y_des, plot = False)
            dmps.append(dmp)
    return dmps

def plot_dmp_trajectory(dmps, segmented, plot = True, external_force = None):
    y_tracks = []
    dy_tracks = []
    ddy_tracks = []
    
    if not segmented:
        y_tracks, dy_tracks, ddy_tracks = dmps.rollout()
    elif segmented:
        last_pos = dmps[0].y0
        for dmp in dmps:
            dmp.y0 = last_pos
            y_track, dy_track, ddy_track = dmp.rollout(external_force =  external_force)
            last_pos = y_track[-1, :]
            y_tracks.append(y_track)
            dy_tracks.append(dy_track)
            ddy_tracks.append(ddy_track)
        y_tracks, dy_tracks, ddy_tracks = recombine_trajs(y_tracks, dy_tracks, ddy_tracks)
    if plot:
        plt.plot(y_tracks[:, 0], y_tracks[:, 1], linewidth=2, color='r')
        plt.scatter(y_tracks[:, 0], y_tracks[:, 1], linewidth=2, color='r', ls=':')
        plt.show()
    return y_tracks, dy_tracks, ddy_tracks

def recombine_trajs(y, dy, ddy):
    combined_y = y[0]
    combined_dy = dy[0]
    combined_ddy = ddy[0]
    for i in range(1, len(y)):
        combined_y = np.append(combined_y, y[i], axis=0)
        combined_dy = np.append(combined_dy, dy[i], axis=0)
        combined_ddy = np.append(combined_ddy, ddy[i], axis=0)
    return combined_y, combined_dy, combined_ddy

def check_w_min_max(dmps, segmented):
    if not segmented:
        w = dmps.w
        min_val = w.min()
        max_val = w.max()
    elif segmented:
        min_val = 1e16
        max_val = 0
        for dmp in dmps:
            w = dmp.w
            min_val = w.min() if w.min() < min_val else min_val
            max_val = w.max() if w.max() > max_val else max_val
    return [min_val, max_val]

def plot_dmp_segment(dmps, idx):
    y_tracks, _, _ = dmps[idx].rollout()
    plt.plot(y_tracks[:, 0], y_tracks[:, 1], linewidth=2, color='r', ls=':')
    print(check_w_min_max(dmps[idx], segmented = False))

def trajpoints2dmp(traj, dmp_bf, dmp_ay, dmp_dt, num_points = [2], w_limit = [-1e+9, 1e+9]):
    Xs = []
    Ys = []
    trajs = []
    dmps = []
    for i in range(len(traj) - 1):
        for n in num_points:
            if i < len(traj) - (n - 1):
                tr = traj[i:i+n,:]
                dmp = generate_dmps(tr, dmp_bf, dmp_ay, dmp_dt, segmented = False)
                if check_w_min_max(dmp, segmented = False)[0] > w_limit[0] and \
                   check_w_min_max(dmp, segmented = False)[1] < w_limit[1]:
                    dmp_param = dmp.y0
                    dmp_param = np.append(dmp_param, dmp.goal)
                    dmp_param = np.append(dmp_param, dmp.w)
                    y_track, _, _ = dmp.rollout(tau=1)
                    Xs.append(dmp_param)
                    Ys.append(y_track.reshape(-1))
                    trajs.append(tr)
                    dmps.append(dmp)
                else:
                    continue
    return Xs, Ys, trajs, dmp

def generate_random_traj(max_points):
    num_points = randint(2, max_points + 1)
    smoothing_degree = randint(2, 6)
    traj = rand(num_points, 2) * 50
    traj = smooth(traj.tolist(), smoothing_degree)
    traj = np.array(traj)
    return traj

def generate_random_dmp(max_points = 6,
                        n_bfs = 15,
                        ay = 15,
                        dt = 0.01,
                        w_limit = [-1e+9, 1e+9],
                        plot_traj = False,
                        plot_dmp = False):

    traj = generate_random_traj(max_points)
    dmp = generate_dmps(traj, n_bfs, ay, dt, segmented = False)

    while check_w_min_max(dmp, segmented = False)[0] <= w_limit[0] and \
          check_w_min_max(dmp, segmented = False)[1] >= w_limit[1]:
        traj = generate_random_traj(max_points)
        dmp = generate_dmps(traj, n_bfs, ay, dt, segmented = False)

    dmp_param = dmp.y0
    dmp_param = np.append(dmp_param, dmp.goal)
    dmp_param = np.append(dmp_param, dmp.w)
    y_track, _, _ = dmp.rollout(tau=1)
    dmp_traj = y_track.reshape(-1)

    # if plot_traj or plot_dmp:
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    if plot_traj:
        plt.plot(traj[:,0], traj[:,1])
    if plot_dmp:
        y_track, _, _ = dmp.rollout()
        plt.plot(y_track[:,0], y_track[:,1])
    if plot_traj or plot_dmp: plt.show()

    return dmp_param, dmp_traj, traj, dmp

def generate_random_rotated_curves_dmp(dmp_bf, dmp_ay, dmp_dt, scale, random_pos, w_limit, plot_traj):


    within_limit = False

    while not within_limit:
        rot = randint(10, 16)
        rotations = [i for i in range(0, 360, rot)]

        length = scale/2 + (scale/2 * rand())
        mid = rand() * length
        height = rand() * length * 1
        curve_points = [[0, 0], [mid, height], [length, 0]]
        # curve = np.array(smooth(curve_points, 3))
        curve = np.array(curve_points)
        line = np.array([[0, 0], [length, 0]])
        # plt.plot(curve[:, 0], curve[:, 1])
        # plt.axis('equal')

        center_curve = curve.min(axis = 0) + ((curve.max(axis = 0) - curve.min(axis = 0))/2)
        center_line = line.min(axis = 0) + ((line.max(axis = 0) - line.min(axis = 0))/2)
        trajs = []
        for r in rotations:
            rotated_curve = rotateCoordNp(curve, center_curve, r)
            if random_pos: rotated_curve[:,0] = rotated_curve[:,0] + (rand() * scale * 1.5)
            if random_pos: rotated_curve[:,1] = rotated_curve[:,1] + (rand() * scale * 1.5)
            trajs.append(rotated_curve)
            rotated_line = rotateCoordNp(line, center_line, r)
            if random_pos: rotated_line[:,0] = rotated_line[:,0] + (rand() * scale * 1.5)
            if random_pos: rotated_line[:,1] = rotated_line[:,1] + (rand() * scale * 1.5)
            trajs.append(rotated_line)
            # if plot_traj:
                # plt.plot(rotated_curve[:, 0], rotated_curve[:, 1])
                # plt.plot(rotated_line[:, 0], rotated_line[:, 1])
                
        dmps = generate_dmps(trajs, dmp_bf, dmp_ay, dmp_dt, segmented = True)

        dmp_params = []
        dmp_trajs = []

        max_w = -1e6
        min_w = 1e6
        for dmp in dmps:
            if dmp.w.max() > max_w: max_w = dmp.w.max()
            if dmp.w.min() < min_w: min_w = dmp.w.min()
            if min_w < w_limit[0] and max_w > w_limit[1]: break
            dmp_param = dmp.y0
            dmp_param = np.append(dmp_param, dmp.goal)
            dmp_param = np.append(dmp_param, dmp.w)
            dmp_params.append(dmp_param)
            y_track, _, _ = dmp.rollout()
            dmp_trajs.append(y_track.reshape(-1))
            if plot_traj:
                plt.plot(y_track[:, 0], y_track[:, 1])
        
        if min_w > w_limit[0] and max_w < w_limit[1]: 
            within_limit = True
            # print('success')
        # else: 
            # print(max_w)
            # plt.plot(y_track[:, 0], y_track[:, 1])
            # plt.show()

    return dmp_params, dmp_trajs, trajs, dmps

    