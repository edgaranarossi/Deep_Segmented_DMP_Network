#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 07:00:16 2022

@author: edgar
"""

import numpy as np
from pydmps import DMPs_discrete
from matplotlib import pyplot as plt

def generate_jagged_reconstruction(traj, bfs):
    """
    Generate jagged reconstruction of a trajectory using DMPs.

    Parameters:
    traj (list): List of trajectory points.
    bfs (list): List of basis functions.

    Returns:
    None
    """
    y_des = np.array(traj)
    ys = []
    for i in range(len(traj) - 1):
        seg = np.array(traj[i:i+2])
        dmp = DMPs_discrete(n_dmps=2, n_bfs=2, ay=np.ones(2) * 3.4, dt=0.01 * (len(traj) - 1))
        dmp.imitate_path(seg.T)
        y, _, _ = dmp.rollout()
        ys.append(y)
    y_full = ys[0]
    for i in ys[1:]:
        y_full = np.append(y_full, i, axis=0)

    ys = []
    rmses = []

    fig = plt.figure(figsize=(16, 14))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)

    ax1.tick_params(bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
    ax1.set_xlim(-0.05, 2.75)
    ax1.set_ylim(-0.1, 1.0)
    ax3.set_ylim(0, 1600)
    ax1.plot(y_full[:, 0], y_full[:, 1], c='black', lw=15, zorder=-1)
    w_intervals = []

    for i, bf in enumerate(bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bf, ay=np.ones(2) * 18, dt=0.01)
        dmp.imitate_path(y_des.T)
        w_intervals.append(dmp.w.max() - dmp.w.min())
        y, _, _ = dmp.rollout()
        ax1.scatter(y[:, 0], y[:, 1], lw=0.01)
        ys.append(y)
        rmse = np.sqrt(((y_full - y) ** 2).mean())
        rmses.append(rmse)
        ax2.scatter(bf, rmse, s=100)
        ax3.scatter(bf, w_intervals[-1], s=100)
    w_intervals = np.array(w_intervals)

    rmses = np.array(rmses)
    ax2.plot(bfs, rmses, zorder=-1, c='red')
    ax3.plot(bfs, w_intervals, zorder=-1, c='red')
    ax2.set_ylim(0, 0.15)
    ax2.tick_params(bottom=False, top=False, left=False, labelbottom=True, labelleft=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax2.set_ylabel('Reconstruction error')
    ax2.set_xlabel('# of basis functions')
    ax3.set_ylabel('w interval')
    ax3.set_xlabel('# of basis functions')
    ax1.legend(['Original'] + bfs)

    plt.savefig('/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/fig_2.pdf')

if __name__ == '__main__':
    traj = [[0.0, 0.0],
            [0.5, 0.7],
            [0.9, 0.1],
            [1.2, 0.9],
            [1.6, 0.2],
            [2.0, 0.8]]
    bfs = [10, 25, 50] + [i for i in range(100, 501, 100)]
    generate_jagged_reconstruction(traj, bfs)