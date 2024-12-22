#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test DMP Parameter Script

This script contains functions to test the DMP parameters for the Deep Segmented DMP Network.
"""

from pydmps import DMPs_discrete
import numpy as np
from matplotlib import pyplot as plt

if __name__=='__main__':
    traj = [[0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0]]
    traj = np.array(traj)
    # traj = generator.map.gripper_history[:,:2]
    dmp = DMPs_discrete(n_dmps = 2, n_bfs = 100, ay = np.ones(2) * 200, dt = 0.001)
    dmp.imitate_path(traj.T)
    y, _, _ = dmp.rollout()
    plt.scatter(y[:, 0], y[:, 1])
    plt.scatter([traj[0, 0], traj[-1, 0]], [traj[0, 1], traj[-1, 1]], c = 'r')