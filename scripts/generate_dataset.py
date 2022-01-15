from utils.data_generator import generate_carrot, generate_random_dmp, generate_cutting_trajectory, split_traj_into_segment, generate_dmps, trajpoints2dmp
from utils.dataset_importer import Scale
import numpy as np
import pickle as pkl
from os.path import join

def generate_random_traj(plot_traj,
                         plot_dmp,
                         max_points,
                         dmp_bf,
                         dmp_ay,
                         dmp_dt,
                         w_limit):
    global X, Y, original_trajs
    dmp_params, dmp_traj, or_traj, _ = generate_random_dmp(plot_traj = plot_traj,
                                                           plot_dmp = plot_dmp,
                                                           max_points = max_points,
                                                           n_bfs = dmp_bf,
                                                           ay = dmp_ay,
                                                           dt = dmp_dt,
                                                           w_limit = w_limit)
    X.append(dmp_params)
    Y.append(dmp_traj)
    original_trajs.append(or_traj)

def generate_carrot_points(tip,
                           cut_dist,
                           cut_lift_height,
                           cut_orientation,
                           cut_traj_length,
                           cut_margin,
                           dmp_bf,
                           dmp_ay,
                           dmp_dt,
                           num_points,
                           w_limit):
    global X, Y, original_trajs
    carrot = generate_carrot(tip = tip)
    traj = generate_cutting_trajectory( obj = carrot,
                                        dist_btw_cut = cut_dist,
                                        lift_height = cut_lift_height,
                                        orientation = cut_orientation,
                                        traj_length = cut_traj_length,
                                        margin = cut_margin
                                      )
    or_traj = np.array(traj['direct'])
    xs, ys, or_trajs, _ = trajpoints2dmp(or_traj,
                                          dmp_bf = dmp_bf,
                                          dmp_ay = dmp_ay,
                                          dmp_dt = dmp_dt,
                                          num_points = num_points,
                                          w_limit = w_limit)
    X = X + xs
    Y = Y + ys
    original_trajs = original_trajs + or_trajs

X = []
Y = []
original_trajs = []

num_dataset = int(3e6)
print_every = 10000
checkpoints = [i for i in range(print_every, num_dataset, print_every)]
w_limit = [-1e+8, 1e+8]
dmp_bf = 15
dmp_ay = 15
dmp_dt = 0.025
num_points = [2]
max_points = 3
plot_traj = 0
cur_num = len(X)
while len(X) < num_dataset:
    generate_carrot_points(tip = 'right',
                        cut_dist = 5,
                        cut_lift_height = 10,
                        cut_orientation = 0,
                        cut_traj_length = 100,
                        cut_margin = 5,
                        dmp_bf = dmp_bf,
                        dmp_ay = dmp_ay,
                        dmp_dt = dmp_dt,
                        num_points = num_points,
                        w_limit = w_limit)
    
    created = len(X) - cur_num
    
    for i in range(created * 5):
        generate_random_traj(plot_traj = plot_traj,
                              plot_dmp = plot_traj,
                              max_points = max_points,
                              dmp_bf = dmp_bf,
                              dmp_ay = dmp_ay,
                              dmp_dt = dmp_dt,
                              w_limit = w_limit)
    
    cur_num = len(X)
    
    while len(checkpoints) > 0 and checkpoints[0] <= len(X):
        print('Generated', int(checkpoints[0]), '/', num_dataset)
        # print('Generated', len(X), '/', num_dataset)
        del checkpoints[0]

print('Generated', num_dataset, '/', num_dataset)

X_np = np.array(X)[:num_dataset]
Y_np = np.array(Y)[:num_dataset]
original_trajs = original_trajs[:num_dataset]
scale = Scale([0.0, 100.0], w_limit)

# assert(X_np[:, 4:].min() >= scale.w_old.min)
# assert(X_np[:, 4:].max() <= scale.w_old.max)

norm_X = scale.normalize(X_np)
re_X = scale.denormalize(norm_X)

dataset = {'dmp_outputs_unscaled'   : X_np,
           'dmp_outputs_scaled'     : norm_X,
           'dmp_traj'               : Y,
           'traj'                   : original_trajs,
           'dmp_bf'                 : dmp_bf,
           'dmp_ay'                 : dmp_ay,
           'dmp_scaling'            : scale,
          }
#%%
filename = 'dmp_parameter-traj' + \
           '_N_' + str(num_dataset) + \
           '_5x_random_points_' + str(max_points) +\
           '_cutting_traj_points_' + str(num_points).replace(' ', '') + \
           '_n-bf_' + str(int(dmp_bf)) + \
           '_ay_' + str(int(dmp_ay)) + \
           '_dt_' + str(int(dmp_dt)) + \
           '_scale-pos_' + str(int(scale.y_new.max)) + \
           '_scale-w_' + str(int(scale.w_new.max)) + \
           '.pkl'
filepath = join('dataset', filename)
pkl.dump(dataset, open(filepath, 'wb'))
print('\nGenerated', num_dataset, 'data')
print('Saved in', filepath)