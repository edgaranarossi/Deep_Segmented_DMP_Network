from utils.data_generator import ObjectGenerator
import numpy as np
from numpy import array
from numpy.random import rand, randint
from matplotlib import pyplot as plt
from pydmps import DMPs_discrete
from datetime import datetime
from os.path import join, isdir
from PIL import Image
from os import makedirs
import pickle as pkl

def generate_cutting_traj(top_left,
                          top_right,
                          distance,
                          top_padding,
                          side_padding,
                          max_segments):
    cut_up = array([[0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])
    cut_down = array([[1.0, 1.0],
                      [1.0, 0.0]])
    cut_up[:, 0] *= distance
    cut_up[:, 1] *= top_left[1]
    cut_up[:, 1] = np.where(cut_up[:, 1] != 0, cut_up[:, 1] + top_padding, cut_up[:, 1])
    
    cut_down[:, 0] *= distance
    cut_down[:, 1] *= top_left[1]
    cut_down[:, 1] = np.where(cut_down[:, 1] != 0, cut_down[:, 1] + top_padding, cut_down[:, 1])
    object_length = top_right[0] - top_left[0] - side_padding
    num_cut = int(object_length // distance)
    extra_padding = object_length - (num_cut * distance)
    x_start = top_left[0] + (side_padding/2) + (extra_padding/2)
    y_modifier = (top_right[1] - top_left[1]) / num_cut
    
    segments = []
    full_traj = []
    init_cut = array([x_start - distance, 0]) + cut_down
    init_cut[:, 1] = np.where(init_cut[:, 1] != 0, init_cut[:, 1] - y_modifier, init_cut[:, 1])
    segments.append(init_cut)
    for seg in init_cut: full_traj.append(seg)
    for i in range(max_segments):
        segments.append(array([x_start, 0]) + cut_up)
        segments.append(array([x_start, 0]) + cut_down)
        if i < num_cut:
            for seg in segments[-2]: full_traj.append(seg)
            for seg in segments[-1]: full_traj.append(seg)
        x_start += distance
        cut_up[:, 1] = np.where(cut_up[:, 1] != 0, cut_up[:, 1] + y_modifier, cut_up[:, 1])
        cut_down[:, 1] = np.where(cut_down[:, 1] != 0, cut_down[:, 1] + y_modifier, cut_down[:, 1])
        # print(cut_down)
    
    # print(segments)
    # print(full_traj)
    full_traj = array(full_traj).reshape(-1, 2)
    # plt.scatter(full_traj[:, 0], full_traj[:, 1])
    # plt.show()
    
    dmps = []
    ys = []
    for seg in segments:
        dmp = DMPs_discrete(n_dmps = 2,
                            n_bfs = 20,
                            ay = np.ones(2) * 7,
                            dt = 0.02)
        dmp.imitate_path(seg.T)
        dmps.append(dmp)
        y, _, _ = dmp.rollout()
        ys.append(y)
    
    y_combined_segment = array(ys).reshape(-1, 2)
    # print(full_traj)
    dmp_full = DMPs_discrete(n_dmps = 2,
                        n_bfs = 300,
                        ay = np.ones(2) * 100,
                        dt = 0.001)
    dmp_full.imitate_path(full_traj.T)
    y_full, _, _ = dmp_full.rollout()
    
    # plt.plot(y_combined_segment[:, 0], y_combined_segment[:, 1], c = 'g')
    # plt.scatter(y_combined_segment[:, 0], y_combined_segment[:, 1], c = 'g')
    # plt.plot(y_full[:, 0], y_full[:, 1], c = 'r')
    # plt.scatter(y_full[:, 0], y_full[:, 1], c = 'r')
    # plt.show()
    
    # print(dmp_full.w.min())
    # print(dmp_full.w.max())
        
    return ys, segments, dmps, num_cut, dmp_full


# base_shape = array([[0.0, 0.0],
#                     [0.0, 0.5 + (0.5 * rand())],
#                     [1.0, 0.5 + (0.5 * rand())],
#                     [1.0, 0.0]])
# gen = ObjectGenerator(base_shape = base_shape)
# x = gen.generate(size_random_magnitude = (1., .5), 
#                  shape_random_magnitude = None, 
#                  plot_shape = True,
#                  plot_save_path = '/home/edgar/rllab/scripts/dmp/tes.jpg',
#                  plot_target_size = (50, 50))

# x_min = x.min(axis = 0)[0]
# x_max = x.max(axis = 0)[0]

# top_left = np.where(x[:, 0] == x_min, x[:, 1], 0).max()
# top_right = np.where(x[:, 0] == x_max, x[:, 1], 0).max()

# ys, segs, dmps, num_cut = generate_cutting_traj(top_left = (x_min, top_left),
#                             top_right = (x_max, top_right),
#                             distance = 0.2,
#                             top_padding = 0.2,
#                             side_padding = 0.05,
#                             max_segments = int(1.5 // 0.2))

# for i in range(len(ys)):
#     if i < num_cut * 2 + 1:
#         plt.plot(ys[i][:, 0], ys[i][:, 1])
#         plt.scatter(ys[i][:, 0], ys[i][:, 1], zorder = 5)
    
#%%
SAVE = 1
PRINT_PLOT = 0
NUM_DATASET = int(1e3)
print_every = 100
generation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SAVE_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/images/cutting_' + str(NUM_DATASET) + '_' + generation_time
if SAVE: makedirs(SAVE_DIR)

DATA_IMAGES = []
DATA_IMAGE_NAMES = []
DATA_NUM_SEGMENT = []

DATA_Y0 = []
DATA_YGOAL_SEGMENTS = []
DATA_W_SEGMENTS = []

DATA_Y0_NORMAL = []
DATA_YGOAL_NORMAL = []
DATA_W_NORMAL = []

DATA_DMP_TRAJ = []

for i in range(NUM_DATASET):
    base_shape = array([[0.25, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.5 + (0.5 * rand())],
                        [1.0, 0.5 + (0.5 * rand())],
                        [1.0, 0.0],
                        [0.75, 0.0]])
    gen = ObjectGenerator(base_shape = base_shape)
    img_name = 'images_' + str(i) + '.jpg'
    save_path = join(SAVE_DIR, img_name)
    x = gen.generate(size_random_magnitude = (1., .5), 
                     shape_random_magnitude = (randint(0, 3)/100, randint(0, 6)/100), 
                     smoothing_magnitude = randint(0, 4),
                     plot_shape = PRINT_PLOT,
                     plot_save_path = None if not SAVE else save_path,
                     # plot_target_size = (50, 50))
                     plot_target_size = None)
    
    x_min = x.min(axis = 0)[0]
    x_max = x.max(axis = 0)[0]

    top_left = np.where(x[:, 0] == x_min, x[:, 1], 0).max()
    top_right = np.where(x[:, 0] == x_max, x[:, 1], 0).max()

    ys, segs, dmps, num_cut, dmp_full = generate_cutting_traj(top_left = (x_min, top_left),
                                top_right = (x_max, top_right),
                                distance = 0.2,
                                top_padding = 0.2,
                                side_padding = 0.05,
                                max_segments = int(1.5 // 0.2))
    num_segment = (num_cut * 2) + 1
    reconstructed_traj = np.array(ys[:num_segment]).reshape(-1, 2)
    if SAVE: DATA_IMAGES.append(array(Image.open(save_path).convert("L").resize((50, 50))))#.reshape(1, 50, 50))
    # DATA_IMAGES.append(array(Image.open(save_path)))
    DATA_IMAGE_NAMES.append(img_name)
    DATA_NUM_SEGMENT.append(num_segment)
    DATA_DMP_TRAJ.append(reconstructed_traj)
    DATA_Y0.append(dmps[0].y0)
    y_goals = []
    ws = []
    for dmp in dmps:
        y_goals.append(dmp.goal)
        ws.append(dmp.w)
    DATA_YGOAL_SEGMENTS.append(y_goals)
    DATA_W_SEGMENTS.append(ws)
    
    DATA_Y0_NORMAL.append(dmp_full.y0)
    DATA_YGOAL_NORMAL.append(dmp_full.goal)
    DATA_W_NORMAL.append(dmp_full.w)
    
    if (i + 1) % print_every == 0:
        print('Generated', (i + 1), '/', NUM_DATASET)

dataset = {'image'              : array(DATA_IMAGES),
           'image_name'         : DATA_IMAGE_NAMES,
           'num_segments'       : array(DATA_NUM_SEGMENT),
           'dmp_traj'           : DATA_DMP_TRAJ,
           'max_segments'       : 15,
           'dmp_y0_segments'    : array(DATA_Y0),
           'dmp_goal_segments'  : array(DATA_YGOAL_SEGMENTS),
           'dmp_w_segments'     : array(DATA_W_SEGMENTS),
           'dmp_y0_normal'      : array(DATA_Y0_NORMAL),
           'dmp_goal_normal'    : array(DATA_YGOAL_NORMAL),
           'dmp_w_normal'       : array(DATA_W_NORMAL)}

PKL_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/cutting_traj/'
if not isdir(PKL_DIR) and SAVE: makedirs(PKL_DIR)

filename = 'image_num-seg_y0_goals_ws'
filename += '_N_' + str(NUM_DATASET)
filename += '+seg=n-bf_' + str(20)
filename += '_ay_' + str(7)
filename += '_dt' + str(0.02)
filename += '_max-seg_' + str(15)
filename += '+cut=dist_' + str(0.2)
filename += '_top-pad_' + str(0.2)
filename += '_side-pad_' + str(0.05)
filename += '_normal-dmp_limited_y'
filename += '_' + generation_time
filename += '.pkl'

filepath = join(PKL_DIR, filename)
if SAVE: pkl.dump(dataset, open(filepath, 'wb'))
if SAVE: print('\nGenerated', NUM_DATASET, 'data')
if SAVE: print('Saved in', filepath)