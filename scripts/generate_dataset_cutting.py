from utils.data_generator import ObjectGenerator
from utils.dataset_importer import ndarray_to_str
import numpy as np
from numpy import array, zeros_like
from numpy.random import rand, randint
from matplotlib import pyplot as plt
from pydmps import DMPs_discrete
from datetime import datetime
from os.path import join, isdir
from PIL import Image
from os import makedirs
import pickle as pkl
from copy import deepcopy
from os.path import isdir

def generate_cutting_traj(top_left,
                          top_right,
                          distance,
                          top_padding,
                          side_padding,
                          max_segments,
                          dmp_bf = None,
                          dmp_ay = None,
                          dmp_dt = None, 
                          dmp_L_bf = None,
                          dmp_L_ay = None,
                          dmp_L_dt = None,
                          seg_dmp_bf = None,
                          seg_dmp_ay = None,
                          seg_dmp_dt = None):
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
    num_segment = (num_cut * 2) + 1
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
    # plt.sc'atter(full_traj[:, 0], full_traj[:, 1])
    # plt.sho'w()
    
    if seg_dmp_bf is not None:
        dmps = []
        ys = []
        for seg in segments:
            dmp = DMPs_discrete(n_dmps = 2,
                                n_bfs = seg_dmp_bf,
                                ay = np.ones(2) * seg_dmp_ay,
                                dt = seg_dmp_dt)
            dmp.imitate_path(seg.T)
            dmps.append(dmp)
            y, _, _ = dmp.rollout()
            ys.append(y)
        
        y_combined_segment = array(ys)[:num_segment].reshape(-1, 2)
    # print(full_traj)
    else:
        ys = None
        dmps = None
        y_combined_segment = None
    
    if dmp_bf is not None:
        dmp_fair = DMPs_discrete(n_dmps = 2,
                            n_bfs = dmp_bf,
                            ay = np.ones(2) * dmp_ay,
                            dt = dmp_dt)
        dmp_fair.imitate_path(full_traj.T)
        y_fair, _, _ = dmp_fair.rollout()
    else:
        dmp_fair = None
        
    
    if dmp_L_bf is not None:
        dmp_accurate = DMPs_discrete(n_dmps = 2,
                                     n_bfs = dmp_L_bf,
                                     ay = np.ones(2) * dmp_L_ay,
                                     dt = dmp_L_dt)
        dmp_accurate.imitate_path(full_traj.T)
        y_accurate, _, _ = dmp_accurate.rollout()
    else:
        dmp_accurate = None
    
    # plt.xlim(-0.1, 1.6)
    # plt.ylim(-0.1, 1.6)
    # plt.plot(y_combined_segment[:, 0], y_combined_segment[:, 1], c = 'g')
    # plt.scatter(y_combined_segment[:, 0], y_combined_segment[:, 1], c = 'g')
    # plt.show()
    
    # plt.scatter(range(y_combined_segment.shape[0]), y_combined_segment[:, 0], c = 'g')
    # plt.show()
    # plt.scatter(range(y_combined_segment.shape[0]), y_combined_segment[:, 1], c = 'g')
    # plt.show()
    
    # plt.plot(y_fair[:, 0], y_fair[:, 1], c = 'r')
    # plt.scatter(y_fair[:, 0], y_fair[:, 1], c = 'r')
    # plt.show()
    # plt.scatter(y_accurate[:, 0], y_accurate[:, 1], c = 'r')
    # plt.show()
    
    # print(dmp_fair.w.min())
    # print(dmp_fair.w.max())
        
    return ys, segments, dmps, num_cut, dmp_fair, dmp_accurate, full_traj, y_combined_segment


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
runcell(0, '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/scripts/generate_dataset_cutting.py')

if __name__ == '__main__':
    
    SAVE = 1
    PRINT_PLOT = 0
    IMAGE_DIM = (100, 100)
    NUM_DATASET = int(1e3)
    # NUM_DATASET = int(1e2)
    print_every = 100
    generation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # SAVE_DIR = 'D:\\rllab\\scripts\\dmp\\Segmented_Deep_DMPs\\data\\images\\cutting_traj\\' + str(NUM_DATASET) + '_' + generation_time
    # SAVE_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/images/cutting_' + str(NUM_DATASET) + '_' + generation_time
    # if SAVE: makedirs(SAVE_DIR)
    max_segments = int(1.5 // 0.2) * 2 + 1
    
    generation_time = datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
    DATA_NAME = 'cutting_{}'.format(NUM_DATASET)
    IMG_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/images/cutting'
    PKL_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/cutting'
    if SAVE: 
        if not isdir(PKL_DIR): 
            makedirs(PKL_DIR)
    IMG_DIR = join(IMG_DIR, DATA_NAME)
    IMG_DIR += generation_time
    PKL_DIR = join(PKL_DIR, DATA_NAME)
    
    if SAVE: makedirs(IMG_DIR)
    
    base_bf = 20
    base_dt = 0.001
    
    pkl_data = {'image': [],
                'image_dim': (1, IMAGE_DIM[0], IMAGE_DIM[1]),
                'original_trajectory': [],
                'normal_dmp_seg_num': np.ones(NUM_DATASET).reshape(-1, 1),
                'normal_dmp_dt': base_dt,
                'normal_dmp_y0': [],
                'normal_dmp_goal': [],
                'normal_dmp_w': [],
                'normal_dmp_tau': [],
                'normal_dmp_bf': max_segments * base_bf,
                'normal_dmp_ay': 100,
                'normal_dmp_trajectory': [],
                'normal_dmp_L_y0': [],
                'normal_dmp_L_goal': [],
                'normal_dmp_L_w': [],
                'normal_dmp_L_tau': [],
                'normal_dmp_L_bf': 1000,
                'normal_dmp_L_ay': 200,
                'normal_dmp_trajectory_accurate': [],
                'segmented_dmp_max_seg_num': max_segments,
                'segmented_dmp_seg_num': [],
                'segmented_dmp_y0': [],
                'segmented_dmp_goal': [],
                'segmented_dmp_w': [],     
                'segmented_dmp_tau': [],
                'segmented_dmp_dt': base_dt * max_segments,
                'segmented_dmp_bf': base_bf,
                'segmented_dmp_ay': 7,
                'segmented_dmp_trajectory': []
                }
    
    
    # DATA_IMAGES = []
    # DATA_IMAGE_NAMES = []
    # DATA_NUM_SEGMENT = []
    
    # DATA_Y0_SEGMENTS = []
    # DATA_YGOAL_SEGMENTS = []
    # DATA_W_SEGMENTS = []
    
    # DATA_Y0_NORMAL = []
    # DATA_YGOAL_NORMAL = []
    # DATA_W_NORMAL = []
    
    # DATA_DMP_TRAJ = []
    
    for i in range(NUM_DATASET):
        base_shape = array([[0.25, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.5 + (0.5 * rand())],
                            [1.0, 0.5 + (0.5 * rand())],
                            [1.0, 0.0],
                            [0.75, 0.0]])
        gen = ObjectGenerator(base_shape = base_shape)
        img_name = 'images_' + str(i+1) + '.jpg'
        save_path = join(IMG_DIR, img_name)
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
    
        ys, segs, dmps, num_cut, dmp_fair, dmp_accurate, original_trajectory, segment_dmp_trajectory = generate_cutting_traj(top_left = (x_min, top_left),
                                                                                                                             top_right = (x_max, top_right),
                                                                                                                             distance = 0.2,
                                                                                                                             top_padding = 0.2,
                                                                                                                             side_padding = 0.05,
                                                                                                                             max_segments = int(1.5 // 0.2),
                                                                                                                             dmp_bf = pkl_data['normal_dmp_bf'],
                                                                                                                             dmp_ay = pkl_data['normal_dmp_ay'],
                                                                                                                             dmp_dt = pkl_data['normal_dmp_dt'], 
                                                                                                                             dmp_L_bf = pkl_data['normal_dmp_L_bf'],
                                                                                                                             dmp_L_ay = pkl_data['normal_dmp_L_ay'],
                                                                                                                             dmp_L_dt = pkl_data['normal_dmp_dt'],
                                                                                                                             seg_dmp_bf = pkl_data['segmented_dmp_bf'],
                                                                                                                             seg_dmp_ay = pkl_data['segmented_dmp_ay'],
                                                                                                                             seg_dmp_dt = pkl_data['segmented_dmp_dt'])
        num_segment = (num_cut * 2) + 1
        reconstructed_traj = np.array(ys[:num_segment]).reshape(-1, 2)
        if SAVE:
            img = array(Image.open(save_path).convert("L").resize((IMAGE_DIM[0], IMAGE_DIM[1])))
            pkl_data['image'].append(img)
            # DATA_IMAGES.append(img)#.reshape(1, 50, 50))
        # DATA_IMAGES.append(array(Image.open(save_path)))
        pkl_data['original_trajectory'].append(original_trajectory)
        
        pkl_data['normal_dmp_y0'].append(dmp_fair.y0)
        pkl_data['normal_dmp_goal'].append(dmp_fair.goal)
        pkl_data['normal_dmp_w'].append(dmp_fair.w)
        pkl_data['normal_dmp_tau'].append(1.)
        y_fair, _, _ = dmp_fair.rollout()
        pkl_data['normal_dmp_trajectory'].append(y_fair)
        
        pkl_data['normal_dmp_L_y0'].append(dmp_accurate.y0)
        pkl_data['normal_dmp_L_goal'].append(dmp_accurate.goal)
        pkl_data['normal_dmp_L_w'].append(dmp_accurate.w)
        pkl_data['normal_dmp_L_tau'].append(1.)
        y_accurate, _, _ = dmp_accurate.rollout()
        pkl_data['normal_dmp_trajectory_accurate'].append(y_accurate)
        
        pkl_data['segmented_dmp_seg_num'].append(num_segment)
        segment_dmp_y0s = []
        segment_dmp_goals = []
        segment_dmp_ws = []
        segment_dmp_taus = []
        for dmp in dmps:
            segment_dmp_y0s.append(dmp.y0)
            segment_dmp_goals.append(dmp.goal)
            segment_dmp_ws.append(dmp.w)
            segment_dmp_taus.append(1.)
        pkl_data['segmented_dmp_y0'].append(segment_dmp_y0s)
        pkl_data['segmented_dmp_goal'].append(segment_dmp_goals)
        pkl_data['segmented_dmp_w'].append(segment_dmp_ws)
        pkl_data['segmented_dmp_tau'].append(segment_dmp_taus)
        pkl_data['segmented_dmp_trajectory'].append(segment_dmp_trajectory)
        
        # DATA_IMAGE_NAMES.append(img_name)
        # DATA_NUM_SEGMENT.append(num_segment)
        # DATA_DMP_TRAJ.append(ndarray_to_str(segment_dmp_trajectory))
        # y_0s = []
        # y_goals = []
        # ws = []
        # for i, dmp in enumerate(dmps):
        #     if i < num_segment:
        #         y_0s.append(dmp.y0)
        #         y_goals.append(dmp.goal)
        #         ws.append(dmp.w)
        #     # else:
        #     #     y_0s.append(y_0s[-1])
        #     #     y_goals.append(y_goals[-1])
        #     #     ws.append(zeros_like(dmp.w))
        # DATA_Y0_SEGMENTS.append(y_0s)
        # DATA_YGOAL_SEGMENTS.append(y_goals)
        # DATA_W_SEGMENTS.append(ws)
        
        # DATA_Y0_NORMAL.append(dmp_fair.y0)
        # DATA_YGOAL_NORMAL.append(dmp_fair.goal)
        # DATA_W_NORMAL.append(dmp_fair.w)
        
        if (i + 1) % print_every == 0:
            print('Generated', (i + 1), '/', NUM_DATASET)
#%%
    to_process = deepcopy(pkl_data)
    
    unique_lengths = []
    for i in to_process['segmented_dmp_w']:
        if len(i) not in unique_lengths:
            unique_lengths.append(len(i))
    unique_lengths = sorted(unique_lengths)
    unique_lengths = [i for i in range(1, unique_lengths[0])] + unique_lengths
    
    idx_segments = {'y0': [[] for i in range(unique_lengths[-1])],
                    'goal': [[] for i in range(unique_lengths[-1])],
                    'w': [[] for i in range(unique_lengths[-1])],
                    'tau': [[] for i in range(unique_lengths[-1])]}
    
    for i in range(len(to_process['segmented_dmp_y0'])):
        for seg in range(len(to_process['segmented_dmp_y0'][i])):
            idx_segments['y0'][seg].append(to_process['segmented_dmp_y0'][i][seg])
            idx_segments['goal'][seg].append(to_process['segmented_dmp_goal'][i][seg])
            idx_segments['w'][seg].append(to_process['segmented_dmp_w'][i][seg])
            idx_segments['tau'][seg].append(to_process['segmented_dmp_tau'][i][seg])
        
    idx_segments['y0'] = [np.array(i) for i in idx_segments['y0']]
    idx_segments['goal'] = [np.array(i) for i in idx_segments['goal']]
    idx_segments['w'] = [np.array(i) for i in idx_segments['w']]
    idx_segments['tau'] = [np.array(i) for i in idx_segments['tau']]
    
    
    pads = idx_segments
    
    for i in range(len(to_process['segmented_dmp_y0'])):
        if len(to_process['segmented_dmp_y0'][i]) < unique_lengths[-1]:
            while len(to_process['segmented_dmp_y0'][i]) < unique_lengths[-1]:
                to_process['segmented_dmp_y0'][i].append(pads['y0'][len(to_process['segmented_dmp_y0'][i])].mean(axis = 0))
                to_process['segmented_dmp_goal'][i].append(pads['goal'][len(to_process['segmented_dmp_goal'][i])].mean(axis = 0))
                to_process['segmented_dmp_w'][i].append(pads['w'][len(to_process['segmented_dmp_w'][i])].mean(axis = 0))
                to_process['segmented_dmp_tau'][i].append(pads['tau'][len(to_process['segmented_dmp_tau'][i])].mean(axis = 0))
    pkl_data = to_process
    
    data_len = len(pkl_data['image'])
    dof = pkl_data['original_trajectory'][0].shape[1]
    
    pkl_data['image']                   = np.array(pkl_data['image'])
    pkl_data['normal_dmp_y0']           = np.array(pkl_data['normal_dmp_y0']).reshape(data_len, 1, dof)
    pkl_data['normal_dmp_goal']         = np.array(pkl_data['normal_dmp_goal']).reshape(data_len, 1, dof)
    pkl_data['normal_dmp_w']            = np.array(pkl_data['normal_dmp_w']).reshape(data_len, 1, dof, pkl_data['normal_dmp_bf'])
    pkl_data['normal_dmp_tau']          = np.array(pkl_data['normal_dmp_tau']).reshape(-1, 1)
    pkl_data['normal_dmp_trajectory']   = np.array(pkl_data['normal_dmp_trajectory'])
    pkl_data['normal_dmp_L_y0']           = np.array(pkl_data['normal_dmp_L_y0']).reshape(data_len, 1, dof)
    pkl_data['normal_dmp_L_goal']         = np.array(pkl_data['normal_dmp_L_goal']).reshape(data_len, 1, dof)
    pkl_data['normal_dmp_L_w']            = np.array(pkl_data['normal_dmp_L_w']).reshape(data_len, 1, dof, pkl_data['normal_dmp_bf_accurate'])
    pkl_data['normal_dmp_L_tau']          = np.array(pkl_data['normal_dmp_L_tau']).reshape(-1, 1)
    pkl_data['normal_dmp_trajectory_accurate']   = np.array(pkl_data['normal_dmp_trajectory_accurate'])
    pkl_data['segmented_dmp_seg_num']   = np.array(pkl_data['segmented_dmp_seg_num']).reshape(-1, 1)
    pkl_data['segmented_dmp_goal']      = np.array(pkl_data['segmented_dmp_goal'])
    pkl_data['segmented_dmp_tau']       = np.array(pkl_data['segmented_dmp_tau'])
    pkl_data['segmented_dmp_w']         = np.array(pkl_data['segmented_dmp_w'])
    pkl_data['segmented_dmp_y0']        = np.array(pkl_data['segmented_dmp_y0'])
    if 'normal_dmp_target_trajectory' in pkl_data: pkl_data['normal_dmp_target_trajectory']   = np.array(pkl_data['normal_dmp_target_trajectory'])
    if 'segmented_dmp_target_trajectory' in pkl_data:  pkl_data['segmented_dmp_target_trajectory']        = np.array(pkl_data['segmented_dmp_target_trajectory'])
    if 'rotation_degrees' in pkl_data:  pkl_data['rotation_degrees'] = np.array(pkl_data['rotation_degrees'])
    
    #%%
    
    PKL_NAME = PKL_DIR
    PKL_NAME += '.num_data_' + str(len(pkl_data['image'])) + '_num_seg_' + str(pkl_data['segmented_dmp_max_seg_num'])
    PKL_NAME += '.normal_dmp_bf_' + str(pkl_data['normal_dmp_bf']) + '_ay_' + str(pkl_data['normal_dmp_ay']) + '_dt_' + str(pkl_data['normal_dmp_dt'])
    PKL_NAME += '.seg_dmp_bf_' + str(pkl_data['segmented_dmp_bf']) + '_ay_' + str(pkl_data['segmented_dmp_ay']) + '_dt_' + str(pkl_data['segmented_dmp_dt'])
    PKL_NAME += generation_time
    PKL_NAME += '.pkl'
    
    pkl.dump(pkl_data, open(PKL_NAME, 'wb'))
    print('Saved as {}'.format(PKL_NAME))
#%%    
"""    dataset = {'image'                      : array(DATA_IMAGES),
               'image_dim'                  : (1, IMAGE_DIM[0], IMAGE_DIM[1]),
               'image_name'                 : DATA_IMAGE_NAMES,
               'dmp_traj'                   : DATA_DMP_TRAJ,
               'segmented_dmp_max_seg_num'  : 15,
               'segmented_dmp_seg_num'      : array(DATA_NUM_SEGMENT).reshape(-1, 1),
               'segmented_dmp_y0'           : array(DATA_Y0_SEGMENTS),
               'segmented_dmp_goal'         : array(DATA_YGOAL_SEGMENTS),
               'segmented_dmp_w'            : array(DATA_W_SEGMENTS),
               'segmented_dmp_tau'          : np.ones((array(DATA_W_SEGMENTS).shape[0], array(DATA_W_SEGMENTS).shape[1])),
               'normal_dmp_y0'              : array(DATA_Y0_NORMAL),
               'normal_dmp_goal'            : array(DATA_YGOAL_NORMAL),
               'normal_dmp_w'               : array(DATA_W_NORMAL),
               'normal_dmp_tau'             : array(np.ones(len(DATA_IMAGE_NAMES))).reshape(-1, 1)}
#%%
    PKL_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/cutting_traj/'
    # PKL_DIR = 'D:\\rllab\\scripts\\dmp\\Segmented_Deep_DMPs\\data\\pkl\\cutting_traj\\'
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
    if SAVE: print('Saved in', filepath)"""