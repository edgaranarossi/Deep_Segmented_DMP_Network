from utils.data_generator import SegmentTrajectoryGenerator
import numpy as np
from PIL import Image
import pickle as pkl
from os.path import join, isdir, dirname
from os import getcwd, makedirs
from datetime import datetime
from utils.dataset_importer import DMPParamScale

def generate_shape_segments(shapes,
                            segment_types,
                            dict_dmp_bf,
                            dict_dmp_ay,
                            dict_dmp_dt,
                            subdiv_traj_length,
                            random_magnitude,
                            save_path,
                            image_target_size,
                            prefix,
                            pad_points,
                            dmp_output_dt = None,
                            dmp_output_bf = None,
                            dmp_output_ay = None):
    global images, image_names, seg_dict_dmp_outputs, seg_dict_dmp_types, traj, traj_interpolated, segment_dmp_traj, segment_dmp_traj_padded, num_segs, dmp_traj, dmp_y0_goal_w_scaled, dmp_y0_goal_w
    if not isdir(save_path): makedirs(save_path)
    shape_segments_generator = SegmentTrajectoryGenerator(shape_templates = shapes,
                                                          segment_types = segment_types,
                                                          dict_dmp_bf = dict_dmp_bf,
                                                          dict_dmp_ay = dict_dmp_ay,
                                                          dict_dmp_dt = dict_dmp_dt,
                                                          subdiv_traj_length = subdiv_traj_length,
                                                          pad_points = pad_points)
    rand_shapes = shape_segments_generator.generateRandomizedShape(magnitude = random_magnitude,
                                                                   return_padded = 1, 
                                                                   return_interpolated = 1,
                                                                   plot_save_path = save_path,
                                                                   plot_prefix = prefix,
                                                                   plot_target_size = image_target_size,
                                                                   plot_linewidth = 7.5,
                                                                   dmp_output_dt = dmp_output_dt,
                                                                   dmp_output_bf = dmp_output_bf,
                                                                   dmp_output_ay = dmp_output_ay)
    for i in range(len(rand_shapes['image_names'])):
        img = Image.open(join(save_path, rand_shapes['image_names'][i]))
        images.append(np.array(img).reshape(3, image_target_size, image_target_size))
        image_names.append(rand_shapes['image_names'][i])
        traj.append(rand_shapes['traj'][i])
        traj_interpolated.append(rand_shapes['traj_interpolated'][i])
        segment_dmp_traj.append(rand_shapes['segment_dmp_traj'][i])
        segment_dmp_traj_padded.append(rand_shapes['segment_dmp_traj_padded'][i])
        seg_dict_dmp_outputs.append(rand_shapes['points_padded'][i])
        seg_dict_dmp_types.append(rand_shapes['segment_types_padded'][i])
        num_segs.append(rand_shapes['segment_num'][i])
        dmp_traj.append(rand_shapes['dmp_traj'][i])
        dmp_y0_goal_w.append(rand_shapes['dmp_y0_goal_w'][i])

shapes = [
          [[0.0, 5.0],
           [4.0, 0.0],
           [6.0, 2.5]],
          [[1.5, 1.0],
           [3.5, 5.0],
           [5.5, 1.0],
           [1.5, 1.0]],
          [[0.0, 6.0],
           [6.0, 6.0],
           [3.0, 4.0],
           [2.5, 0.5]],
          [[2.0, 2.0],
           [2.0, 4.0],
           [4.0, 4.0],
           [4.0, 2.0],
           [2.0, 2.0]],
          [[2.5, 5.0],
           [0.0, 1.0],
           [4.5, 1.5],
           [3.5, 3.0],
           [5.25, 0.5]],
          [[1.0, 0.0],
           [2.5, 5.0],
           [4.0, 0.0],
           [0.0, 3.0],
           [5.0, 3.0],
           [1.0, 0.0]],
          [[1.0, 0.0],
           [0.0, 3.0],
           [2.5, 5.0],
           [5.0, 3.0],
           [4.0, 0.0],
           [1.0, 0.0]],
          [[1.0, 0.0],
           [1.0, 5.0],
           [2.5, 5.0],
           [2.5, 0.0],
           [2.5, 5.0],
           [4.0, 5.0],
           [4.0, 0.0]]
         ]
seg_types = None
images = []
image_names = []
seg_dict_dmp_outputs = []
seg_dict_dmp_types = []
segment_dmp_traj = []
segment_dmp_traj_padded = []
traj = []
traj_interpolated = []
dmp_traj = []
dmp_y0_goal_w_scaled = []
dmp_y0_goal_w = []
num_segs = []

num_shape = len(shapes)
num_dataset = 12500
# num_dataset = int(1e5)
print_every = 100
checkpoints = [i for i in range(print_every, num_dataset * len(shapes), print_every)]
start_num = 0
cur_num = start_num

dict_dmp_bf = 1
dict_dmp_ay = 4
dict_dmp_dt = 0.05
dmp_output_dt = 0.01
dmp_output_bf = 6
dmp_output_ay = 10
subdiv_traj_length = 300
random_magnitude = 1e-1
image_target_size = 150
generation_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATASET_NAME = 'shapes-8'
# DATASET_NAME = 'random_lines_4'
IMAGE_DIR = join(dirname(getcwd()), 'data/images/' + DATASET_NAME + '_'+ generation_date)
max_segments = 4
max_num_points = max_segments + 1
# max_num_points = None

while cur_num < start_num + num_dataset:
    
    # size = 5
    # num_points = np.random.randint(2, max_num_points + 1)
    # random_shape = (np.random.rand(1, num_points, 2) * size)# * 2)# - size
    # shapes = random_shape.tolist()
    # seg_types = np.random.randint(1, 3, (1, num_points-1)).tolist()

    generate_shape_segments(shapes,
                            seg_types,
                            dict_dmp_bf,
                            dict_dmp_ay,
                            dict_dmp_dt,
                            subdiv_traj_length,
                            random_magnitude,
                            save_path = IMAGE_DIR,
                            image_target_size = image_target_size,
                            prefix = 'iter_'+str(cur_num + 1)+'-',
                            # pad_points = max_num_points,
                            pad_points = None,
                            dmp_output_dt = dmp_output_dt,
                            dmp_output_bf = dmp_output_bf,
                            dmp_output_ay = dmp_output_ay)
    cur_num += 1

    while len(checkpoints) > 0 and checkpoints[0] <= len(images):
        print('Generated', int(checkpoints[0]), '/', int(num_dataset * len(shapes)))
        # print('Generated', len(X), '/', num_dataset)
        del checkpoints[0]

dmp_y0_goal_w = np.array(dmp_y0_goal_w)
scale = DMPParamScale([dmp_y0_goal_w[:,:4].min(), dmp_y0_goal_w[:,:4].max()], [dmp_y0_goal_w[:,4:].min(), dmp_y0_goal_w[:,4:].max()])
dmp_y0_goal_w_scaled = scale.normalize(dmp_y0_goal_w)
print('Generated', len(images), '/', int(num_dataset * len(shapes)))

dataset = {'image'                      : np.array(images),
           'image_name'                 : image_names,
           'points_padded'              : np.array(seg_dict_dmp_outputs),
           'segment_types_padded'       : np.array(seg_dict_dmp_types),
           'segment_dmp_traj'           : segment_dmp_traj,
           'segment_dmp_traj_padded'    : np.array(segment_dmp_traj_padded),
           'traj'                       : np.array(traj),
           'traj_interpolated'          : np.array(traj_interpolated),
           'dict_dmp_bf'                : dict_dmp_bf,
           'dict_dmp_ay'                : dict_dmp_ay,
           'dict_dmp_dt'                : dict_dmp_dt,
           'normal_dmp_bf'              : dmp_output_bf,
           'normal_dmp_ay'              : dmp_output_ay,
           'normal_dmp_dt'              : dmp_output_dt,
           'normal_dmp_y_track'         : np.array(dmp_traj),
        #    'normal_dmp_dy_track'        : np.array(dmp_traj),
        #    'normal_dmp_ddy_track'       : np.array(dmp_traj),
           'num_segments'               : num_segs,
           'dmp_y0_goal_w'              : dmp_y0_goal_w,
           'dmp_y0_goal_w_scaled'       : dmp_y0_goal_w_scaled,
           'dmp_scaling'                : scale
          }
#%%
filename = 'image-dict_output-traj'
filename += '_N_' + str(len(images))
# filename += '_max-points_' + str(max_num_points)
filename += '_dict_n-bf_' + str(int(dict_dmp_bf))
filename += '_ay_' + str(int(dict_dmp_ay))
filename += '_dt_' + str(dict_dmp_dt)
filename += '_normal_n-bf_' + str(int(dmp_output_bf))
filename += '_ay_' + str(int(dmp_output_ay))
filename += '_dt_' + str(dmp_output_dt)
filename += '_' + generation_date
filename += '.pkl'
if not isdir(join(dirname(getcwd()), 'data/pkl', DATASET_NAME)): makedirs(join(dirname(getcwd()), 'data/pkl', DATASET_NAME))
filepath = join(dirname(getcwd()), 'data/pkl', DATASET_NAME, filename)
pkl.dump(dataset, open(filepath, 'wb'))
print('\nGenerated', len(images), 'data')
print('Saved in', filepath)