from utils.data_generator import SegmentTrajectoryGenerator
import numpy as np
from PIL import Image
import pickle as pkl
from os.path import join, isdir, dirname
from os import getcwd, makedirs
from datetime import datetime

def generate_shape_segments(shapes,
                            segment_types,
                            dict_dmp_bf,
                            dict_dmp_ay,
                            dict_dmp_dt,
                            subdiv_traj_length,
                            random_magnitude,
                            save_path,
                            image_target_size,
                            prefix):
    global images, image_names, seg_dict_dmp_outputs, seg_dict_dmp_types, dmp_traj_interpolated, dmp_traj_padded, num_segs
    if not isdir(save_path): makedirs(save_path)
    shape_segments_generator = SegmentTrajectoryGenerator(shape_templates = shapes,
                                                          segment_types = segment_types,
                                                          dict_dmp_bf = dict_dmp_bf,
                                                          dict_dmp_ay = dict_dmp_ay,
                                                          dict_dmp_dt = dict_dmp_dt,
                                                          subdiv_traj_length = subdiv_traj_length)
    rand_shapes = shape_segments_generator.generateRandomizedShape(magnitude = random_magnitude,
                                                                   return_padded = 1, 
                                                                   return_interpolated = 1,
                                                                   plot_save_path = save_path,
                                                                   plot_prefix = prefix,
                                                                   plot_target_size = image_target_size,
                                                                   plot_linewidth = 7.5)
    for i in range(len(rand_shapes['dmp_traj'])):
        img = Image.open(join(save_path, rand_shapes['image_names'][i]))
        images.append(np.array(img).reshape(3, image_target_size, image_target_size))
        image_names.append(rand_shapes['image_names'][i])
        dmp_traj_interpolated.append(rand_shapes['dmp_traj_interpolated'][i])
        dmp_traj_padded.append(rand_shapes['dmp_traj_padded'][i])
        seg_dict_dmp_outputs.append(rand_shapes['points_padded'][i])
        seg_dict_dmp_types.append(rand_shapes['segment_types_padded'][i])
        num_segs.append(rand_shapes['segment_num'][i])

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
images = []
image_names = []
seg_dict_dmp_outputs = []
seg_dict_dmp_types = []
dmp_traj_padded = []
dmp_traj_interpolated = []
num_segs = []

num_shape = len(shapes)
num_dataset = int(1e4)
print_every = 100
checkpoints = [i for i in range(print_every, num_dataset, print_every)]
start_num = 0
cur_num = start_num

dict_dmp_bf = 5
dict_dmp_ay = 4
dict_dmp_dt = 0.05
subdiv_traj_length = 300
random_magnitude = 1e-1
image_target_size = 150
generation_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
IMAGE_DIR = join(dirname(getcwd()), 'data/images/shapes-8_80000_'+ generation_date)

while cur_num < start_num + num_dataset:

    generate_shape_segments(shapes,
                            None,
                            dict_dmp_bf,
                            dict_dmp_ay,
                            dict_dmp_dt,
                            subdiv_traj_length,
                            random_magnitude,
                            save_path = IMAGE_DIR,
                            image_target_size = image_target_size,
                            prefix = 'iter_'+str(cur_num + 1)+'-')
    cur_num += 1

    while len(checkpoints) > 0 and checkpoints[0] <= cur_num:
        print('Generated', int(checkpoints[0]), '/', num_dataset)
        # print('Generated', len(X), '/', num_dataset)
        del checkpoints[0]

print('Generated', num_dataset, '/', num_dataset)

dataset = {'image'                      : np.array(images),
           'image_name'                 : image_names,
           'segmented_dict_dmp_outputs' : np.array(seg_dict_dmp_outputs),
           'segmented_dict_dmp_types'   : np.array(seg_dict_dmp_types),
           'dmp_traj_padded'            : np.array(dmp_traj_padded),
           'dmp_traj_interpolated'      : np.array(dmp_traj_interpolated),
           'dict_dmp_bf'                : dict_dmp_bf,
           'dict_dmp_ay'                : dict_dmp_ay,
           'dict_dmp_dt'                : dict_dmp_dt,
           'num_segments'               : num_segs
          }
#%%
filename = 'image-dict_output-traj'
filename += '_N_' + str(num_dataset)
filename += '_n-bf_' + str(int(dict_dmp_bf))
filename += '_ay_' + str(int(dict_dmp_ay))
filename += '_dt_' + str(int(dict_dmp_dt))
filename += '_' + generation_date
filename += '.pkl'
filepath = join(dirname(getcwd()), 'data/pkl', 'shapes-8', filename)
pkl.dump(dataset, open(filepath, 'wb'))
print('\nGenerated', num_dataset, 'data')
print('Saved in', filepath)