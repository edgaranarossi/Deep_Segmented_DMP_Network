import torch
from torch import from_numpy, ones
import numpy as np
from pydmps.dmp_discrete import DMPs_discrete
from os.path import join, isdir
from os import makedirs
from datetime import datetime
from utils.networks import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingParameters:
    def __init__(self):
        self.root_dir = '/home/edgar/rllab/scripts/Segmented_Deep_DMPs'
        # self.root_dir = 'D:\\rllab\\scripts\\dmp\\Segmented_Deep_DMPs'
        self.init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Dataset parameter
        # self.dataset_dir = join(self.root_dir, 'data/pkl/random_lines_3')
        # self.dataset_name = 'image-dict_output-traj_N_100000_dict_n-bf_5_ay_4_dt_0.05_normal_n-bf_6_ay_6_dt_0.01_2022-02-05_04-15-34.pkl'

        # self.dataset_dir = join(self.root_dir, 'data/pkl/shapes-8')
        # self.dataset_name = 'image-dict_output-traj_N_100000_dict_n-bf_1_ay_4_dt_0.05_normal_n-bf_200_ay_75_dt_0.01_2022-02-09_01-26-12.pkl'
        # self.dataset_name = 'image-dict_output-traj_N_100000_dict_n-bf_1_ay_4_dt_0.05_normal_n-bf_6_ay_10_dt_0.01_2022-02-09_17-57-35.pkl'

        # self.dataset_dir = join(self.root_dir, 'data/pkl/cutting')
        # self.dataset_name = 'image_num-seg_y0_goals_ws_N_10000+seg=n-bf_20_ay_7_dt0.02_max-seg_15+cut=dist_0.2_top-pad_0.2_side-pad_0.05_2022-03-16_16-41-59.pkl'
        # self.dataset_name = 'image_num-seg_y0_goals_ws_N_10000+seg=n-bf_20_ay_7_dt0.02_max-seg_15+cut=dist_0.2_top-pad_0.2_side-pad_0.05_2022-03-23_09-24-58.pkl'
        # self.dataset_name = 'image_num-seg_y0_goals_ws_N_10000+seg=n-bf_20_ay_7_dt0.02_max-seg_15+cut=dist_0.2_top-pad_0.2_side-pad_0.05_normal-dmp_limited_y_2022-03-23_11-19-22.pkl'
        # self.dataset_name = 'image_num-seg_y0_goals_ws_N_10000+seg=n-bf_20_ay_7_dt0.02_max-seg_15+cut=dist_0.2_top-pad_0.2_side-pad_0.05_normal-dmp_limited_y_2022-03-23_18-46-09.pkl'
        # self.dataset_name = 'image_num-seg_y0_goals_ws_N_1000+seg=n-bf_20_ay_7_dt0.02_max-seg_15_padded+cut=dist_0.2_top-pad_0.2_side-pad_0.05_normal-dmp_limited_y_2022-04-08_01-07-33.pkl'
        # self.dataset_name = 'image_num-seg_y0_goals_ws_N_1000+seg=n-bf_20_ay_7_dt0.02_max-seg_15+cut=dist_0.2_top-pad_0.2_side-pad_0.05_normal-dmp_limited_y_2022-06-06_00-57-12.pkl'
        # self.dataset_name = 'cutting_5000.num_data_5000_num_seg_15.normal_dmp_bf_300_ay_100_dt_0.001.seg_dmp_bf_20_ay_7_dt_0.015_2022-07-26_23-58-46.pkl'
        
        # self.dataset_name = 'real_distanced_trajectory.num_data_200_num_seg_21.normal_dmp_bf_630_ay_25_dt_0.001.seg_dmp_bf_30_ay_15_dt_0.005.2022-05-31_18-08-28.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_300_num_seg_11.normal_dmp_bf_330_ay_25_dt_0.001.seg_dmp_bf_30_ay_15_dt_0.005.2022-06-04_01-43-25.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_300_num_seg_27.normal_dmp_bf_540_ay_25_dt_0.001.seg_dmp_bf_20_ay_15_dt_0.005.2022-06-05_05-46-43.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_5_num_seg_4.normal_dmp_bf_120_ay_25_dt_0.001.seg_dmp_bf_30_ay_15_dt_0.005.2022-06-06_15-10-38.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_246_num_seg_27.normal_dmp_bf_540_ay_25_dt_0.001.seg_dmp_bf_20_ay_15_dt_0.005.2022-06-06_16-04-48.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_191_num_seg_27.normal_dmp_bf_540_ay_25_dt_0.001.seg_dmp_bf_20_ay_15_dt_0.005.2022-06-06_16-15-19.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_181_num_seg_18.normal_dmp_bf_540_ay_25_dt_0.001.seg_dmp_bf_30_ay_15_dt_0.005.2022-06-07_09-02-30.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_60_num_seg_17.normal_dmp_bf_510_ay_25_dt_0.0001.seg_dmp_bf_30_ay_15_dt_0.01.2022-06-13_05-01-13.pkl'
        # self.dataset_name = 'real_distanced_trajectory.num_data_60_num_seg_17.normal_dmp_bf_340_ay_25_dt_0.0001.seg_dmp_bf_20_ay_15_dt_0.01.2022-06-13_18-05-17.pkl'
        # self.dataset_name = 'rotated_real_distanced_trajectory.num_data_60_num_seg_5.normal_dmp_bf_250_ay_25_dt_0.001.seg_dmp_bf_50_ay_15_dt_0.005.2022-07-01_22-37-57.pkl'
        # self.dataset_name = 'rotated_real_distanced_trajectory.num_data_60_num_seg_5.normal_dmp_bf_250_ay_25_dt_0.001.seg_dmp_bf_50_ay_15_dt_0.005.2022-07-29_00-09-29.pkl'
        # self.dataset_name = 'cutting_100.num_data_100_num_seg_15.normal_dmp_bf_300_ay_100_dt_0.001.seg_dmp_bf_20_ay_7_dt_0.015_2022-09-11_06-56-12.pkl'
        # self.dataset_name = 'rotated_real_distanced_trajectory.num_data_54_num_seg_5.normal_dmp_bf_250_ay_25_dt_0.001.seg_dmp_bf_50_ay_15_dt_0.005.2022-09-12_02-22-15.pkl'
        # self.dataset_name = 'cutting_1000.num_data_1000_num_seg_15.normal_dmp_bf_300_ay_100_dt_0.001.seg_dmp_bf_20_ay_7_dt_0.015_2022-09-12_07-05-00.pkl'


        # self.dataset_dir = join(self.root_dir, 'data/pkl/stacking')
        # self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_5000_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_2022-07-19_17-25-19.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3].num_data_15000_num_seg_24.normal_dmp_bf_48_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.024_2022-07-31_18-07-28.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3].num_data_600_num_seg_24.normal_dmp_bf_48_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.024_2022-08-04_19-07-20.pkl'
        # self.dataset_name = 'stacking_[1].num_data_500_num_seg_8.normal_dmp_bf_16_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.008_2022-08-04_23-14-48.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_5000_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_2022-08-06_10-50-04.pkl'
        # self.dataset_name = 'stacking_[1, 2].num_data_2000_num_seg_16.normal_dmp_bf_32_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.016_2022-08-08_00-18-01.pkl'
        # self.dataset_name = 'stacking_[1].num_data_10_num_seg_8.normal_dmp_bf_16_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.008_2022-08-08_10-58-18.pkl'
        # self.dataset_name = 'stacking_[1, 2].num_data_20_num_seg_16.normal_dmp_bf_32_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.016_2022-08-08_12-49-24.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_50_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_2022-08-08_19-43-13.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_500_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_init_pos_rand_target_pos_rand_2022-08-08_22-06-24.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_500_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_init_pos_fixed_target_pos_rand_2022-08-08_23-10-40.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_50_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_init_pos_rand_target_pos_rand_2022-08-09_17-03-02.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3].num_data_150_num_seg_24.normal_dmp_bf_48_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.024_init_pos_rand_target_pos_fixed_2022-08-09_20-35-58.pkl'
        # self.dataset_name = 'stacking_[1, 2].num_data_500_num_seg_16.normal_dmp_bf_32_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.016_init_pos_rand_target_pos_fixed_2022-08-10_14-24-37.pkl'
        # self.dataset_name = 'stacking_[1, 2].num_data_1000_num_seg_16.normal_dmp_bf_32_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.016_init_pos_rand_target_pos_fixed_2022-08-10_16-12-44.pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-225][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-11_18-32-56].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-450][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-11_18-36-40].pkl'
        # self.dataset_name = 'stacking_[1, 2][num-data-150][max-seg-16][normal-dmp_bf-32_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.016][block_permute-True_random-pos-False][target_random-pos-False][2022-08-11_19-32-47].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-150][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-08-11_22-28-08].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-450][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-12_15-13-01].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-900][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-12_16-08-06].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-900][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-18_16-36-59].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-300][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-False][target_random-pos-False][2022-08-18_16-48-19].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-270][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-18_20-38-00].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-180][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-True_random-pos-False][target_random-pos-False][2022-08-18_22-58-58].pkl'
        # self.dataset_name = 'stacking_[1][num-data-10][max-seg-8][normal-dmp_bf-16_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.008][block_permute-False_random-pos-False][target_random-pos-False][2022-08-24_03-49-05].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-15000][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-08-24_04-40-10].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-150][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-False][target_random-pos-False][2022-09-08_08-18-31].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-9000][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-09-09_05-20-31].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-450][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-09-11_00-44-55].pkl'
        # self.dataset_name = 'stacking_[1, 2, 3][num-data-3000][max-seg-24][normal-dmp_bf-48_ay-25_dt-0.001][seg-dmp_bf-2_ay-3.4_dt-0.024][block_permute-False_random-pos-True][target_random-pos-True][2022-09-11_00-40-48].pkl'

        self.dataset_dir = join(self.root_dir, 'data/pkl/pepper_shaking_6_target')
        self.dataset_name = 'pepper_shaking_6_target_num.90_dof.3_dsdnet[seg.4-bf.50-ay.15-dt.0.004]_cimednet[bf.200-ay.25-dt.0.001]_cimednet_L[bf.1000-ay.25-dt.0.001]_2023-01-04_18-31-27.pkl'

        self.dataset_path   = join(self.dataset_dir,  self.dataset_name)
        self.data_limit     = None
        # self.data_limit     = 358 # 250
        # self.data_limit     = 100 # 70
        self.shuffle_data   = True
        self.scaler         = None

        self.load_model_name = None
        self.data_loaders_model_name = None

        # self.load_model_name = 'Model_SegmentDMPCNN_2022-06-05_23-36-18'
        # self.data_loaders_model_name = self.load_model_name
        # self.data_loaders_model_name = 'Model_DSDNet_2022-07-31_09-16-31'
        # self.data_loaders_model_name = 'Model_DSDNet_2022-07-27_23-14-06'
        # self.data_loaders_model_name = 'Model_DSDNet_2022-07-29_00-13-44'
        
        """
        Model options:
        - DSDNetV0
        - DSDNetV1
        - DSDNetV2
        - CIMEDNet
        - CIMEDNet_L
        """
        # self.model_type = 'DSDNetV0'
        self.model_type = 'DSDNetV1'
        # self.model_type = 'DSDNetV2'
        # self.model_type = 'CIMEDNet'
        # self.model_type = 'CIMEDNet_L'
        # self.model_type = 'PosNet'
        # self.model_type = 'DSDPosNet'
        
        if 'CIMEDNet' in self.model_type:
            self.data_loaders_model_name = 'Model_DSDNetV1_2022-12-05_19-37-02'

        # self.load_model_name = 'Model_DSDNetV2_2022-08-09_21-25-20'
        # self.data_loaders_model_name = 'Model_DSDNetV1_2022-09-11_15-58-06'

        self.memory_percentage_limit = 95

        self.model_param = ModelParameters(model_type = self.model_type)
        self.model_name = str(self.model_param.model).split("'")[1].split(".")[-1]
        # Model directory
        self.model_save_name = 'Model_'
        self.model_save_name += self.model_name
        self.model_save_name += '_' + self.init_time

        self.model_save_path = self.root_dir
        self.model_save_path += '/models/'
        self.model_save_path += self.model_name
        self.model_save_path += '/'
        self.model_save_path += self.model_save_name
        if not isdir(self.model_save_path): makedirs(self.model_save_path)
        self.log_writer_path = join(self.model_save_path, 'network_description.txt')

        # Optimizer parameters
        self.optimizer_type = 'adam'
        self.sdtw_gamma = 1e-4
        self.learning_rate = 1e-5
        self.eps = 5e-3
        self.weight_decay = None

        # Stopping conditions
        self.max_epoch = int(5e4)
        self.max_val_fail = 100
        self.loss_threshold = 1e-10
        
        # Training parameters
        self.validation_interval = 1
        self.model_save_interval = 50
        self.log_interval = 1
        self.plot_interval = None
        self.plot_num = 5

        # Data parameters
        self.batch_size = 1
        self.training_ratio = 7
        self.validation_ratio = 2
        self.test_ratio = 1
        self.includes_tau = 0

        # Processed parameters # No need to manually modify
        self.data_ratio = [self.training_ratio, self. validation_ratio, self.test_ratio]


    def writeLog(self, log):

        if log[0] == '\n':
            log = '\n' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S :: ") + log[1:]
        else:
            log = datetime.now().strftime("%Y-%m-%d_%H-%M-%S :: ") + log
        print(log)

        if self.log_writer_path != None:
            LOG_FILE = open(self.log_writer_path, 'a')
            LOG_FILE.write('\n' + log)
            LOG_FILE.close()

    def writeInitLog(self, model):
        self.writeLog('\nModel Name      : ' + self.model_save_name)
        self.writeLog('Network Created : ' + self.init_time)
        self.writeLog('Model           : ' + str(self.model_param.model))
        self.writeLog('Model Save Path : ' + self.model_save_path)
        self.writeLog('Model Type      : ' + self.model_type)
        self.writeLog('Data Path       : ' + self.dataset_path)
        self.writeLog('Data Limit      : ' + str(self.data_limit))
        self.writeLog('Batch Size      : ' + str(self.batch_size))
        self.writeLog('Layer Sizes     : ' + str(self.model_param.hidden_layer_sizes))
        self.writeLog(model.__str__())

class Conv2dParam:
    def __init__(self, out_channels, kernel_size, description = None):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.description = description

class ModelParameters:
    def __init__(self, model_type):
        self.model_type = model_type
        self.keys_to_normalize      = []

        # Network Parameters
        # self.image_dim              = (1, 100, 100)
        # self.image_dim              = (3, 100, 100)
        self.image_dim              = (3, 100, 100)
        self.dropout_prob           = 0.5

        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = 20), Conv2dParam(out_channels = 64, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (20, 80), description = 'width')],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (80, 20), description = 'height')]]

        # cutting
        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = 20), Conv2dParam(out_channels = 64, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
        #                           [Conv2dParam(out_channels = 32, kernel_size = (self.image_dim[1] - 10, 20), description = 'width')],
        #                           [Conv2dParam(out_channels = 32, kernel_size = (20, self.image_dim[2] - 10), description = 'height')]]
        # self.max_pool_size = 2

        # stacking

        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = 5), Conv2dParam(out_channels = 64, kernel_size = 3)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5), Conv2dParam(out_channels = 64, kernel_size = 3)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = 20), Conv2dParam(out_channels = 64, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5), Conv2dParam(out_channels = 64, kernel_size = 3)],
                                #   [Conv2dParam(out_channels = 128, kernel_size = 20), Conv2dParam(out_channels = 64, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
                                #   ]

        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = (self.image_dim[1] - 10, 20), description = 'width')],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (20, self.image_dim[2] - 10), description = 'height')]]
        # self.max_pool_size = 2

        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = (self.image_dim[1] - 10, 20), description = 'width')],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (20, self.image_dim[2] - 10), description = 'height')],
        #                           [Conv2dParam(out_channels = 256, kernel_size = (self.image_dim[2] - 5, self.image_dim[2] - 5), description = 'height')]]

        # self.conv_layer_params = [[Conv2dParam(out_channels = 32, kernel_size = 10)],
        #                           [Conv2dParam(out_channels = 32, kernel_size = 20)]]
                                #   [Conv2dParam(out_channels = 128, kernel_size = 10)]]

        # pepper

        self.conv_layer_params = [[Conv2dParam(out_channels = 64, kernel_size = 10), 
                                   Conv2dParam(out_channels = 64, kernel_size = 10), 
                                   Conv2dParam(out_channels = 64, kernel_size = 10)]]

        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = (self.image_dim[1] - 10, 20), description = 'width')],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (20, self.image_dim[2] - 10), description = 'height')]]
        self.max_pool_size = 2

        # Define hidden layers sizes (No need to define output layer size)
        # self.hidden_layer_sizes            = [4096, 4096, 4096, 2048, 256]
        # self.hidden_layer_sizes            = [2048, 2048, 2048, 1024, 128]
        # self.hidden_layer_sizes            = [2048, 2048, 2048, 1024, 512]
        # self.hidden_layer_sizes            = [2048, 1024, 512, 256, 128]
        # self.hidden_layer_sizes            = [1024, 1024, 1024, 1024, 1024, 512, 64]
        self.hidden_layer_sizes            = [1024, 1024, 1024, 512, 64]
        # self.hidden_layer_sizes            = [1024, 512, 256, 128, 64]
        # self.hidden_layer_sizes            = [512, 512, 512, 256, 64]
        # self.hidden_layer_sizes            = [512, 512, 512, 512, 512, 256, 64]
        # self.hidden_layer_sizes            = [1024, 1024, 1024]
        # self.hidden_layer_sizes            = [512, 512, 512]
        # self.hidden_layer_sizes            = [256, 256, 256]
        # self.hidden_layer_sizes            = [128, 128, 128]
        # self.hidden_layer_sizes            = [128, 64, 32]
        # self.hidden_layer_sizes            = [64, 64, 64]
        # self.hidden_layer_sizes            = [2048, 2048, 1024, 512, 256, 128, 64]
        # self.hidden_layer_sizes            = [512, 128]
        # self.hidden_layer_sizes            = [200, 50]
        # self.hidden_layer_sizes            = [100, 8]
        # self.hidden_layer_sizes            = [1600, 1500, 1000, 600, 200, 50]
        # self.hidden_layer_sizes            = [1600, 1500, 1000, 600, 200, 100]

        self.keys_to_normalize      = ['normal_dmp_y0', 
                                       'normal_dmp_goal', 
                                       'normal_dmp_w',
                                       'normal_dmp_L_y0', 
                                       'normal_dmp_L_goal', 
                                       'normal_dmp_L_w',
                                       'normal_dmp_tau',
                                       'segmented_dmp_observable_pos',
                                       'segmented_dmp_seg_num',
                                       'segmented_dmp_y0', 
                                       'segmented_dmp_goal', 
                                       'segmented_dmp_w', 
                                       'segmented_dmp_tau']
        
        if self.model_type == 'DSDNetV0':
            self.model                  = DSDNetV0

            self.input_mode             = ['image']
            self.output_mode            = ['segmented_dmp_y0', 'segmented_dmp_goal', 'segmented_dmp_w', 'segmented_dmp_tau']
            self.keys_to_normalize      = ['normal_dmp_y0', 'normal_dmp_goal', 'normal_dmp_w', 'segmented_dmp_y0', 'segmented_dmp_goal', 'segmented_dmp_w', 'segmented_dmp_tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments           = 27
            self.dmp_param              = DMPParameters(dof = 3, n_bf = 30, ay = 15)

        elif self.model_type == 'DSDNetV1':
            self.model                  = DSDNetV1

            self.input_mode             = ['image']
            self.loss_name              = ['seg', 'y0', 'goal', 'w', 'tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE', 'MSE']
            self.output_mode            = ['segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal', 
                                           'segmented_dmp_w', 
                                           'segmented_dmp_tau']
            
            self.backbone_option        = None
            # self.backbone_option        = 'keypointrcnn_resnet50_fpn'
            self.backbone_eval          = False

            ## Cutting dataset
            # self.max_segments           = 15
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 20, dt = 0.015, ay = 7)

            # self.max_segments           = 1
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 300, dt = 0.001, ay = 100)
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 1000, dt = 0.001, ay = 200)

            ## Stacking dataset
            # self.max_segments           = 24
            # self.max_segments           = 8
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 20, dt = 0.015, ay = 7)
            # self.dmp_param              = DMPParameters(dof = 4, n_bf = 2, dt = 0.024, ay = 3.4)
            # self.latent_w_size          = self.dmp_param.dof

            ## Pepper dataset
            self.max_segments           = 4
            # self.max_segments           = 8
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 20, dt = 0.015, ay = 7)
            self.dmp_param              = DMPParameters(dof = 3, n_bf = 50, dt = 0.003, ay = 15)
            self.latent_w_size          = self.dmp_param.dof

            # self.decoder_layer_sizes    = [64, 512, 1024, 1024, 1024]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 2048, 4096, 4096, 4096]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048, 2048]
            self.decoder_layer_sizes    = [256, 256, 256]
            # self.decoder_layer_sizes    = [128, 128, 128]
            # self.decoder_layer_sizes    = [64, 64, 64]
            # self.decoder_layer_sizes    = [256, 512, 1024]
            # self.keys_to_normalize      = self.output_mode
        
        elif self.model_type in ['CIMEDNet', 'CIMEDNet_L']:
            self.model              = CIMEDNet

            self.input_mode         = ['image']
                                           
            if self.model_type == 'CIMEDNet':
                self.output_mode    = ['normal_dmp_y0',
                                       'normal_dmp_goal',
                                       'normal_dmp_w',
                                       'normal_dmp_tau']
                # self.dmp_param      = DMPParameters(dof = 2, n_bf = 300, dt = 0.001, ay = 100)
                # self.dmp_param      = DMPParameters(dof = 3, n_bf = 250, dt = 0.001, ay = 25)
                self.dmp_param      = DMPParameters(dof = 3, n_bf = 200, dt = 0.001, ay = 25)
            elif self.model_type == 'CIMEDNet_L':
                self.output_mode    = ['normal_dmp_L_y0',
                                       'normal_dmp_L_goal',
                                       'normal_dmp_L_w',
                                       'normal_dmp_L_tau']
                # self.dmp_param      = DMPParameters(dof = 2, n_bf = 1000, dt = 0.001, ay = 200)
                # self.dmp_param      = DMPParameters(dof = 3, n_bf = 1000, dt = 0.001, ay = 250)
                self.dmp_param      = DMPParameters(dof = 3, n_bf = 1000, dt = 0.001, ay = 200)
                                        
            self.loss_name          = ['y0', 'goal', 'w', 'tau']
            self.loss_type          = ['MSE', 'MSE', 'MSE', 'MSE']

        elif self.model_type == 'DSDNetV2':
            self.model                  = DSDNetV2

            self.input_mode             = ['image']
            self.loss_name              = ['pos', 'seg', 'y0', 'goal', 'w', 'tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE', 'MSE', 'MSE']
            self.output_mode            = ['segmented_dmp_observable_pos',
                                           'segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal', 
                                           'segmented_dmp_w', 
                                           'segmented_dmp_tau']

            ## Cutting dataset
            # self.max_segments           = 15
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 20, dt = 0.015, ay = 7)

            # self.max_segments           = 1
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 300, dt = 0.001, ay = 100)
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 1000, dt = 0.001, ay = 200)

            ## Stacking dataset
            self.max_segments           = 24
            self.max_observable_pos     = 4
            # self.max_segments           = 8
            self.dmp_param              = DMPParameters(dof = 4, n_bf = 2, dt = 0.04, ay = 3.4)
            self.latent_w_size          = self.dmp_param.dof

            # self.decoder_layer_sizes    = [64, 512, 1024, 1024, 1024]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 2048, 4096, 4096, 4096]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 256, 256]
            # self.decoder_layer_sizes    = [128, 128, 128]
            self.decoder_layer_sizes    = [64, 64, 64]
            # self.decoder_layer_sizes    = [256, 512, 1024]
            # self.keys_to_normalize      = self.output_mode

        elif self.model_type == 'DSDNetPosFineTuning':
            self.model                  = DSDNetV1
            
            # self.backbone_option        = None
            self.backbone_option        = 'keypointrcnn_resnet50_fpn'
            self.backbone_eval                   = False

            self.input_mode             = ['image']
            self.loss_name              = ['seg', 'y0', 'goal']
            self.loss_type              = ['MSE', 'MSE', 'MSE']
            self.output_mode            = ['segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal']
            self.max_segments           = 3

        elif self.model_type == 'PosNet':
            self.model              = PosNet

            self.input_mode         = ['image']
            self.output_mode        = ['segmented_dmp_observable_pos']
            self.loss_name          = ['pos']
            self.loss_type          = ['MSE']

            self.max_observable_pos     = 3
            self.dmp_param              = DMPParameters(dof = 4, n_bf = 2, dt = 0.04, ay = 3.4)

        elif self.model_type == 'DSDPosNet':
            self.model              = DSDPosNet

            self.load_pos_net_dir   = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/PosNet'
            self.load_pos_net_name  = 'Model_PosNet_2022-08-11_20-21-46'
            self.load_pos_net_path  = join(self.load_pos_net_dir, self.load_pos_net_name)
            self.load_pos_net_train_param = pkl.load(open(join(self.load_pos_net_path, 'train-model-dmp_param.pkl'), 'rb'))
            self.pos_net = PosNet(self.load_pos_net_train_param)
            self.pos_net.load_state_dict(torch.load(join(self.load_pos_net_path, 'best_net_parameters')))
            self.pos_net.eval()

            self.input_mode         = ['image']
            self.output_mode        = ['segmented_dmp_y0',
                                       'segmented_dmp_goal']
            self.loss_name          = ['y0', 'goal']
            self.loss_type          = ['MSE', 'MSE']

            self.max_observable_pos     = 3
            self.max_segments           = 16
            self.dmp_param              = DMPParameters(dof = 4, n_bf = 2, dt = 0.04, ay = 3.4)


        else:
            raise ValueError('Wrong network configuration input')

class DMPParameters:
    def __init__(self, dof = None, n_bf = None, dt = None, ay = None, by = None, tau = None):
        self.dof            = 3 if dof == None else dof
        # self.n_bf           = 3 if n_bf == None else n_bf
        self.n_bf           = 30 if n_bf == None else n_bf
        self.dt             = .02 if dt == None else dt
        self.tau            = 1. if tau == None else tau

        # Canonical System Parameters
        self.cs_runtime     = 1.0
        self.cs_ax          = 1.0

        # Dynamical System Parameters
        # self.ay             = 2.8 if ay == None else ay
        self.ay             = 7 if ay == None else ay
        self.by             = (self.ay / 4) if by == None else by

        self.timesteps      = int(self.cs_runtime / self.dt)
        
#%% Test
# if __name__=='__main__':
#     train_param = TrainingParameters()
#     dmp_dict = torch.from_numpy(train_param.model_param.dmp_param.dmp_dict)
#     test_indices = torch.randint(0, 5, (20,))
    