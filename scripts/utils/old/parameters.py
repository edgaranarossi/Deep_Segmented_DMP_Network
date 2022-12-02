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
        self.root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
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

        self.dataset_dir = join(self.root_dir, 'data/pkl/stacking')
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
        self.dataset_name = 'stacking_[1, 2, 3, 4, 5].num_data_500_num_seg_40.normal_dmp_bf_80_ay_25_dt_0.001.seg_dmp_bf_2_ay_3.4_dt_0.04_init_pos_fixed_target_pos_rand_2022-08-08_23-10-40.pkl'

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
        
        self.model_type = 'dsdnet'
        # self.model_type = 'cimednet'
        # self.model_type = 'cimednet-accurate'

        if self.model_type != 'dsdnet':
            self.data_loaders_model_name = 'Model_DSDNet_2022-08-08_19-46-00'

        self.load_model_name = 'Model_DSDNet_2022-08-08_19-46-00'
        # self.data_loaders_model_name = self.load_model_name

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
        self.max_epoch = int(5e3)
        self.max_val_fail = 50
        self.loss_threshold = 1e-10
        
        # Training parameters
        self.validation_interval = 1
        # self.model_save_interval = 50
        self.log_interval = 1
        if self.model_param.network_configuration in ['1', '2']:
            self.plot_interval = 5
        else:
            self.plot_interval = 5
        self.plot_num = 5

        # Data parameters
        # self.batch_size = 50
        self.batch_size = 10
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
        self.image_dim              = (3, 150, 150)
        self.dropout_prob           = 0.0

        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = 20), Conv2dParam(out_channels = 64, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (20, 80), description = 'width')],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (80, 20), description = 'height')]]

        # Network Parameters
        # self.image_dim              = (1, 37, 90)
        # self.dropout_prob           = 0.0

        # cutting
        self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
                                  [Conv2dParam(out_channels = 128, kernel_size = 20), Conv2dParam(out_channels = 64, kernel_size = 10), Conv2dParam(out_channels = 64, kernel_size = 5)],
                                  [Conv2dParam(out_channels = 32, kernel_size = (self.image_dim[1] - 10, 20), description = 'width')],
                                  [Conv2dParam(out_channels = 32, kernel_size = (20, self.image_dim[2] - 10), description = 'height')]]

        # stacking
        # self.conv_layer_params = [[Conv2dParam(out_channels = 128, kernel_size = (self.image_dim[1] - 10, 20), description = 'width')],
        #                           [Conv2dParam(out_channels = 128, kernel_size = (20, self.image_dim[2] - 10), description = 'height')],
        #                           [Conv2dParam(out_channels = 256, kernel_size = (self.image_dim[2] - 5, self.image_dim[2] - 5), description = 'height')]]

        # self.conv_layer_params = [[Conv2dParam(out_channels = 64, kernel_size = 3), 
        #                            Conv2dParam(out_channels = 64, kernel_size = 3), 
        #                            Conv2dParam(out_channels = 128, kernel_size = 3), 
        #                            Conv2dParam(out_channels = 128, kernel_size = 3)]]

        # Define hidden layers sizes (No need to define output layer size)
        # self.hidden_layer_sizes            = [4096, 4096, 4096, 2048, 256]
        # self.hidden_layer_sizes            = [2048, 2048, 2048, 1024, 128]
        # self.hidden_layer_sizes            = [2048, 2048, 2048, 1024, 512]
        # self.hidden_layer_sizes             = [1024, 1024, 1024, 1024, 1024, 512, 64]
        # self.hidden_layer_sizes             = [1024, 1024, 1024, 512, 64]
        # self.hidden_layer_sizes             = [512, 512, 512, 256, 64]
        # self.hidden_layer_sizes             = [512, 512, 512, 512, 512, 256, 64]
        # self.hidden_layer_sizes            = [1024, 512, 256, 128, 64]
        # self.hidden_layer_sizes            = [1024, 1024, 1024]
        # self.hidden_layer_sizes            = [256, 256, 256]
        # self.hidden_layer_sizes            = [128, 128, 128]
        # self.hidden_layer_sizes            = [128, 64, 32]
        self.hidden_layer_sizes            = [64, 64, 64]
        # self.hidden_layer_sizes            = [2048, 2048, 1024, 512, 256, 128, 64]
        # self.hidden_layer_sizes            = [512, 128]
        # self.hidden_layer_sizes            = [200, 50]
        # self.hidden_layer_sizes            = [100, 8]
        # self.hidden_layer_sizes            = [1600, 1500, 1000, 600, 200, 50]
        # self.hidden_layer_sizes            = [1600, 1500, 1000, 600, 200, 100]

        self.network_configuration  = '17'
        """
        Network configurations:
        1: CNNDMPNet (outputs scaled)
        2: CNNDMPNet (outputs unscaled)
        3: CNNDMPNet (trajectory)
        4: FixedSegmentDictDMPNet
        5: DynamicSegmentDictDMPNet
        6: SegmentNumCNN
        7: SegmentDictionaryDMPNet
        8: FixedSegmentDictDMPNet
        9: DynamicParameterDMPNet
        10: FirstStageCNN
        11: LSTMRemainingSegments
        12: SegmentPosNet
        13: SecondStageDMPWeightsLSTM
        14: SegmentWeightNet
        15: SegmentDMPCNN
        16: CNNDeepDMP
        17: DSDNet
        18: SegmentedDMPJoinedNetwork
        19: NormalDMPJoinedNetwork
        20: PositionNetwork
        21: SegmentedDMPNetworkV2
        22: SDDMPsTrajLoss
        """

        if self.network_configuration == '1':
            self.model = CNNDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['dmp_y0_goal_w_scaled']
            self.loss_type              = ['MSE']

        elif self.network_configuration == '2':
            self.model = CNNDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['dmp_y0_goal_w']
            self.loss_type              = ['MSE']

        elif self.network_configuration == '3':
            self.model = CNNDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['normal_dmp_y_track']
            self.loss_type              = ['DMPIntegrationMSE']

        elif self.network_configuration == '4':
            self.model = FixedSegmentDictDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['traj_interpolated']
            self.loss_type              = ['SDTW']

            self.max_segments           = 8
            self.dmp_param              = DMPParameters()
            self.dmp_dict_param         = DMPDictionaryParameters(self.dmp_param)

        elif self.network_configuration == '5':
            self.model = DynamicSegmentDictDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['points_padded', 'num_segments', 'segment_types_padded']
            self.loss_type              = ['MSE', 'MSE', 'MSE']

            self.max_segments           = 8
            self.dmp_param              = DMPParameters()
            self.dmp_dict_param         = DMPDictionaryParameters(self.dmp_param)

        elif self.network_configuration == '6':
            self.model = SegmentNumCNN

            self.input_mode             = ['image']
            self.output_mode            = ['num_segments']
            self.loss_type              = ['MSE']
            
            self.max_segments           = 8
            self.dmp_param              = DMPParameters()
            self.dmp_dict_param         = DMPDictionaryParameters(self.dmp_param)

        elif self.network_configuration == '7':
            self.model = SegmentDictionaryDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['normal_dmp_y_track', 'normal_dmp_dy_track']
            self.loss_type              = ['SDTW', 'SDTW']
            
            self.dictionary_size        = 5
            self.max_segments           = 10

        elif self.network_configuration == '8':
            self.model = FixedSegmentDictDMPNet

            self.input_mode             = ['image']
            self.output_mode            = ['traj_interpolated']
            self.loss_type              = ['SDTW']

            self.max_segments           = 35
            self.dmp_param              = DMPParameters()
            self.base_traj_param        = BaseTrajectoryParameters(dmp_param = self.dmp_param)
            self.traj_length            = int(1/self.base_traj_param.dmp_dt)
            # self.traj_length            = 200

        elif self.network_configuration == '9':
            self.model = DynamicParameterDMPNet
            # self.dynamical_model_dropout_prob = 0.2

            self.input_mode             = ['image']
            self.output_mode            = ['num_segments', 'y0', 'goal_w']
            self.loss_type              = ['MSE', 'MSE', 'MSE']

            self.max_segments           = 10
            self.connect_segments       = True
            self.dmp_param              = DMPParameters()

            self.dynamical_model_type   = 'LSTMNet'
            self.dynamical_model_dropout_prob = self.dropout_prob

            if self.dynamical_model_type == 'AutoEncoderNet':
                self.dynamical_model = AutoEncoderNet
                self.dynamical_model_hidden_layers = [1024, 1024, 50, 50, 1024, 1024, ]
                # self.dynamical_model_hidden_layers = [20,200,600,1000,600,200,200,600,1000,600,200,20]
                # self.dynamical_model_hidden_layers = [1600, 1500, 1000, 600, 200, 20, 20, 200, 600, 1000, 1500, 1600]
            elif self.dynamical_model_type == 'LSTMNet':
                self.pre_lstm_hidden_sizes = [512, 512]

                self.dynamical_model = LSTMNet
                self.num_layers = 2
                self.hidden_size = 512
                self.seq_length = 1

        elif self.network_configuration == '10':
            self.model = FirstStageCNN

            self.input_mode             = ['image']
            self.output_mode            = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'first_segment_w']
            self.keys_to_normalize      = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments           = 15
            self.dmp_param              = DMPParameters()

        elif self.network_configuration == '11':
            self.model                      = LSTMRemainingSegments

            self.cnn_model_dir              = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/FirstStageCNN/Model_FirstStageCNN_2022-03-17_15-32-00'
            self.cnn_model_train_param_path = join(self.cnn_model_dir, 'train-model-dmp_param.pkl')
            self.cnn_model_train_param      = pkl.load(open(self.cnn_model_train_param_path, 'rb'))
            self.cnn_model                  = self.cnn_model_train_param.model_param.model(self.cnn_model_train_param)
            self.cnn_model_parameter_path   = join(self.cnn_model_dir, 'best_net_parameters')
            self.cnn_model.load_state_dict(torch.load(self.cnn_model_parameter_path))

            self.input_mode                 = self.cnn_model_train_param.model_param.input_mode
            self.output_mode                = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            self.keys_to_normalize          = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            self.loss_type                  = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments               = self.cnn_model_train_param.model_param.max_segments
            self.dmp_param                  = self.cnn_model_train_param.model_param.dmp_param

            self.lstm_goal_state_size       = self.dmp_param.dof
            self.lstm_w_state_size          = self.dmp_param.dof + (self.dmp_param.dof * self.dmp_param.n_bf)
            self.lstm_goal_hidden_size      = 1024
            self.lstm_w_hidden_size         = 256
            self.lstm_goal_num_layer        = 4
            self.lstm_w_num_layer           = 2
            self.pre_lstm_goal_hidden_size  = [1024, 1024, 1024]
            self.pre_lstm_w_hidden_size     = [128, 128, 128]

        elif self.network_configuration == '12':
            self.model                      = SegmentPosNet

            self.input_mode             = ['image']
            self.output_mode            = ['dmp_y0_segments', 'dmp_goal_segments']
            self.keys_to_normalize      = ['dmp_y0_segments', 'dmp_goal_segments']
            self.loss_type              = ['MSE', 'MSE']

            self.max_segments           = 15
            self.dmp_param              = DMPParameters()

        elif self.network_configuration == '13':
            self.model                      = SecondStageDMPWeightsLSTM

            self.cnn_model_dir              = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/FirstStageCNN/'
            self.model_name                 = 'Model_FirstStageCNN_2022-03-22_15-52-46'
            self.cnn_model_path             = join(self.cnn_model_dir, self.model_name)
            self.cnn_model_train_param_path = join(self.cnn_model_path, 'train-model-dmp_param.pkl')
            self.cnn_model_train_param      = pkl.load(open(self.cnn_model_train_param_path, 'rb'))
            self.cnn_model                  = self.cnn_model_train_param.model_param.model(self.cnn_model_train_param)
            self.cnn_model_parameter_path   = join(self.cnn_model_path, 'best_net_parameters')
            self.cnn_model.load_state_dict(torch.load(self.cnn_model_parameter_path))
            # self.cnn_model.eval()

            self.input_mode                 = self.cnn_model_train_param.model_param.input_mode
            self.output_mode                = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            self.keys_to_normalize          = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            self.loss_type                  = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments               = self.cnn_model_train_param.model_param.max_segments
            self.dmp_param                  = self.cnn_model_train_param.model_param.dmp_param

            self.lstm_w_state_size          = self.dmp_param.dof + (self.dmp_param.dof * self.dmp_param.n_bf)
            self.lstm_w_hidden_size         = 512
            self.lstm_w_num_layer           = 2
            self.pre_lstm_w_hidden_size     = [256, 256, 256]
            self.pre_lstm_w_hidden_size     = [self.lstm_w_state_size] + self.pre_lstm_w_hidden_size

        elif self.network_configuration == '14':
            self.model                      = SegmentWeightNet

            self.input_mode             = ['image']
            self.output_mode            = ['dmp_w_segments']
            self.keys_to_normalize      = ['dmp_w_segments']
            self.loss_type              = ['MSE']

            self.max_segments           = 15
            self.dmp_param              = DMPParameters()

        elif self.network_configuration == '15':
            self.model                  = SegmentDMPCNN

            self.input_mode             = ['image']
            # self.output_mode            = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            # self.keys_to_normalize      = ['num_segments', 'dmp_y0_segments', 'dmp_goal_segments', 'dmp_w_segments']
            # self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']
            self.output_mode            = ['segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal', 
                                           'segmented_dmp_w', 
                                           'segmented_dmp_tau']
            self.keys_to_normalize      = ['segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal', 
                                           'segmented_dmp_w', 
                                           'segmented_dmp_tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments           = 11
            self.dmp_param              = DMPParameters(dof = 3, n_bf = 30)

        elif self.network_configuration == '16':
            self.model                  = CNNDeepDMP

            self.input_mode             = ['image']
            self.output_mode            = ['dmp_y0_normal', 'dmp_goal_normal', 'dmp_w_normal']
            self.keys_to_normalize      = ['dmp_y0_normal', 'dmp_goal_normal', 'dmp_w_normal']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments           = 1
            self.dmp_param              = DMPParameters(n_bf = 540, dt = 0.001, ay = 25)
        
        elif self.network_configuration == '17':
            self.model                  = DSDNet

            self.input_mode             = ['image']
            # cutting
            # self.keys_to_normalize      = ['normal_dmp_goal', 
            #                                'normal_dmp_w',
            #                                'normal_dmp_L_goal', 
            #                                'normal_dmp_L_w',
            #                                'segmented_dmp_seg_num',
            #                                'segmented_dmp_y0', 
            #                                'segmented_dmp_goal', 
            #                                'segmented_dmp_w']
            # stacking
            self.keys_to_normalize      = ['normal_dmp_y0', 
                                           'normal_dmp_goal', 
                                           'normal_dmp_w',
                                           'normal_dmp_L_y0', 
                                           'normal_dmp_L_goal', 
                                           'normal_dmp_L_w',
                                           'segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal', 
                                           'segmented_dmp_w']
            # self.keys_to_normalize      = ['segmented_dmp_w',
            #                                'segmented_dmp_tau']
            # self.keys_to_normalize      = ['segmented_dmp_y0', 
            #                                'segmented_dmp_goal', 
            #                                'segmented_dmp_w', 
            #                                'segmented_dmp_tau']
            self.loss_name              = ['seg', 'y0', 'goal', 'w', 'tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE', 'MSE']

            ## Cutting dataset
            # self.max_segments           = 15
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 20, dt = 0.015, ay = 7)

            # self.max_segments           = 1
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 300, dt = 0.001, ay = 100)
            # self.dmp_param              = DMPParameters(dof = 2, n_bf = 1000, dt = 0.001, ay = 200)

            ## Stacking dataset
            if self.model_type == 'dsdnet':
                self.max_segments           = 40
                # self.max_segments           = 8
                self.dmp_param              = DMPParameters(dof = 4, n_bf = 2, dt = 0.04, ay = 3.4)
            elif self.model_type == 'cimednet':
                self.max_segments           = 1
                self.dmp_param              = DMPParameters(dof = 4, n_bf = 80, dt = 0.001, ay = 25)
                # self.dmp_param              = DMPParameters(dof = 4, n_bf = 48, dt = 0.001, ay = 25)
            elif self.model_type == 'cimednet-accurate':
                self.max_segments           = 1
                self.dmp_param              = DMPParameters(dof = 4, n_bf = 1000, dt = 0.001, ay = 200)

            ## Real cutting dataset
            # self.max_segments           = 5
            # self.dmp_param              = DMPParameters(dof = 3, n_bf = 50, dt = 0.005, ay = 15)

            # self.max_segments           = 1
            # self.dmp_param              = DMPParameters(dof = 3, n_bf = 250, dt = 0.001, ay = 25)

            # self.max_segments           = 1
            # self.dmp_param              = DMPParameters(dof = 3, n_bf = 800, dt = 0.001, ay = 250)

            # self.decoder_layer_sizes    = [64, 512, 1024, 1024, 1024]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 2048, 4096, 4096, 4096]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 256, 256]
            self.decoder_layer_sizes    = [128, 128, 128]
            # self.decoder_layer_sizes    = [64, 64, 64]
            # self.decoder_layer_sizes    = [256, 512, 1024]

            if self.max_segments == 1:
                self.latent_w_size          = 32
                if self.model_type == 'cimednet':
                    self.output_mode    = ['normal_dmp_seg_num',
                                           'normal_dmp_y0',
                                           'normal_dmp_goal',
                                           'normal_dmp_w',
                                           'normal_dmp_tau']
                elif self.model_type == 'cimednet-accurate':
                    self.output_mode    = ['normal_dmp_seg_num',
                                           'normal_dmp_L_y0',
                                           'normal_dmp_L_goal',
                                           'normal_dmp_L_w',
                                           'normal_dmp_L_tau']
            else:
                self.latent_w_size          = self.dmp_param.dof
                self.output_mode            = ['segmented_dmp_seg_num',
                                               'segmented_dmp_y0', 
                                               'segmented_dmp_goal', 
                                               'segmented_dmp_w', 
                                               'segmented_dmp_tau']
            self.evaluate               = self.output_mode
            # self.keys_to_normalize      = self.output_mode
        
        elif self.network_configuration == '18':
            self.model                  = SegmentedDMPJoinedNetwork

            self.input_mode             = ['image']
            self.output_mode            = ['segmented_dmp_y0', 'segmented_dmp_goal', 'segmented_dmp_w', 'segmented_dmp_tau']
            self.keys_to_normalize      = ['normal_dmp_y0', 'normal_dmp_goal', 'normal_dmp_w', 'segmented_dmp_y0', 'segmented_dmp_goal', 'segmented_dmp_w', 'segmented_dmp_tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments           = 27
            self.dmp_param              = DMPParameters(dof = 3, n_bf = 30, ay = 15)
        
        elif self.network_configuration == '19':
            self.model                  = NormalDMPJoinedNetwork

            self.input_mode             = ['image']
            self.output_mode            = ['normal_dmp_y0', 'normal_dmp_goal', 'normal_dmp_w', 'normal_dmp_tau']
            self.evaluate               = self.output_mode
            self.keys_to_normalize      = ['normal_dmp_y0', 'normal_dmp_goal', 'normal_dmp_w', 'segmented_dmp_y0', 'segmented_dmp_goal', 'segmented_dmp_w', 'segmented_dmp_tau']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']

            self.dmp_param              = DMPParameters(dof = 3, n_bf = 510)
        
        elif self.network_configuration == '20':
            self.model                  = PositionNetwork

            self.input_mode             = ['seg_num', 'pos_y0', 'image']
            self.output_mode            = ['pos_goal',
                                           'segmented_dmp_seg_num',
                                           'dmp_y0',
                                           'segmented_dmp_w', 
                                           'segmented_dmp_tau']
            self.evaluate               = ['pos_goal']
            self.keys_to_normalize      = ['segmented_dmp_seg_num',
                                           'segmented_dmp_w']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE', 'MSE']

            self.dmp_param              = DMPParameters(dof = 3, n_bf = 20, dt = 0.005, ay = 15)
        
        elif self.network_configuration == '21':
            self.model                  = SegmentedDMPNetworkV2

            self.input_mode             = ['image']
            self.output_mode            = ['segmented_dmp_seg_num',
                                           'dmp_y0',
                                           'segmented_dmp_w', 
                                           'segmented_dmp_tau']
            self.evaluate               = self.output_mode
            self.keys_to_normalize      = ['segmented_dmp_seg_num',
                                           'segmented_dmp_w']
            self.loss_type              = ['MSE', 'MSE', 'MSE', 'MSE']

            self.max_segments           = 18
            self.dmp_param              = DMPParameters(dof = 3, n_bf = 20, dt = 0.005, ay = 15)

            self.latent_w_size          = 1
            self.decoder_layer_sizes    = [256, 512, 1024, 1024, 1024]
        
        elif self.network_configuration == '22':
            self.model                  = SDDMPsTrajLoss

            self.input_mode             = ['image']
            self.output_mode            = ['segmented_dmp_seg_num',
                                           'segmented_dmp_target_trajectory']
            self.keys_to_normalize      = ['normal_dmp_goal', 
                                           'normal_dmp_w',
                                           'segmented_dmp_seg_num',
                                           'segmented_dmp_y0', 
                                           'segmented_dmp_goal', 
                                           'segmented_dmp_w']
            self.loss_type              = ['MSE', 'MSE']
            self.evaluate               = self.output_mode
            self.max_segments           = 5
            self.dmp_param              = DMPParameters(dof = 3, n_bf = 50, dt = 0.005, ay = 15)

            # self.max_segments           = 1
            # self.dmp_param              = DMPParameters(dof = 3, n_bf = 540, dt = 0.001, ay = 25)

            self.latent_w_size          = self.dmp_param.dof
            self.decoder_layer_sizes    = [64, 512, 1024, 1024, 1024]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 2048, 4096, 4096, 4096]
            # self.decoder_layer_sizes    = [512, 1024, 2048, 2048, 2048, 2048]
            # self.decoder_layer_sizes    = [256, 256, 256]
            # self.decoder_layer_sizes    = [128, 128, 128]
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

class DMPDictionaryParameters:
    def __init__(self, dmp_param):
        self.n_bf = dmp_param.n_bf
        self.ay = np.array(dmp_param.ay)
        self.dt = dmp_param.dt

        dict_trajectories   = [
                            #    [[0.0, 0.0],
                            #     [1.0, 1.0]], # 0: Straight line
                              #  [[0.0, 0.0],
                              #   [1.0, 0.0],
                              #   [1.0, 1.0]], # 1: Diagonal curve bottom
                              #  [[0.0, 0.0],
                              #   [0.0, 1.0],
                              #   [1.0, 1.0]], # 2: Diagonal curve top
                               [[0.0, 0.0],
                                [0.0, 1.0],
                                [1.0, 1.0],
                                [1.0, 0.0]], # 3: Curve horizontal
                              #  [[0.0, 0.0],
                              #   [1.0, 0.0],
                              #   [1.0, 1.0],
                              #   [0.0, 1.0]], # 4: Curve vertical
                              ]

        self.dmp_dict      = []
        for dict_trajectory in dict_trajectories:
            traj = np.array(dict_trajectory)
            dmp = DMPs_discrete(n_dmps = traj.shape[-1],
                                n_bfs=self.n_bf,
                                ay = np.ones(traj.shape[-1]) * self.ay,
                                dt = self.dt)
            dmp.imitate_path(traj.T)
            y, _, _ = dmp.rollout()
            # self.dmp_dict.append((traj, y, dmp))
            self.dmp_dict.append(y)
        
        self.segment_traj_length = int(1 / self.dt)
        # self.dmp_dict = np.array(dmp_dict, dtype = [('traj',object),('dmp_traj',object),('dmp',DMPs_discrete)])
        self.dmp_dict = torch.from_numpy(np.array(self.dmp_dict)).to(DEVICE)

class BaseTrajectoryParameters:
    def __init__(self, dmp_param):
        # traj = [[0.0, 0.0],
        #         [0.0, 1.0],
        #         [1.0, 1.0],
        #         [1.0, 0.0]]
        # traj = [[0.0, 0.0],
        #         [1.0, 1.0]]
        traj = [[0.0, 0.0],
                [1.0, 0.0]]
        traj_np = np.array(traj)

        dmp_bf = dmp_param.n_bf
        dmp_ay = dmp_param.ay
        self.dmp_dt = dmp_param.dt

        dmp = DMPs_discrete(n_dmps = traj_np.shape[-1],
                            n_bfs = dmp_bf,
                            ay = np.ones(traj_np.shape[-1]) * dmp_ay,
                            dt = self.dmp_dt)
        dmp.imitate_path(traj_np.T)
        y_track, _, _ = dmp.rollout()

        traj_np = y_track

        dmp_bf = 100
        dmp_ay = 20

        dmp = DMPs_discrete(n_dmps = traj_np.shape[-1],
                            n_bfs = dmp_bf,
                            ay = np.ones(traj_np.shape[-1]) * dmp_ay,
                            dt = self.dmp_dt)
        dmp.imitate_path(traj_np.T)
        y_track, dy_track, ddy_track = dmp.rollout()

        self.base_traj = from_numpy(y_track.reshape(1, y_track.shape[0], y_track.shape[1])).float().to(DEVICE)
        # self.base_traj[0,0,:] = torch.tensor([0., 0.])
        # self.base_traj[0,-1,:] = torch.tensor([1., 0.])
        
#%% Test
# if __name__=='__main__':
#     train_param = TrainingParameters()
#     dmp_dict = torch.from_numpy(train_param.model_param.dmp_param.dmp_dict)
#     test_indices = torch.randint(0, 5, (20,))
    