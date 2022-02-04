import torch
from torch import ones
import numpy as np
from pydmps.dmp_discrete import DMPs_discrete
from os.path import join, isdir
from os import makedirs
from datetime import datetime
from utils.networks import CNNDMPNet, NewCNNDMPNet, FixedSegmentDictDMPNet, DynamicSegmentDictDMPNet, SegmentNumCNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingParameters:
    def __init__(self):
        self.root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
        self.init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Dataset parameter
        self.dataset_dir = join(self.root_dir, 'data/pkl/shapes-8')
        self.dataset_name = 'image-dict_output-traj_N_100000_n-bf_5_ay_4_dt_0_2022-02-03_22-04-03.pkl'
        self.dataset_path = join(self.dataset_dir,  self.dataset_name)

        self.model_param = ModelParameters()
        self.model_name = str(self.model_param.model).split("'")[1].split(".")[-1]
        # Model directory
        self.model_save_path = self.root_dir
        self.model_save_path += '/models/'
        self.model_save_path += self.model_name
        self.model_save_path += '/Model_'
        self.model_save_path += self.model_name
        self.model_save_path += '_' + self.init_time
        if not isdir(self.model_save_path): makedirs(self.model_save_path)
        self.log_writer_path = join(self.model_save_path, 'network_description.txt')

        # Optimizer parameters
        self.optimizer_type = 'adam'
        self.sdtw_gamma = 1e-3
        self.learning_rate = 1e-3
        self.eps = 5e-3
        self.weight_decay = None

        # Training parameters
        self.max_epoch = None
        self.max_val_fail = 500
        self.validation_interval = 1
        self.log_interval = 1
        if self.model_param.model in [CNNDMPNet, SegmentNumCNN]:
            self.plot_interval = None
        else:
            self.plot_interval = 1
        self.plot_num = 5

        # Data parameters
        self.batch_size = 25
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
        self.writeLog('Network created: ' + self.init_time)
        self.writeLog('Model : ' + str(self.model_param.model))
        self.writeLog(model.__str__())
        self.writeLog('Data Path : ' + self.dataset_path)
        self.writeLog('Model Save Path : ' + self.model_save_path)
        self.writeLog('Layer Sizes : ' + str(self.model_param.layer_sizes))

class ModelParameters:
    def __init__(self):
        """
        output_mode:
        'dmp' : Use old loss function
        'traj' : Use new loss function which compares trajectory
        """
        # Network Parameters
        self.input_mode = ['image']  
        self.output_mode = ['dmp_y0_goal_w_scaled']
        # self.output_mode = ['segmented_dict_dmp_outputs', 'num_segments', 'segmented_dict_dmp_types']
        # self.output_mode = ['num_segments']
        self.image_dim = (3, 150, 150)

        # Define hidden layers sizes (No need to define output layer size)
        # self.layer_sizes = [4096, 2048, 2048]
        # self.layer_sizes = [2048, 2048, 2048]
        # self.layer_sizes = [1024, 1024, 1024]
        # self.layer_sizes = [128, 128, 128]
        self.layer_sizes = [2048, 2048, 1024, 512, 256, 128, 64]
        # self.layer_sizes = [200, 50]
        self.dropout_prob = 0.3

        self.dmp_param = DMPParameters()

        if sorted(self.input_mode) == sorted(['image']) and \
           sorted(self.output_mode) == sorted(['dmp_y0_goal_w_scaled']) and\
           self.dmp_param.segments == None :
            self.model = CNNDMPNet
            self.loss_type = ['MSE']
        elif sorted(self.input_mode) == sorted(['image']) and \
           sorted(self.output_mode) == sorted(['dmp_y0_goal_w_unscaled']) and\
           self.dmp_param.segments == None :
            self.model = CNNDMPNet
            self.loss_type = ['MSE']
        elif sorted(self.input_mode) == sorted(['image']) and \
             sorted(self.output_mode) == sorted(['traj_interpolated']) and \
             self.dmp_param.segments == None :
            self.model = NewCNNDMPNet
            self.loss_type = ['MSE']
        elif sorted(self.input_mode) == sorted(['image']) and \
             sorted(self.output_mode) == sorted(['traj_interpolated']) and \
             self.dmp_param.segments != None :
            self.model = FixedSegmentDictDMPNet
            self.loss_type = ['SDTW']
        elif sorted(self.input_mode) == sorted(['image']) and \
             sorted(self.output_mode) == sorted(['segmented_dict_dmp_outputs', 'num_segments', 'segmented_dict_dmp_types']) and \
             self.dmp_param.segments != None :
            self.model = DynamicSegmentDictDMPNet
            self.loss_type = ['MSE', 'MSE', 'MSE']
        elif sorted(self.input_mode) == sorted(['image']) and \
             sorted(self.output_mode) == sorted(['num_segments']) and \
             self.dmp_param.segments != None :
            self.model = SegmentNumCNN
            self.loss_type = ['MSE']
        else:
            raise ValueError('Wrong input-output-network configuration')

        ## Processed parameters # No need to manually modify
        self.dmp_param.dof = len(self.image_dim) - 1
        if self.dmp_param.segments != None:
            self.dmp_param.ay = ones(self.dmp_param.segments, self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        else:
            self.dmp_param.ay = ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        if self.dmp_param.by == None:
            self.dmp_param.by = self.dmp_param.ay / 4
        else:
            ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.by

        if self.model == CNNDMPNet and self.dmp_param.segments == None:
            """
            Calculate output layer size and add it to self.layer_sizes
            """
            self.layer_sizes = self.layer_sizes + [(self.dmp_param.n_bf * self.dmp_param.dof) + (2 * self.dmp_param.dof) + (1 if self.dmp_param.tau == None else 0)]
        
        self.dmp_param.timesteps = int(self.dmp_param.cs_runtime / self.dmp_param.dt)

class DMPParameters:
    def __init__(self):
        self.segments   = None # Set to None for NewCNNDMPNet; Set to (int) for SegmentedDMPNet
        self.dof        = None # No need to pre-define
        self.n_bf       = 200
        self.scale      = None # NEED to be defined. See dataset_importer
        self.dt         = .01 # * (1 if self.segments == None else self.segments)
        self.tau        = 1. # None if network include tau, assign a float value if not included

        # Canonical System Parameters
        self.cs_runtime = 1.0
        self.cs_ax      = 1.0

        # Dynamical System Parameters
        self.ay         = 75.
        self.by         = None # If not defined by = ay / 4

        self.timesteps = None # No need to pre-define

        # dict_trajectories = [
        #                      [[0.0, 0.0],
        #                       [1.0, 1.0]], # 0: Straight line
        #                      [[0.0, 0.0],
        #                       [1.0, 0.0],
        #                       [1.0, 1.0]], # 1: Diagonal curve bottom
        #                      [[0.0, 0.0],
        #                       [0.0, 1.0],
        #                       [1.0, 1.0]], # 2: Diagonal curve top
        #                      [[0.0, 0.0],
        #                       [0.0, 1.0],
        #                       [1.0, 1.0],
        #                       [1.0, 0.0]], # 3: Curve horizontal (Cannot move vertical)
        #                      [[0.0, 0.0],
        #                       [1.0, 0.0],
        #                       [1.0, 1.0],
        #                       [0.0, 1.0]], # 4: Curve vertical (Cannot move horizontal)
        #                     ]

        dict_trajectories = [
                             [[0.0, 0.0],
                              [1.0, 1.0]] # 0: Straight line
                            ]

        self.traj_dict = []
        for dict_trajectory in dict_trajectories:
            traj = np.array(dict_trajectory)
            dmp = DMPs_discrete(n_dmps = traj.shape[-1],
                                n_bfs=self.n_bf,
                                ay = np.ones(traj.shape[-1]) * self.ay,
                                dt = self.dt)
            dmp.imitate_path(traj.T)
            y, _, _ = dmp.rollout()
            # self.traj_dict.append((traj, y, dmp))
            self.traj_dict.append(y)
        
        self.segment_traj_length = int(1 / self.dt)
        # self.traj_dict = np.array(traj_dict, dtype = [('traj',object),('dmp_traj',object),('dmp',DMPs_discrete)])
        self.traj_dict = torch.from_numpy(np.array(self.traj_dict)).to(DEVICE)
        # print(self.traj_dict.shape)
        
#%% Test
# if __name__=='__main__':
#     train_param = TrainingParameters()
#     traj_dict = torch.from_numpy(train_param.model_param.dmp_param.traj_dict)
#     test_indices = torch.randint(0, 5, (20,))
    