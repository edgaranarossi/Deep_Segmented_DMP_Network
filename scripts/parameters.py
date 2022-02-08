import torch
from torch import ones
import numpy as np
from pydmps.dmp_discrete import DMPs_discrete
from os.path import join, isdir
from os import makedirs
from datetime import datetime
from utils.networks import CNNDMPNet, FixedSegmentDictDMPNet, DynamicSegmentDictDMPNet, SegmentNumCNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingParameters:
    def __init__(self):
        self.root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
        self.init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Dataset parameter
        # self.dataset_dir = join(self.root_dir, 'data/pkl/random_lines_3')
        # self.dataset_name = 'image-dict_output-traj_N_100000_dict_n-bf_5_ay_4_dt_0.05_normal_n-bf_6_ay_6_dt_0.01_2022-02-05_04-15-34.pkl'
        self.dataset_dir = join(self.root_dir, 'data/pkl/shapes-8')
        # self.dataset_name = 'image-dict_output-traj_N_100000_dict_n-bf_5_ay_4_dt_0.05_normal_n-bf_6_ay_6_dt_0.01_2022-02-05_04-13-23.pkl'
        self.dataset_name = 'image-dict_output-traj_N_100000_dict_n-bf_1_ay_4_dt_0.05_normal_n-bf_200_ay_75_dt_0.01_2022-02-07_21-56-44.pkl'
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
        self.learning_rate = 3e-6
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
        self.writeLog('Network created: ' + self.init_time)
        self.writeLog('Model : ' + str(self.model_param.model))
        self.writeLog(model.__str__())
        self.writeLog('Data Path : ' + self.dataset_path)
        self.writeLog('Model Save Path : ' + self.model_save_path)
        self.writeLog('Layer Sizes : ' + str(self.model_param.layer_sizes))

class ModelParameters:
    def __init__(self):
        # Network Parameters
        self.image_dim = (1, 150, 150)

        # Define hidden layers sizes (No need to define output layer size)
        # self.layer_sizes = [4096, 2048, 2048]
        # self.layer_sizes = [2048, 2048, 2048]
        # self.layer_sizes = [1024, 1024, 1024]
        self.layer_sizes = [128, 128, 128]
        # self.layer_sizes = [2048, 2048, 1024, 512, 256, 128, 64]
        # self.layer_sizes = [200, 50]
        # self.layer_sizes = [100, 8]
        self.dropout_prob = 0.3

        self.dmp_param = DMPParameters()

        """
        Network configurations:
        1.1: CNNDMPNet (outputs scaled)
        1.2: CNNDMPNet (outputs unscaled)
        2: NewCNNDMPNet
        3: FixedSegmentDictDMPNet
        4: DynamicSegmentDictDMPNet
        5: SegmentNumCNN
        """
        self.network_configuration = '2'

        if self.network_configuration == '1.1':
            self.model = CNNDMPNet

            self.input_mode     = ['image']
            self.output_mode    = ['dmp_y0_goal_w_scaled']
            self.loss_type      = ['MSE']
            self.dmp_param.segments = None

        elif self.network_configuration == '1.2':
            self.model = CNNDMPNet

            self.input_mode     = ['image']
            self.output_mode    = ['dmp_y0_goal_w']
            self.loss_type      = ['MSE']
            self.dmp_param.segments = None

        if self.network_configuration == '2':
            self.model = CNNDMPNet

            self.input_mode     = ['image']
            self.output_mode    = ['normal_dmp_traj']
            self.loss_type      = ['DMPIntegrationMSE']
            self.dmp_param.segments = None

        elif self.network_configuration == '3':
            self.model = FixedSegmentDictDMPNet

            self.input_mode     = ['image']
            self.output_mode    = ['traj_interpolated']
            self.loss_type      = ['SDTW']

            assert self.dmp_param.segments != None

        elif self.network_configuration == '4':
            self.model = DynamicSegmentDictDMPNet

            self.input_mode     = ['image']
            self.output_mode    = ['points_padded', 'num_segments', 'segment_types_padded']
            self.loss_type      = ['MSE', 'MSE', 'MSE']

            assert self.dmp_param.segments != None

        elif self.network_configuration == '5':
            self.model = SegmentNumCNN

            self.input_mode     = ['image']
            self.output_mode    = ['num_segments']
            self.loss_type      = ['MSE']
            
            assert self.dmp_param.segments != None

        else:
            raise ValueError('Wrong network configuration input')

        ## Processed parameters # No need to manually modify
        if self.dmp_param.segments != None:
            self.dmp_param.ay = ones(self.dmp_param.segments, self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        else:
            self.dmp_param.ay = ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        if self.dmp_param.by == None:
            self.dmp_param.by = self.dmp_param.ay / 4
        else:
            self.dmp_param.by = ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.by

        """
        Calculate output layer size and add it to self.layer_sizes
        """
        if self.model == CNNDMPNet:
            self.layer_sizes = self.layer_sizes + [(self.dmp_param.n_bf * self.dmp_param.dof) + (2 * self.dmp_param.dof) + (1 if self.dmp_param.tau == None else 0)]
        
        self.dmp_param.timesteps = int(self.dmp_param.cs_runtime / self.dmp_param.dt)

class DMPParameters:
    def __init__(self):
        self.segments   = 10
        self.dof        = 2
        self.n_bf       = 200
        self.scale      = None # NEED to be defined. See dataset_importer
        self.dt         = .01
        self.tau        = 1. # None if network include tau, assign a float value if not included

        # Canonical System Parameters
        self.cs_runtime = 1.0
        self.cs_ax      = 1.0

        # Dynamical System Parameters
        self.ay         = 75.
        self.by         = None # If not defined by = ay / 4

        self.timesteps = None # No need to pre-define

        dict_trajectories = [
                             [[0.0, 0.0],
                              [1.0, 1.0]], # 0: Straight line
                             [[0.0, 0.0],
                              [1.0, 0.0],
                              [1.0, 1.0]], # 1: Diagonal curve bottom
                             [[0.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 1.0]], # 2: Diagonal curve top
                            #  [[0.0, 0.0],
                            #   [0.0, 1.0],
                            #   [1.0, 1.0],
                            #   [1.0, 0.0]], # 3: Curve horizontal (Cannot move vertical)
                            #  [[0.0, 0.0],
                            #   [1.0, 0.0],
                            #   [1.0, 1.0],
                            #   [0.0, 1.0]], # 4: Curve vertical (Cannot move horizontal)
                            #  [[0.0, 0.0],
                            #   [-1.0, 1.0]], # 0: Straight line
                            #  [[0.0, 0.0],
                            #   [-1.0, 0.0],
                            #   [-1.0, 1.0]], # 1: Diagonal curve bottom
                            #  [[0.0, 0.0],
                            #   [0.0, 1.0],
                            #   [-1.0, 1.0]], # 2: Diagonal curve top
                            #  [[0.0, 0.0],
                            #   [0.0, 1.0],
                            #   [-1.0, 1.0],
                            #   [-1.0, 0.0]], # 3: Curve horizontal (Cannot move vertical)
                            #  [[0.0, 0.0],
                            #   [-1.0, 0.0],
                            #   [-1.0, 1.0],
                            #   [0.0, 1.0]], # 4: Curve vertical (Cannot move horizontal)
                            #  [[0.0, 0.0],
                            #   [1.0, -1.0]], # 0: Straight line
                            #  [[0.0, 0.0],
                            #   [1.0, 0.0],
                            #   [1.0, -1.0]], # 1: Diagonal curve bottom
                            #  [[0.0, 0.0],
                            #   [0.0, -1.0],
                            #   [1.0, -1.0]], # 2: Diagonal curve top
                            #  [[0.0, 0.0],
                            #   [0.0, -1.0],
                            #   [1.0, -1.0],
                            #   [1.0, 0.0]], # 3: Curve horizontal (Cannot move vertical)
                            #  [[0.0, 0.0],
                            #   [1.0, 0.0],
                            #   [1.0, -1.0],
                            #   [0.0, -1.0]], # 4: Curve vertical (Cannot move horizontal)
                            #  [[0.0, 0.0],
                            #   [-1.0, -1.0]], # 0: Straight line
                            #  [[0.0, 0.0],
                            #   [-1.0, 0.0],
                            #   [-1.0, -1.0]], # 1: Diagonal curve bottom
                            #  [[0.0, 0.0],
                            #   [0.0, -1.0],
                            #   [-1.0, -1.0]], # 2: Diagonal curve top
                            #  [[0.0, 0.0],
                            #   [0.0, -1.0],
                            #   [-1.0, -1.0],
                            #   [-1.0, 0.0]], # 3: Curve horizontal (Cannot move vertical)
                            #  [[0.0, 0.0],
                            #   [-1.0, 0.0],
                            #   [-1.0, -1.0],
                            #   [0.0, -1.0]], # 4: Curve vertical (Cannot move horizontal)
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
    