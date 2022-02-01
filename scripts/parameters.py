import torch
from torch import ones
import numpy as np
from pydmps.dmp_discrete import DMPs_discrete

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingParameters:
    def __init__(self):
        # Optimizer parameters
        self.optimizer_type = 'adam'
        """
        loss:
        - MSE  : Mean Squared Error
        - SDTW : Soft Dynamic Time Warping
        - None : Model default
        """
        # self.loss_type = 'SDTW'
        self.loss_type = ['MSE', 'MSE', 'MSE']
        # self.loss_type = ['MSE']
        self.sdtw_gamma = 1e-3
        self.learning_rate = 3e-4
        self.eps = 5e-3
        self.weight_decay = None

        # Training parameters
        self.max_epoch = None
        self.max_val_fail = 500
        self.validation_interval = 1
        self.log_interval = 1
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

        self.model_param = ModelParameters()

class ModelParameters:
    def __init__(self):
        """
        output_mode:
        'dmp' : Use old loss function
        'traj' : Use new loss function which compares trajectory
        """
        # Network Parameters
        self.input_mode = 'image'
        self.output_mode = ['segmented_dict_dmp_outputs', 'num_segments', 'segmented_dict_dmp_types']
        # self.output_mode = ['num_segments']
        self.image_dim = (3, 150, 150)
        # Define hidden layers sizes (No need to define output layer size)
        # self.layer_sizes = [4096, 2048, 2048]
        # self.layer_sizes = [2048, 2048, 2048]
        self.layer_sizes = [1024, 1024, 1024]
        # self.layer_sizes = [128, 128, 128]
        self.dropout_prob = 0.5

        self.dmp_param = DMPParameters()

        ## Processed parameters # No need to manually modify
        # Fill DMP None
        self.dmp_param.dof = len(self.image_dim) - 1
        self.dmp_param.ay = ones(self.dmp_param.segments, self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        if self.dmp_param.by == None:
            self.dmp_param.by = self.dmp_param.ay / 4
        else:
            ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.by

        if self.input_mode == 'image':
            """
            Calculate output layer size and add it to self.layer_sizes
            """
            if self.dmp_param.segments == None:
                self.layer_sizes = self.layer_sizes + [(self.dmp_param.n_bf * self.dmp_param.dof) + (2 * self.dmp_param.dof) + (1 if self.dmp_param.tau == None else 0)]
            # elif self.dmp_param.segments > 0 and 'dmp_param' not in self.input_mode:
            #     self.max_segmentsment_points = self.dmp_param.segments + 1
            #     self.max_segmentsment_weights = self.dmp_param.segments
            #     self.len_segment_points = self.max_segmentsment_points * self.dmp_param.dof
            #     self.len_segment_weights = self.max_segmentsment_weights * self.dmp_param.dof * self.dmp_param.n_bf
            #     self.layer_sizes = self.layer_sizes +\
            #                         [(1 if self.dmp_param.tau == None else 0) +\
            #                         self.len_segment_points +\
            #                         self.len_segment_weights]
            # self.dmp_param.dt = self.dmp_param.dt * self.dmp_param.segments
        # else:
        #     raise ValueError('self.dmp_param.segments must be either None or > 0')
        self.dmp_param.timesteps = int(self.dmp_param.cs_runtime / self.dmp_param.dt)

class DMPParameters:
    def __init__(self):
        self.segments   = 6 # Set to None for NewCNNDMPNet; Set to (int) for SegmentedDMPNet
        self.dof        = None # No need to pre-define
        self.n_bf       = 1
        self.scale      = None # NEED to be defined. See dataset_importer
        self.dt         = .05 # * (1 if self.segments == None else self.segments)
        self.tau        = 1. # None if network include tau, assign a float value if not included

        # Canonical System Parameters
        self.cs_runtime = 1.0
        self.cs_ax      = 1.0

        # Dynamical System Parameters
        self.ay         = 4.
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
    