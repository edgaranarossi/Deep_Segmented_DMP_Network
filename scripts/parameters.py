import torch
from torch import ones

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
        self.loss_type = None
        self.learning_rate = 1e-4
        self.eps = 1e-3
        self.weight_decay = None

        # Training parameters
        self.max_epoch = None
        self.max_val_fail = 50
        self.validation_interval = 1
        self.log_interval = 1
        self.plot_interval = 5

        # Data parameters
        self.batch_size = 50
        self.training_ratio = 7
        self.validation_ratio = 2
        self.test_ratio = 1
        self.includes_tau = 0

        # Processed parameters # No need to manually modify
        self.data_ratio = [self.training_ratio, self. validation_ratio, self.test_ratio]

        self.model_param = ModelParameters()

class DMPParameters:
    def __init__(self):
        self.segments   = 50 # Set to None for NewCNNDMPNet; Set to (int) for SegmentedDMPNet
        self.dof        = None # No need to pre-define
        self.n_bf       = 15
        self.scale      = None # NEED to be defined. See dataset_importer
        self.dt         = .025 # * (1 if self.segments == None else self.segments)
        self.tau        = 1. # None if network include tau, assign a float value if not included

        # Canonical System Parameters
        self.cs_runtime = 1.0
        self.cs_ax      = 1.0

        # Dynamical System Parameters
        self.ay         = 15.
        self.by         = None # If not defined by = ay / 4

        self.timesteps = None # No need to pre-define

class ModelParameters:
    def __init__(self):
        """
        output_mode:
        'dmp' : Use old loss function
        'traj' : Use new loss function which compares trajectory
        """
        # Network Parameters
        self.input_mode = 'dmp_param_scaled'
        self.output_mode = 'dmp_traj'
        self.image_dim = (1, 50, 50)
        self.layer_sizes = [1024, 1024, 1024] # Define hidden layers sizes (No need to define output layer size)

        self.dmp_param = DMPParameters()

        ## Processed parameters # No need to manually modify
        # Fill DMP None
        self.dmp_param.dof = len(self.image_dim) - 1
        self.dmp_param.ay = ones(self.dmp_param.segments, self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        if self.dmp_param.by == None:
            self.dmp_param.by = self.dmp_param.ay / 4
        else:
            ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.by

        if self.input_mode == 'img':
            """
            Calculate output layer size and add it to self.layer_sizes
            """
            if self.dmp_param.segments == None:
                self.layer_sizes = self.layer_sizes + [(self.dmp_param.n_bf * self.dmp_param.dof) + (2 * self.dmp_param.dof) + (1 if self.dmp_param.tau == None else 0)]
            elif self.dmp_param.segments > 0 and 'dmp_param' not in self.input_mode:
                self.num_segment_points = self.dmp_param.segments + 1
                self.num_segment_weights = self.dmp_param.segments
                self.len_segment_points = self.num_segment_points * self.dmp_param.dof
                self.len_segment_weights = self.num_segment_weights * self.dmp_param.dof * self.dmp_param.n_bf
                self.layer_sizes = self.layer_sizes +\
                                    [(1 if self.dmp_param.tau == None else 0) +\
                                    self.len_segment_points +\
                                    self.len_segment_weights]
            # self.dmp_param.dt = self.dmp_param.dt * self.dmp_param.segments
        # else:
        #     raise ValueError('self.dmp_param.segments must be either None or > 0')
        self.dmp_param.timesteps = int(self.dmp_param.cs_runtime / self.dmp_param.dt)
