#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape Check Script

This script contains functions to check the shape of the network outputs for the Deep Segmented DMP Network.
"""

import torch
from torch import nn, flatten, ones, zeros, tensor, exp, linspace, sum, swapaxes, clamp
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingParameters:
    """
    Class to hold training parameters.
    """
    def __init__(self):
        # Optimizer parameters
        self.optimizer_type = 'adam'
        self.loss_type = None
        self.learning_rate = 1e-5
        self.eps = 1e-3
        self.weight_decay = None

        # Training parameters
        self.max_epoch = None
        self.max_val_fail = 60
        self.validation_interval = 1
        self.log_interval = 1

        # Data parameters
        self.batch_size = 140
        self.training_ratio = 7
        self.validation_ratio = 2
        self.test_ratio = 1
        self.includes_tau = 1

        # Processed parameters
        self.data_ratio = [self.training_ratio, self.validation_ratio, self.test_ratio]
        self.model_param = ModelParameters()

class DMPParameters:
    """
    Class to hold DMP parameters.
    """
    def __init__(self):
        self.segments = 50
        self.dof = None
        self.n_bf = 10
        self.scale = None
        self.dt = .05
        self.tau = 1.0
        self.cs_runtime = 1.0
        self.cs_ax = 1.0
        self.ay = 20.0
        self.by = None
        self.timesteps = None

class ModelParameters:
    """
    Class to hold model parameters.
    """
    def __init__(self):
        self.input_mode = 'image'
        self.output_mode = 'traj'
        self.image_dim = (1, 50, 50)
        self.hidden_layer_sizes = [20, 35]
        self.dmp_param = DMPParameters()
        self.dmp_param.dof = len(self.image_dim) - 1
        self.dmp_param.ay = ones(self.dmp_param.segments, self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.ay
        if self.dmp_param.by is None:
            self.dmp_param.by = self.dmp_param.ay / 4
        else:
            ones(self.dmp_param.dof, 1).to(DEVICE) * self.dmp_param.by

        if self.dmp_param.segments is None:
            self.hidden_layer_sizes += [(self.dmp_param.n_bf * self.dmp_param.dof) + (2 * self.dmp_param.dof) + (1 if self.dmp_param.tau is None else 0)]
        elif self.dmp_param.segments > 0:
            self.max_segmentsment_points = self.dmp_param.segments + 1
            self.max_segmentsment_weights = self.dmp_param.segments
            self.len_segment_points = self.max_segmentsment_points * self.dmp_param.dof
            self.len_segment_weights = self.max_segmentsment_weights * self.dmp_param.dof * self.dmp_param.n_bf
            self.hidden_layer_sizes += [(1 if self.dmp_param.tau is None else 0) + self.len_segment_points + self.len_segment_weights]
        else:
            raise ValueError('self.dmp_param.segments must be either None or > 0')
        self.dmp_param.timesteps = int(self.dmp_param.cs_runtime / self.dmp_param.dt)

class SegmentedDMPNet(nn.Module):
    """
    Segmented DMP Network class.
    """
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.tanh = torch.nn.Tanh().to(DEVICE)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5).to(DEVICE)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        layer_sizes = [conv_output_size] + self.model_param.hidden_layer_sizes
        
        self.fc = []
        for idx in range(len(layer_sizes[:-1])):
            self.fc.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]).to(DEVICE))

    def forwardConv(self, x):
        """
        Forward pass through convolutional layers.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2), inplace=False)
        x = F.relu(F.max_pool2d(self.conv2(x), 2), inplace=False)
        x = flatten(x, 1)
        return x.to(DEVICE)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        output = self.fc[-1](x)
        traj = self.integrateDMP(output)
        traj = clamp(traj, min=0, max=1)
        return traj

    def integrateDMP(self, x, **kwargs):
        """
        Integrate DMP to generate trajectory.

        Parameters:
        x (tensor): The network output tensor.

        Returns:
        tensor: The integrated trajectory.
        """
        dmp_param = self.model_param.dmp_param
        train_param = self.train_param
        batch_size_x = x.shape[0]

        def genDMPParametersFromOutput(x):
            splitOutput(x)
            genY0sGoalsFromSegmentsPoints()

        def rescaleDMPParameters(x):
            y_min = self.model_param.scale.y_min
            y_max = self.model_param.scale.y_max
            x_min = self.model_param.scale.x_min
            x_max = self.model_param.scale.x_max
            rescaled_x = (x - y_min) * (x_max - x_min) / (y_max - y_min) + x_min
            return rescaled_x

        def splitOutput(x):
            if dmp_param.tau is None:
                self.tau = x[:, 0].reshape(-1, 1, 1, 1)
                start_idx = 1
            else:
                self.tau = torch.ones(batch_size_x, 1, 1, 1).to(DEVICE) * dmp_param.tau
                start_idx = 0
            self.segment_points = x[:, start_idx:start_idx+self.model_param.len_segment_points]
            self.segment_points = self.segment_points.reshape(-1,
                                                    self.model_param.num_segment_points, 
                                                    dmp_param.dof)
            self.weights = x[:, start_idx+self.model_param.len_segment_points:]
            self.weights = self.weights.reshape(-1,
                                      self.model_param.num_segment_weights, 
                                      dmp_param.dof, 
                                      dmp_param.n_bf)

        def genY0sGoalsFromSegmentsPoints():
            self.y0s = self.segment_points[:,:-1].reshape(batch_size_x, model_param.segments, dmp_param.dof, 1)
            self.goals = self.segment_points[:,1:].reshape(batch_size_x, model_param.segments, dmp_param.dof, 1)
            self.y0s = clamp(self.y0s, min = 0, max = 1)
            self.goals = clamp(self.goals, min = 0, max = 1)

        def initializeDMP():
            self.x = ones(batch_size_x, model_param.segments, 1, 1).to(DEVICE)
            self.c = exp(-dmp_param.cs_ax * linspace(0, dmp_param.cs_runtime, dmp_param.n_bf).reshape(-1, 1)).to(DEVICE)
            self.c = self.c.repeat(model_param.segments, 1, 1)
            self.h = ones(dmp_param.n_bf, 1).to(DEVICE) * dmp_param.n_bf**1.5 / self.c / dmp_param.cs_ax
            self.y = torch.clone(self.y0s)
            self.dy = zeros(batch_size_x, model_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.ddy = zeros(batch_size_x, model_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.y_track_segment = zeros(batch_size_x, dmp_param.timesteps, model_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.dy_track_segment = zeros(batch_size_x, dmp_param.timesteps, model_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.ddy_track_segment = zeros(batch_size_x, dmp_param.timesteps, model_param.segments, dmp_param.dof, 1).to(DEVICE)

        def integrate():
            for t in range(dmp_param.timesteps):
                self.y_track_segment[:, t] , self.dy_track_segment[:, t], self.ddy_track_segment[:, t] = step()

        def step():
            canonicalStep()
            psi = (exp(-self.h * (self.x - self.c)**2)).double()
            f = zeros(batch_size_x, model_param.segments, dmp_param.dof, 1).to(DEVICE)
            for segment in range(model_param.segments):
                f[:, segment] = frontTerm()[:, segment] * (self.weights[:, segment] @ psi[:, segment]) / sum(psi[:, segment], axis=1).reshape(-1, 1, 1)
                
            self.ddy = (dmp_param.ay * (dmp_param.by * (self.goals - self.y) - self.dy / self.tau) + f) * self.tau
            self.dy = self.dy + (self.ddy * self.tau * dmp_param.dt)
            self.y = self.y + (self.dy * dmp_param.dt)
            return self.y, self.dy, self.ddy  

        def canonicalStep():
            self.x = self.x + (-dmp_param.cs_ax * self.x * self.tau * dmp_param.dt)

        def frontTerm():
            self.term = self.x * (self.goals - self.y0s)
            return self.term

        def recombineSegments():
            self.y_track = swapaxes(self.y_track_segment, 1, 2).reshape(batch_size_x, -1, dmp_param.dof, 1)
            self.dy_track = swapaxes(self.dy_track_segment, 1, 2).reshape(batch_size_x, -1, dmp_param.dof, 1)
            self.ddy_track = swapaxes(self.ddy_track_segment, 1, 2).reshape(batch_size_x, -1, dmp_param.dof, 1)

        def clearMemory():
            del self.x, self.c, self.h, self.y, self.dy, 
            self.ddy, self.y_track_segment, self.dy_track_segment, 
            self.ddy_track_segment, self.dy_track, self.ddy_track, 
            self.y0s, self.goals, self.segment_points, self.weights, self.tau

        genDMPParametersFromOutput(x)
        initializeDMP()
        integrate()
        recombineSegments()
        clearMemory()
        return self.y_track

if __name__ == '__main__':
    """
    Main function to run the shape check script.
    """
    train_param = TrainingParameters()
    model_param = train_param.model_param
    net = SegmentedDMPNet(train_param)

    tau = 1

    x = tensor(network_output.reshape(1, -1)).to(DEVICE)
    y_track = net.integrateDMP(x)

    from matplotlib import pyplot as plt
    y_track_np = y_track.squeeze().cpu().numpy()
    plt.plot(y_track_np[:,0], y_track_np[:,1])