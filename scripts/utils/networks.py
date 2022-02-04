from torch import nn, flatten, clone, ones, zeros, tensor, exp, linspace, sum, swapaxes, clamp
from torch.nn import ModuleList
import torch.nn.functional as F
import torch
from pydmps.dmp_discrete import DMPs_discrete
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNDMPNet(nn.Module):
    def __init__(self, train_param):
        """
        Deep DMP Network with old loss function (DMP parameters MSE)

        References:
        1. Training of deep neural networks for the generation of dynamic movementprimitives, Pahic et al, 2020, https://github.com/abr-ijs/imednet
        """
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param

        super().__init__()
        self.tanh = torch.nn.Tanh().to(DEVICE)
        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3).to(DEVICE)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)
        # self.cnn_model = MNISTNet()
        # self.cnn_model.load_state_dict(torch.load(model_param.pretrained_cnn_model_path))
        # self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.modules())[1:-3])

        # Define convolution layers
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5).to(DEVICE)
        # self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5).to(DEVICE)

        # Get convolution layers output shape and add it to layer_sizes
        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        layer_sizes = [conv_output_size] + self.model_param.layer_sizes
        
        # Define fully-connected layers
        self.fc = []
        for idx in range(len(layer_sizes[:-1])):
            self.fc.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]).to(DEVICE))

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_2(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_3(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_4(x1), 2), inplace=False)

        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = flatten(x, 1) # flatten all dimensions except batch
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        # x = self.fc[-1](x)
        x = self.fc[-1](self.dropout(x))
        return x

class NewCNNDMPNet(nn.Module):
    def __init__(self, model_param):
        """
        Deep DMP Network with new loss function (Trajectory MSE)

        References:
        1. Training of deep neural networks for the generation of dynamic movementprimitives, Pahic et al, 2020, https://github.com/abr-ijs/imednet
        """
        super().__init__()
        self.scale = model_param.scale
        self.tanh = torch.nn.Tanh().to(DEVICE)

        # Define convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5).to(DEVICE)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5).to(DEVICE)

        # Get convolution layers output shape and add it to layer_sizes
        _x = torch.ones(1, model_param.image_dim[0], model_param.image_dim[1], model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        layer_sizes = [conv_output_size] + model_param.layer_sizes
        
        # Define fully-connected layers
        self.fc = ModuleList()
        for idx in range(len(layer_sizes[:-1])):
            self.fc.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]).to(DEVICE))

    def forwardConv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = flatten(x, 1) # flatten all dimensions except batch
        return x.cuda()

    def forward(self, x):
        x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        output = self.fc[-1](x)
        traj = self.integrateDMP(output)
        return traj

    def integrateDMP(self, x):
        raise NotImplementedError

class SegmentedDMPNet(nn.Module):
    def __init__(self, train_param, output_size, surrogate_model):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.tanh = torch.nn.Tanh().to(DEVICE)

        # Define convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5).to(DEVICE)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5).to(DEVICE)

        # Get convolution layers output shape and add it to layer_sizes
        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        layer_sizes = [conv_output_size] + self.model_param.layer_sizes + output_size
        self.output_size = output_size
        # print(layer_sizes)
        # Define fully-connected layers
        self.fc = ModuleList()
        for idx in range(len(layer_sizes[:-1])):
            self.fc.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]).to(DEVICE))

        self.surrogate_model = surrogate_model

    def forwardConv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2), inplace=False)
        x = F.relu(F.max_pool2d(self.conv2(x), 2), inplace=False)
        x = flatten(x, 1) # flatten all dimensions except batch
        return x.cuda()

    def forward(self, x):
        x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        output = self.fc[-1](x)
        batch_size = output.shape[0]
        output = output.reshape(-1, self.surrogate_model.layer_sizes[0])
        # print(output.shape)
        # traj = self.integrateDMP(output)
        segment_traj = self.surrogate_model(output)
        traj = segment_traj.reshape(batch_size, -1)
        # traj = clamp(traj, min = 0, max = 1)
        return traj

    def integrateDMP(self, x, **kwargs):
        """
        Original DMP formulation based on pydmps modified to include segments tensor processing

        References:
        1. Dynamic Movement Primitives-A Framework for Motor Control in Humans and Humanoid Robotics, Schaal, 2002
        2. Dynamic Movement Primitives: Learning Attractor Models for Motor Behaviors, Ijspeert et al, 2013
        3. pydmps, DeWolf, 2013, https://github.com/studywolf/pydmps
        """
        dmp_param = self.model_param.dmp_param
        train_param = self.train_param
        batch_size_x = x.shape[0]

        def genDMPParametersFromOutput(x):
            splitOutput(rescaleDMPParameters(x))
            # splitOutput(x)
            genY0sGoalsFromSegmentsPoints()

        def rescaleDMPParameters(x):
            # print("Rescaling output")
            y_min = self.model_param.scale.y_min
            y_max = self.model_param.scale.y_max
            x_min = self.model_param.scale.x_min
            x_max = self.model_param.scale.x_max
            rescaled_x = (x - y_min) * (x_max - x_min) / (y_max - y_min) + x_min
            return rescaled_x

        def splitOutput(x):
            # print("Splitting output")
            if dmp_param.tau == None:
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
            # print(self.segment_points.shape)
            # print(self.weights.shape)

        def genY0sGoalsFromSegmentsPoints():
            # print("Splitting segments into y0 and goal")
            # print(self.segment_points[:,:-1].shape)
            self.y0s = self.segment_points[:,:-1].reshape(batch_size_x, dmp_param.segments, dmp_param.dof, 1)
            self.goals = self.segment_points[:,1:].reshape(batch_size_x, dmp_param.segments, dmp_param.dof, 1)
            # self.y0s = clamp(self.y0s, min = 0, max = 1)
            # self.goals = clamp(self.goals, min = 0, max = 1)

        def initializeDMP():
            self.x = ones(batch_size_x, dmp_param.segments, 1, 1).to(DEVICE)
            self.c = exp(-dmp_param.cs_ax * linspace(0, dmp_param.cs_runtime, dmp_param.n_bf).reshape(-1, 1)).to(DEVICE)
            self.c = self.c.repeat(dmp_param.segments, 1, 1)
            self.h = ones(dmp_param.n_bf, 1).to(DEVICE) * dmp_param.n_bf**1.5 / self.c / dmp_param.cs_ax
            self.y = torch.clone(self.y0s)
            self.dy = zeros(batch_size_x, dmp_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.ddy = zeros(batch_size_x, dmp_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.y_track_segment = zeros(batch_size_x, dmp_param.timesteps, dmp_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.dy_track_segment = zeros(batch_size_x, dmp_param.timesteps, dmp_param.segments, dmp_param.dof, 1).to(DEVICE)
            self.ddy_track_segment = zeros(batch_size_x, dmp_param.timesteps, dmp_param.segments, dmp_param.dof, 1).to(DEVICE)

        def integrate():
            for t in range(dmp_param.timesteps):
                self.y_track_segment[:, t] , self.dy_track_segment[:, t], self.ddy_track_segment[:, t] = step()

        def step():
            canonicalStep()
            psi = (exp(-self.h * (self.x - self.c)**2))
            f = zeros(batch_size_x, dmp_param.segments, dmp_param.dof, 1).to(DEVICE)
            for segment in range(dmp_param.segments):
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

        # print("Start integration", x.shape)
        genDMPParametersFromOutput(x)
        initializeDMP()
        integrate()
        recombineSegments()
        clearMemory()
        # print("Finish integration", self.y_track_segment.shape, self.y_track.shape)
        return self.y_track

class DMPIntegratorNet(nn.Module):
    def __init__(self, train_param, input_size, output_size, layer_sizes):
        super().__init__()
        self.tanh = torch.nn.Tanh().to(DEVICE)
        self.dropout = nn.Dropout(p=0.5)

        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param   = self.model_param.dmp_param

        # self.layer_sizes = [1024, 1024, 512, 256, 256, 512, 1024, 1024]
        self.layer_sizes = layer_sizes
        self.layer_sizes = [input_size] + self.layer_sizes + [output_size]
        # print(self.layer_sizes)

        # Define fully-connected layers
        self.fc = ModuleList()
        for idx in range(len(self.layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx+1]).to(DEVICE))
            # self.params.append(self.fc[-1].parameters())
            # self.parameters = Parameter(self.fc[:-1])

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        output = self.fc[-1](self.dropout(x))
        return output

class FixedSegmentDictDMPNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.dmp_param.segments
        self.traj_dict = self.dmp_param.traj_dict
        
        self.dmp_traj_length = self.traj_dict.shape[1]
        self.total_traj_length = self.traj_dict.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3).to(DEVICE)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)
        # self.conv1_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3).to(DEVICE)

        # self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=50).to(DEVICE)
        # self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=15).to(DEVICE)
        # self.conv2_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5).to(DEVICE)

        # self.conv3_1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=100).to(DEVICE)
        # self.conv3_2 = nn.Conv2d(in_channels=256, out_cha nnels=512, kernel_size=10).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.layer_sizes = self.model_param.layer_sizes
        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.layer_sizes = [conv_output_size] + self.layer_sizes

        # output size:
        # init pos x, init pos y, 
        # (traj_idx, mul_x, mul_y) * num_seg
        output_size = 2 + \
                      (3 * self.max_segments)
        self.layer_sizes = self.layer_sizes + [output_size]
        self.fc = ModuleList()
        for idx in range(len(self.layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx+1]).to(DEVICE))

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_2(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_3(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_4(x1), 2), inplace=False)
        # x1 = F.relu(F.max_pool2d(self.conv1_5(x1), 2), inplace=False)
        x1 = flatten(x1, 1) # flatten all dimensions except batch

        # x2 = F.relu(F.max_pool2d(self.conv2_1(x), 2), inplace=False)
        # x2 = F.relu(F.max_pool2d(self.conv2_2(x2), 2), inplace=False)
        # x2 = F.relu(F.max_pool2d(self.conv2_3(x2), 2), inplace=False)
        # x2 = flatten(x2, 1) # flatten all dimensions except batch

        # x3 = F.relu(F.max_pool2d(self.conv3_1(x), 2), inplace=False)
        # x3 = F.relu(F.max_pool2d(self.conv3_2(x3), 2), inplace=False)
        # x3 = flatten(x3, 1) # flatten all dimensions except batch

        # x = torch.cat([x1, x2, x3], dim = 1)
        # x = torch.cat([x1, x2], dim = 1)
        x = x1
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        x = self.fc[-1](self.dropout(x))
        # x = self.fc[-1](x)
        output = self.processDMPModifier(x)
        # print(output.shape)
        # return output, x
        return [output]

    def processDMPModifier(self, x):
        batch_s = x.shape[0]
        traj = torch.zeros(batch_s, self.total_traj_length, 2).to(DEVICE)
        last_pos = x[:, :2]
        for i in range(self.max_segments):
            cur_traj = self.getTrajFromDict(x[:, 2+(3*i)])
            x_mod = cur_traj[:,:,0] * x[:, 2+(3*i)+1].reshape(-1, 1)
            y_mod = cur_traj[:,:,1] * x[:, 2+(3*i)+2].reshape(-1, 1)
            x_mod = x_mod + last_pos[:, 0].reshape(-1, 1)
            y_mod = y_mod + last_pos[:, 1].reshape(-1, 1)
            last_pos = torch.cat([x_mod[:,-1].reshape(-1,1), y_mod[:,-1].reshape(-1,1)], dim = 1)
            traj[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length, 0] = x_mod
            traj[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length, 1] = y_mod
        return traj

    def getTrajFromDict(self, x):
        batch_s = x.shape[0]
        traj_dict = torch.zeros(batch_s, self.traj_dict.shape[1], self.traj_dict.shape[2]).to(DEVICE)
        for i in range(self.traj_dict.shape[0]):
            multiplier = torch.tensor(i).to(DEVICE) - torch.clamp(torch.round(x), min = 0, max = self.traj_dict.shape[0] - 1)
            multiplier = torch.abs(torch.abs(torch.sign(multiplier)) - torch.tensor(1).to(DEVICE))
            traj_dict = traj_dict + (torch.tile(self.traj_dict[i].reshape(1, self.traj_dict.shape[1], self.traj_dict.shape[2]), (batch_s, 1, 1)) * multiplier.reshape(batch_s, 1, 1))
        return traj_dict

class DynamicSegmentDictDMPNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.dmp_param.segments
        self.traj_dict = self.dmp_param.traj_dict

        self.dmp_traj_length = self.traj_dict.shape[1]
        self.total_traj_length = self.traj_dict.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)
        # self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3).to(DEVICE)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)

        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=10).to(DEVICE)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=10).to(DEVICE)
        self.conv2_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=10).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.layer_sizes = self.model_param.layer_sizes
        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.layer_sizes = [conv_output_size] + self.layer_sizes

        # output size:
        # num_seg, 
        # init pos x, init pos y, 
        # (traj_idx, mul_x, mul_y) * num_seg
        output_size = 1 + \
                      2 + \
                     (1 + 1 + 1) * self.max_segments 
        self.layer_sizes = self.layer_sizes + [output_size]
        self.fc = ModuleList()
        for idx in range(len(self.layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx+1]).to(DEVICE))
            
    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_2(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_3(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_4(x1), 2), inplace=False)
        # x1 = F.relu(F.max_pool2d(self.conv1_5(x1), 2), inplace=False)
        x1 = flatten(x1, 1) # flatten all dimensions except batch

        x2 = F.relu(F.max_pool2d(self.conv2_1(x), 2), inplace=False)
        x2 = F.relu(F.max_pool2d(self.conv2_2(x2), 2), inplace=False)
        x2 = F.relu(F.max_pool2d(self.conv2_3(x2), 2), inplace=False)
        x2 = flatten(x2, 1) # flatten all dimensions except batch

        # x3 = F.relu(F.max_pool2d(self.conv3_1(x), 2), inplace=False)
        # x3 = F.relu(F.max_pool2d(self.conv3_2(x3), 2), inplace=False)
        # x3 = flatten(x3, 1) # flatten all dimensions except batch

        # x = torch.cat([x1, x2, x3], dim = 1)
        x = torch.cat([x1, x2], dim = 1)
        # x = x1
        return x.cuda()
    
    def forward(self, x):
        # print(x.type())
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)

        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        x = self.fc[-1](self.dropout(x))
        # x = self.fc[-1](x)
        traj, points, segment_num, segment_types = self.processDMPModifier(x)
        # return traj, x, points, segment_num.reshape(-1, 1)
        return [points, segment_num, segment_types]
    
    def processDMPModifier(self, x):
        batch_s = x.shape[0]
        num_seg = x[:, 0] * self.max_segments
        segment_num = torch.round(torch.clamp(num_seg, min = 1, max = self.max_segments))
        points = x[:, 1:3].float().reshape(batch_s, 1, -1)
        traj = torch.zeros(batch_s, self.total_traj_length, 2).to(DEVICE)
        last_pos = x[:, 1:3]
        segment_types = None
        for i in range(self.max_segments):
            multiplier = torch.clamp(torch.sign(segment_num - i), min = 0, max = 1).reshape(-1, 1)
            # neg_multiplier = -(multiplier - 1)
            cur_traj = self.getTrajFromDict(x[:, 3+(3*i)])
            if segment_types == None:
                segment_types = x[:, 3+(3*i)].reshape(-1, 1)
            else:
                segment_types = torch.cat([segment_types, x[:, 3+(3*i)].reshape(-1, 1)], dim = 1)

            x_mod = cur_traj[:,:,0] * x[:, 3+(3*i)+1].reshape(-1, 1)
            y_mod = cur_traj[:,:,1] * x[:, 3+(3*i)+2].reshape(-1, 1)
            x_mod = x_mod * multiplier + last_pos[:, 0].reshape(-1, 1)
            y_mod = y_mod * multiplier + last_pos[:, 1].reshape(-1, 1)
            last_pos = torch.cat([x_mod[:,-1].reshape(-1,1), y_mod[:,-1].reshape(-1,1)], dim = 1)
            points = torch.cat((points, last_pos.reshape(batch_s, 1, -1).float()), dim = 1)
            traj[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length, 0] = x_mod
            traj[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length, 1] = y_mod
    
        return traj, points, num_seg.reshape(batch_s, 1), segment_types

    def getTrajFromDict(self, x):
        batch_s = x.shape[0]
        traj_dict = torch.zeros(batch_s, self.traj_dict.shape[1], self.traj_dict.shape[2]).to(DEVICE)
        for i in range(self.traj_dict.shape[0]):
            multiplier = torch.tensor(i).to(DEVICE) - torch.clamp(torch.round(x), min = 0, max = self.traj_dict.shape[0] - 1)
            multiplier = torch.abs(torch.abs(torch.sign(multiplier)) - torch.tensor(1).to(DEVICE))
            traj_dict = traj_dict + (torch.tile(self.traj_dict[i].reshape(1, self.traj_dict.shape[1], self.traj_dict.shape[2]), (batch_s, 1, 1)) * multiplier.reshape(batch_s, 1, 1))
        return traj_dict

class SegmentNumCNN(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.dmp_param.segments
        self.traj_dict = self.dmp_param.traj_dict

        self.dmp_traj_length = self.traj_dict.shape[1]
        self.total_traj_length = self.traj_dict.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)
        # self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3).to(DEVICE)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]

        self.layer_sizes = self.model_param.layer_sizes
        self.layer_sizes = [conv_output_size] + self.layer_sizes

        output_size = 1
        self.layer_sizes = self.layer_sizes + [output_size]
        self.fc = ModuleList()
        for idx in range(len(self.layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx+1]).to(DEVICE))

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_2(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_3(x1), 2), inplace=False)
        x1 = F.relu(F.max_pool2d(self.conv1_4(x1), 2), inplace=False)
        x1 = flatten(x1, 1) # flatten all dimensions except batch
        x = x1
        return x.cuda()

    def forward(self, x):
        x = self.forwardConv(x['image'])
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        x = self.fc[-1](self.dropout(x))
        # x = self.fc[-1](x)
        x = x * self.max_segments
        # traj, points, segment_num, segment_types = self.processDMPModifier(x)
        # return traj, x, points, segment_num.reshape(-1, 1)
        return [x]