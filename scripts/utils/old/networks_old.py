from torch import nn, flatten, clone, ones, zeros, tensor, exp, linspace, sum, swapaxes, clamp
from torch.nn import ModuleList
import torch.nn.functional as F
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class NewCNNDMPNet(nn.Module):
#     def __init__(self, train_param):
#         """
#         Deep DMP Network with new loss function (Trajectory MSE)

#         References:
#         1. Training of deep neural networks for the generation of dynamic movementprimitives, Pahic et al, 2020, https://github.com/abr-ijs/imednet
#         """
#         super().__init__()
#         self.train_param = train_param
#         self.model_param = self.train_param.model_param
#         self.dmp_param = self.model_param.dmp_param
#         self.scale = self.dmp_param.scale
#         self.tanh = torch.nn.Tanh().to(DEVICE)

#         # Define convolution layers
#         self.conv1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=10, kernel_size=5).to(DEVICE)
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5).to(DEVICE)

#         # Get convolution layers output shape and add it to layer_sizes
#         _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
#         conv_output_size = self.forwardConv(_x).shape[1]
#         layer_sizes = [conv_output_size] + self.model_param.hidden_layer_sizes
        
#         # Define fully-connected layers
#         self.fc = ModuleList()
#         for idx in range(len(layer_sizes[:-1])):
#             self.fc.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]).to(DEVICE))

#     def forwardConv(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = flatten(x, 1) # flatten all dimensions except batch
#         return x.to(DEVICE)

#     def forward(self, x):
#         x = self.forwardConv(x)
#         for fc in self.fc[:-1]:
#             x = self.tanh(fc(x))
#         output = self.fc[-1](x)
#         traj = self.integrateDMP(output)
#         return traj

#     def integrateDMP(self, x):
#         self.scale.denormalize_torch(x)
#         print(x.shape)

class SegmentedDMPNet(nn.Module):
    def __init__(self, train_param, output_size, surrogate_model):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.tanh = torch.nn.Tanh().to(DEVICE)

        # Define convolution layers
        self.conv1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=10, kernel_size=5).to(DEVICE)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5).to(DEVICE)

        # Get convolution layers output shape and add it to layer_sizes
        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        layer_sizes = [conv_output_size] + self.model_param.hidden_layer_sizes + output_size
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
        return x.to(DEVICE)

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
        model_param = self.model_param
        dmp_param = model_param.dmp_param
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
            self.y0s = self.segment_points[:,:-1].reshape(batch_size_x, model_param.segments, dmp_param.dof, 1)
            self.goals = self.segment_points[:,1:].reshape(batch_size_x, model_param.segments, dmp_param.dof, 1)
            # self.y0s = clamp(self.y0s, min = 0, max = 1)
            # self.goals = clamp(self.goals, min = 0, max = 1)

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
            psi = (exp(-self.h * (self.x - self.c)**2))
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

        # self.hidden_layer_sizes = [1024, 1024, 512, 256, 256, 512, 1024, 1024]
        self.hidden_layer_sizes = layer_sizes
        self.hidden_layer_sizes = [input_size] + self.hidden_layer_sizes + [output_size]
        # print(self.hidden_layer_sizes)

        # Define fully-connected layers
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))
            # self.params.append(self.fc[-1].parameters())
            # self.parameters = Parameter(self.fc[:-1])

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        output = self.fc[-1](self.dropout(x))
        return output
