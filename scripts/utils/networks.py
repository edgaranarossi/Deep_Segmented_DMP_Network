from torch import nn, flatten, clone, ones, zeros, tensor, exp, linspace, sum, swapaxes, clamp, tile, abs, sign, round, zeros_like, cos, sin, cat, ceil, floor, remainder, tanh, sqrt, atan2
from torch.nn import ModuleList, LSTM
import torch.nn.functional as F
import torch
import numpy as np
from .pydmps_torch import DMPs_discrete_torch
import pickle as pkl

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

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=64, kernel_size=3).to(DEVICE)
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

        layer_sizes            = layer_sizes + \
                                 [(self.dmp_param.n_bf * self.dmp_param.dof) + \
                                  (2 * self.dmp_param.dof)]
        
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
        return [x]

class FixedSegmentDictDMPNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.model_param.max_segments
        self.dmp_dict = self.dmp_param.dmp_dict_param.dmp_dict
        
        self.dmp_traj_length = self.dmp_dict.shape[1]
        self.total_traj_length = self.dmp_dict.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=64, kernel_size=3).to(DEVICE)
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
        return [output, x]
        # return [output]

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
        dmp_dict = torch.zeros(batch_s, self.dmp_dict.shape[1], self.dmp_dict.shape[2]).to(DEVICE)
        for i in range(self.dmp_dict.shape[0]):
            multiplier = torch.tensor(i).to(DEVICE) - torch.clamp(torch.round(x * (self.dmp_dict.shape[0] - 1)), min = 0, max = self.dmp_dict.shape[0] - 1)
            multiplier = torch.abs(torch.abs(torch.sign(multiplier)) - torch.tensor(1).to(DEVICE))
            dmp_dict = dmp_dict + (torch.tile(self.dmp_dict[i].reshape(1, self.dmp_dict.shape[1], self.dmp_dict.shape[2]), (batch_s, 1, 1)) * multiplier.reshape(batch_s, 1, 1))
        return dmp_dict

class DynamicSegmentDictDMPNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.model_param.max_segments
        self.dmp_dict = self.dmp_param.dmp_dict_param.dmp_dict

        self.dmp_traj_length = self.dmp_dict.shape[1]
        self.total_traj_length = self.dmp_dict.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)
        # self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=64, kernel_size=3).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3).to(DEVICE)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)

        self.conv2_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=10).to(DEVICE)
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
        dmp_dict = torch.zeros(batch_s, self.dmp_dict.shape[1], self.dmp_dict.shape[2]).to(DEVICE)
        for i in range(self.dmp_dict.shape[0]):
            multiplier = torch.tensor(i).to(DEVICE) - torch.clamp(torch.round(x * (self.dmp_dict.shape[0] - 1)), min = 0, max = self.dmp_dict.shape[0] - 1)
            multiplier = torch.abs(torch.abs(torch.sign(multiplier)) - torch.tensor(1).to(DEVICE))
            dmp_dict = dmp_dict + (torch.tile(self.dmp_dict[i].reshape(1, self.dmp_dict.shape[1], self.dmp_dict.shape[2]), (batch_s, 1, 1)) * multiplier.reshape(batch_s, 1, 1))
        return dmp_dict

class SegmentNumCNN(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.model_param.max_segments
        self.dmp_dict = self.dmp_param.dmp_dict_param.dmp_dict

        self.dmp_traj_length = self.dmp_dict.shape[1]
        self.total_traj_length = self.dmp_dict.shape[1] * self.max_segments

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

class SegmentDictionaryDMPNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.max_segments = self.model_param.max_segments
        self.dmp_traj_length = int(1 / self.dmp_param.dt)
        self.total_traj_length = self.dmp_traj_length * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)
        self.selector_softmax = nn.Softmax(dim = 2).to(DEVICE)
        # self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=64, kernel_size=3).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3).to(DEVICE)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)

        self.conv2_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=10).to(DEVICE)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=10).to(DEVICE)
        self.conv2_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=10).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.hidden_layer_sizes = self.model_param.layer_sizes
        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes

        # output size = (dictionary_size * 
        #                (n_bf + y_goal) *
        #                dof) +
        #               y_init_each_dof + 
        #               (num_segments *
        #                (dictionary_index + traj_scaling_each_dof))
        output_dict_parameters_size = (self.model_param.dictionary_size * \
                                       (self.dmp_param.n_bf + 1) * \
                                      self.dmp_param.dof)
        output_init_pos_size        = self.dmp_param.dof
        output_dict_selector_size   = self.max_segments * self.model_param.dictionary_size
        output_dict_scaling_size    = self.max_segments * self.dmp_param.dof

        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        self.output_dict_parameters = nn.Linear(self.hidden_layer_sizes[-1], output_dict_parameters_size).to(DEVICE)
        self.output_init_pos        = nn.Linear(self.hidden_layer_sizes[-1], output_init_pos_size).to(DEVICE)
        self.output_dict_selector   = nn.Linear(self.hidden_layer_sizes[-1], output_dict_selector_size).to(DEVICE)
        self.output_dict_scaling    = nn.Linear(self.hidden_layer_sizes[-1], output_dict_scaling_size).to(DEVICE)

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
        self.cur_batch_size = x.shape[0]
        # print(x.type())
        if type(x) == dict:
            # for dataset
            x = self.forwardConv(x['image'])
        else:
            # for torchsummary
            x = self.forwardConv(x)

        for fc in self.fc:
            x = self.tanh(fc(x))

        dict_parameters = self.output_dict_parameters(x)
        # dict_parameters = self.output_dict_parameters(self.dropout(x))

        init_pos        = self.output_init_pos(x)
        # init_pos        = self.output_init_pos(self.dropout(x))

        # dict_selector   = self.output_dict_selector_size(x)
        dict_selector   = self.output_dict_selector(self.dropout(x))

        dict_scaling    = self.output_dict_scaling(x)
        # dict_scaling    = self.output_dict_scaling(self.dropout(x))

        y_track, dy_track, ddy_track = self.processOutput(dict_parameters, init_pos, dict_selector, dict_scaling)
        return [y_track, dy_track]
    
    def processOutput(self, dict_parameters, init_pos, dict_selector, dict_scaling):
        self.parseOutputs(dict_parameters, init_pos, dict_selector, dict_scaling)
        self.initializeDMP()
        self.reconstructTrajectory()
        return self.y_track, self.dy_track, self.ddy_track

    def parseOutputs(self, dict_parameters, init_pos, dict_selector, dict_scaling):
        self.dict_parameters    = dict_parameters.mean(dim = 0).reshape(self.model_param.dictionary_size,
                                                                        (self.dmp_param.n_bf + 1) * \
                                                                         self.dmp_param.dof)
        self.dict_goal          = dict_parameters[:, :self.model_param.dictionary_size * self.dmp_param.dof].reshape(self.model_param.dictionary_size, 
                                                                                                                     self.dmp_param.dof,
                                                                                                                     1)
        self.dict_w             = dict_parameters[:, self.model_param.dictionary_size * self.dmp_param.dof:].reshape(self.model_param.dictionary_size,
                                                                                                                     self.dmp_param.dof,
                                                                                                                     self.dmp_param.n_bf) * 1e1
        self.init_pos           = init_pos.reshape(self.cur_batch_size,
                                                   self.dmp_param.dof,
                                                   1)
        self.dict_selector      = dict_selector.reshape(self.cur_batch_size, 
                                                        self.max_segments, 
                                                        self.model_param.dictionary_size)
        self.dict_selector      = self.selector_softmax(self.dict_selector).argmax(dim=2)
        print(self.dict_selector)
        self.dict_scaling       = dict_scaling.reshape(self.cur_batch_size,
                                                       self.max_segments,
                                                       self.dmp_param.dof,
                                                       1)

    def initializeDMP(self):
        self.dmp                = DMPs_discrete_torch(n_dmps = self.dmp_param.dof, 
                                                      n_bfs  = self.dmp_param.n_bf, 
                                                      ay     = self.dmp_param.ay, 
                                                      dt     = self.dmp_param.dt)
    
    def reconstructTrajectory(self,):
        self.y_track = zeros(self.cur_batch_size, self.total_traj_length, self.dmp_param.dof).to(DEVICE)
        self.dy_track = zeros(self.cur_batch_size, self.total_traj_length, self.dmp_param.dof).to(DEVICE)
        self.ddy_track = zeros(self.cur_batch_size, self.total_traj_length, self.dmp_param.dof).to(DEVICE)
        last_y = clone(self.init_pos).to(DEVICE).reshape(self.cur_batch_size, self.dmp_param.dof)
        last_dy = clone(self.init_pos).to(DEVICE).reshape(self.cur_batch_size, self.dmp_param.dof)
        last_ddy = clone(self.init_pos).to(DEVICE).reshape(self.cur_batch_size, self.dmp_param.dof)
        for i in range(self.max_segments):
            self.setSegmentDMPParameters(self.dict_selector[:, i])
            # self.rolloutSegment(self.dict_scaling[:, i], last_dy = last_dy, last_ddy = last_ddy)
            self.rolloutSegment(self.dict_scaling[:, i])
            segment_y_track     = self.segment_y_track + last_y
            segment_dy_track    = self.segment_dy_track
            segment_ddy_track   = self.segment_ddy_track
            last_y = segment_y_track[:, -1, :]
            last_dy = segment_dy_track[:, -1, :]
            last_ddy = segment_ddy_track[:, -1, :]
            self.y_track[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length] = segment_y_track
            self.dy_track[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length] = segment_dy_track
            self.ddy_track[:, (self.dmp_traj_length*i):(self.dmp_traj_length*i)+self.dmp_traj_length] = segment_ddy_track

    def setSegmentDMPParameters(self, indices):
        self.dmp.w = zeros(self.cur_batch_size, self.dmp_param.dof, self.dmp_param.n_bf).to(DEVICE)
        self.dmp.goal = zeros(self.cur_batch_size, self.dmp_param.dof, 1).to(DEVICE)
        self.dmp.y0 = zeros_like(self.dmp.goal).to(DEVICE)
        for i in range(self.model_param.dictionary_size):
            multiplier = tensor(i).to(DEVICE) - indices
            multiplier = abs(abs(sign(multiplier)) - tensor(1).to(DEVICE))
            self.dmp.w = self.dmp.w + (tile(self.dict_w[i].reshape(1, self.dmp_param.dof, self.dmp_param.n_bf), (self.cur_batch_size, 1, 1)) * multiplier.reshape(self.cur_batch_size, 1, 1))
            self.dmp.goal = self.dmp.goal + (tile(self.dict_goal[i].reshape(1, self.dmp_param.dof, 1), (self.cur_batch_size, 1, 1)) * multiplier.reshape(self.cur_batch_size, 1, 1))

    def rolloutSegment(self, scaling, last_dy = None, last_ddy = None):
        # self.dmp.goal = self.dmp.goal * scaling.reshape(self.cur_batch_size, self.dmp_param.dof, 1)
        self.segment_y_track, self.segment_dy_track, self.segment_ddy_track = self.dmp.rollout(prev_dy = last_dy, prev_ddy = last_ddy)
    
class SampledSegmentDictDMPV1(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.base_traj = self.model_param.base_traj_param.base_traj
        self.max_segments = self.model_param.max_segments
        self.traj_length = self.model_param.traj_length
        self.hidden_layer_sizes = self.model_param.layer_sizes
        
        self.base_traj_length = self.base_traj.shape[1]
        self.total_traj_length = self.base_traj.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=64, kernel_size=3).to(DEVICE)
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
        self.sampling_ratio_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_init_pos_size            = self.dmp_param.dof
        output_segment_modifier_size    = (self.dmp_param.dof + 1) * self.max_segments
        output_sampling_ratio_size      = self.max_segments

        self.output_init_pos            = nn.Linear(self.hidden_layer_sizes[-1], output_init_pos_size).to(DEVICE)
        self.output_segment_modifier    = nn.Linear(self.hidden_layer_sizes[-1], output_segment_modifier_size).to(DEVICE)
        self.output_sampling_ratio      = nn.Linear(self.hidden_layer_sizes[-1], output_sampling_ratio_size).to(DEVICE)

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

        init_pos = self.output_init_pos(x)
        segment_modifier = self.output_segment_modifier(x).reshape(x.shape[0], self.max_segments, (self.dmp_param.dof + 1))
        sampling_ratio = self.sampling_ratio_softmax(self.output_sampling_ratio(x))

        y_track, segment_start_end, unsampled_traj = self.processOutput(init_pos, segment_modifier, sampling_ratio)
        return [y_track, [segment_start_end, sampling_ratio], unsampled_traj]
    
    def processOutput(self, init_pos, segment_modifier, sampling_ratio):
        batch_s = init_pos.shape[0]
        traj = zeros(batch_s, self.total_traj_length, self.dmp_param.dof).to(DEVICE)
        last_pos = init_pos.reshape(batch_s, 1, self.dmp_param.dof)
        segment_start_end = clone(last_pos)
        for i in range(self.max_segments):
            cur_traj = tile(self.base_traj, (batch_s, 1, 1))
            scaling = segment_modifier[:, i, :-1]
            rotation = tanh(segment_modifier[:, i, -1].reshape(batch_s, 1, 1)) * 3.14
            modded_traj = cur_traj * scaling.reshape(batch_s, 1, -1)
            modded_traj = self.rotateCoord2D(modded_traj, modded_traj[:, 0, :], rotation)
            modded_traj = last_pos + modded_traj
            last_pos = modded_traj[:, -1, :].reshape(batch_s, 1, self.dmp_param.dof)
            segment_start_end = cat([segment_start_end, last_pos], dim = 1)
            traj[:, (self.base_traj_length*i):(self.base_traj_length*i)+self.base_traj_length] = modded_traj
        sampled_traj = self.sample(traj, sampling_ratio)
        return sampled_traj, segment_start_end.float(), traj

    def sample(self, original_traj, sampling_ratio):
        batch_s         = original_traj.shape[0]
        total_limit     = self.traj_length
        segment_limit   = ceil(sampling_ratio * total_limit)
        over_limit      = segment_limit.sum(dim = 1) - self.base_traj_length

        for i in range(over_limit.max().int().item()):
            max_val = segment_limit.max(dim = 1).values #modify to sequential
            still_over = (segment_limit.sum(dim = 1) - self.base_traj_length).bool().int()
            reduced = ones(batch_s).to(DEVICE)
            for j in range(self.max_segments):
                multiplier = abs(sign(max_val - segment_limit[:, j]) - 1)
                segment_limit[:, j] = segment_limit[:, j] - (multiplier * reduced * still_over)
                reduced = clamp(reduced - multiplier, min = 0)

        velocity_mask   = zeros(batch_s, self.total_traj_length, self.dmp_param.dof).to(DEVICE)
        sampled_count   = zeros(batch_s, 1).to(DEVICE)
        segment_sampled_count = zeros(batch_s, self.max_segments, 1).to(DEVICE)
        di = floor(((self.base_traj_length - 1) - segment_sampled_count[:, 0, :]) / (segment_limit[:, 0].reshape(batch_s, 1) - segment_sampled_count[:, 0, :]))
        last_idx = zeros(batch_s, 1).to(DEVICE)

        for i in range(self.total_traj_length):
            cur_segment_idx = floor(tensor(i / self.base_traj_length).to(DEVICE)).int()
            # print(segment_sampled_count[:, cur_segment_idx, :].shape)
            new_di          = floor(((self.base_traj_length - 1) - (last_idx - (cur_segment_idx * self.base_traj_length))) / (segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]))
            max_idx         = di < new_di
            if i > 0 and cur_segment_idx != floor(tensor(i-1 / self.base_traj_length).to(DEVICE)).int():
                max_idx = ones(batch_s, 1).bool()
            di[max_idx]     = new_di[max_idx]
            # di              = floor(self.base_traj_length / (segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]))
            # print(cat([tile(tensor(self.base_traj_length).reshape(-1, 1).to(DEVICE), (batch_s, 1)), di, segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]], axis = 1))
            total_length_check = sign(clamp(total_limit - sampled_count, min = 0))
            segment_length_check = (segment_sampled_count[:, cur_segment_idx, :] < segment_limit[:, cur_segment_idx].reshape(-1, 1)).int()
            # index_check = abs(sign(remainder(i, di[:, cur_segment_idx].reshape(-1, 1))) - 1).reshape(batch_s, 1)
            index_check = ((ones(batch_s, 1).to(DEVICE) * i) - last_idx == di).int()
            total_check = total_length_check * segment_length_check * index_check

            new_idx = i * total_check
            max_last = last_idx < new_idx
            last_idx[max_last] = new_idx[max_last]
            # index_check = abs(sign(remainder(i, di)) - 1).reshape(batch_s, 1)

            # if not (total_length_check * segment_length_check * index_check).max().bool().item():
            #     print(i, '\n',
            #           last_idx, 'last_idx\n',
            #           di, 'di\n',
            #         #   new_di, 'new_di\n',
            #         #   segment_limit[:, cur_segment_idx].reshape(batch_s, 1), 'seg_lim\n',
            #           (self.base_traj_length - (last_idx - (cur_segment_idx * self.base_traj_length))), 'seg_left\n',
            #           (segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]), 'limit_left\n',
            #         #   ((ones(batch_s, 1).to(DEVICE) * i - (cur_segment_idx * self.base_traj_length)) / di[:, cur_segment_idx].reshape(-1, 1)), '\n',
            #         #   segment_limit[:, cur_segment_idx].reshape(-1, 1), '\n',
            #           (ones(batch_s, 1).to(DEVICE) * i) - last_idx, '(ones(batch_s, 1).to(DEVICE) * i) - last_idx\n',
            #         #   sampled_count, 'sampled_count\n',
            #         #   total_length_check, 'total_length_check\n', 
            #         #   segment_length_check, 'segment_length_check\n', 
            #           index_check, 'index_check\n')
                # input()
                      
            # input()
            velocity_mask[:, i, :] = velocity_mask[:, i, :] + total_check
            sampled_count = sampled_count + total_check
            segment_sampled_count[:, cur_segment_idx, :] = segment_sampled_count[:, cur_segment_idx, :] + total_check
            # print(last_idx)
        velocity_mask = velocity_mask.bool()
        sampled_traj = original_traj[velocity_mask].reshape(batch_s, total_limit, self.dmp_param.dof)
        return sampled_traj

    def rotateCoord2D(self, y, y0, rot_deg):
        batch_s = y.shape[0]
        px = y[:, :, 0].reshape(batch_s, -1, 1)
        py = y[:, :, 1].reshape(batch_s, -1, 1)
        
        cx = y0[:, 0].reshape(batch_s, 1, 1)
        cy = y0[:, 1].reshape(batch_s, 1, 1)
        # print(y.shape, y0.shape, rot_deg.shape)
        new_x = cos(rot_deg).to(DEVICE) * (px-cx) - sin(rot_deg).to(DEVICE) * (py-cy) + cx
        new_y = sin(rot_deg).to(DEVICE) * (px-cx) + cos(rot_deg).to(DEVICE) * (py-cy) + cy
        # print(new_x.shape, new_y.shape)
        
        y_rot = cat([new_x, new_y], dim = 2)#.reshape(batch_s, -1, self.dmp_param.dof)
        return y_rot
    
class SampledSegmentDictDMPV2(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.base_traj = self.model_param.base_traj_param.base_traj
        self.max_segments = self.model_param.max_segments
        self.traj_length = self.model_param.traj_length
        self.hidden_layer_sizes = self.model_param.layer_sizes
        
        self.base_traj_length = self.base_traj.shape[1]
        self.total_traj_length = self.base_traj.shape[1] * self.max_segments

        self.tanh = torch.nn.Tanh().to(DEVICE)

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=64, kernel_size=3).to(DEVICE)
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
        self.sampling_ratio_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_segment_points_size      = self.dmp_param.dof * (self.max_segments + 1)
        output_y_scaling_size           = self.max_segments
        output_sampling_ratio_size      = self.max_segments

        self.output_segment_points      = nn.Linear(self.hidden_layer_sizes[-1], output_segment_points_size).to(DEVICE)
        self.output_y_scaling           = nn.Linear(self.hidden_layer_sizes[-1], output_y_scaling_size).to(DEVICE)
        self.output_sampling_ratio      = nn.Linear(self.hidden_layer_sizes[-1], output_sampling_ratio_size).to(DEVICE)

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
        self.batch_s = x.shape[0]
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
        x = self.fc[-1](self.dropout(x))
        # x = self.fc[-1](x)

        segment_points = self.output_segment_points(x).reshape(self.batch_s, -1, self.dmp_param.dof)
        y_scaling = tanh(self.output_y_scaling(x).reshape(self.batch_s, -1, 1))
        sampling_ratio = self.sampling_ratio_softmax(self.output_sampling_ratio(x))

        sampled_traj, unsampled_traj = self.processOutput(segment_points, y_scaling, sampling_ratio)
        return [sampled_traj, [segment_points, sampling_ratio], unsampled_traj]
    
    def processOutput(self, segment_points, y_scaling, sampling_ratio):
        unsampled_traj, scalings, rotations = self.parseTrajectory(segment_points, y_scaling)
        sampled_traj                        = self.sampleTrajectory(unsampled_traj, sampling_ratio)
        return sampled_traj, unsampled_traj

    def parseTrajectory(self, segment_points, y_scaling):
        start_points    = segment_points[:, :-1, :].reshape(self.batch_s, self.max_segments, 1, self.dmp_param.dof)
        end_points      = segment_points[:, 1:, :].reshape(self.batch_s, self.max_segments, 1, self.dmp_param.dof)
        x_scaling       = sqrt(sum((end_points - start_points)**2, dim = 3))
        scaling         = cat([x_scaling, y_scaling], dim = 2).reshape(self.batch_s, self.max_segments, 1, self.dmp_param.dof)
        rotations       = atan2((end_points[:, :, :, 1] - start_points[:, :, :, 1]), (end_points[:, :, :, 0] - start_points[:, :, :, 0])).reshape(self.batch_s, self.max_segments, 1, 1)
        unsampled_traj  = tile(self.base_traj.reshape(1, 1, -1, self.dmp_param.dof), (self.batch_s, self.max_segments, 1, 1))
        unsampled_traj  = unsampled_traj * scaling
        unsampled_traj  = unsampled_traj + start_points.reshape(self.batch_s, self.max_segments, 1, self.dmp_param.dof)
        unsampled_traj  = self.rotateSegments2D(unsampled_traj, start_points, rotations).reshape(self.batch_s, -1, self.dmp_param.dof)
        return unsampled_traj, scaling, rotations

    def rotateSegments2D(self, p, p0, rot_deg):
        px          = p[:, :, :, 0].reshape(self.batch_s, self.max_segments, -1, 1)
        py          = p[:, :, :, 1].reshape(self.batch_s, self.max_segments, -1, 1)
        cx          = p0[:, :, :, 0].reshape(self.batch_s, self.max_segments, 1, 1)
        cy          = p0[:, :, :, 1].reshape(self.batch_s, self.max_segments, 1, 1)
        new_x       = cos(rot_deg) * (px-cx) - sin(rot_deg) * (py-cy) + cx
        new_y       = sin(rot_deg) * (px-cx) + cos(rot_deg) * (py-cy) + cy
        rotated_y   = cat([new_x, new_y], dim = 3)
        return rotated_y

    def sampleTrajectory(self, original_traj, sampling_ratio):
        batch_s         = original_traj.shape[0]
        total_limit     = self.traj_length
        segment_limit   = ceil(sampling_ratio * total_limit)
        over_limit      = segment_limit.sum(dim = 1) - self.base_traj_length

        # Decrease segments sequentially
        # all_segment_reducer = floor(over_limit / self.max_segments)
        # segment_limit = segment_limit - all_segment_reducer.reshape(self.batch_s, 1)
        # over_limit      = over_limit - (all_segment_reducer * self.max_segments)
        # i = 0
        # while over_limit.max().int().item() > 0:
        # # for i in range(over_limit.max().int().item()):
        #     segment_limit[:, i] = segment_limit[:, i] - (1 * sign(over_limit) * clamp(sign(segment_limit[:, i] - 2), min = 0))
        #     over_limit = over_limit - (1 * sign(over_limit) * clamp(sign(segment_limit[:, i] - 2), min = 0))
        #     i += 1

        # Decrease highest segment count
        for i in range(over_limit.max().int().item()):
            max_val = segment_limit.max(dim = 1).values
            still_over = (segment_limit.sum(dim = 1) - self.base_traj_length).bool().int()
            reduced = ones(batch_s).to(DEVICE)
            for j in range(self.max_segments):
                multiplier = abs(sign(max_val - segment_limit[:, j]) - 1)
                segment_limit[:, j] = segment_limit[:, j] - (multiplier * reduced * still_over)
                reduced = clamp(reduced - multiplier, min = 0)

        velocity_mask   = zeros(batch_s, self.total_traj_length, self.dmp_param.dof).to(DEVICE)
        sampled_count   = zeros(batch_s, 1).to(DEVICE)
        segment_sampled_count = zeros(batch_s, self.max_segments, 1).to(DEVICE)
        di = floor(((self.base_traj_length - 1) - segment_sampled_count[:, 0, :]) / (segment_limit[:, 0].reshape(batch_s, 1) - segment_sampled_count[:, 0, :]))
        last_idx = zeros(batch_s, 1).to(DEVICE)

        for i in range(self.total_traj_length):
            cur_segment_idx = floor(tensor(i / self.base_traj_length).to(DEVICE)).int()
            # print(segment_sampled_count[:, cur_segment_idx, :].shape)
            new_di          = floor(((self.base_traj_length - 1) - (last_idx - (cur_segment_idx * self.base_traj_length))) / (segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]))
            max_idx         = di < new_di
            if i > 0 and cur_segment_idx != floor(tensor(i-1 / self.base_traj_length).to(DEVICE)).int():
                max_idx = ones(batch_s, 1).bool()
            di[max_idx]     = new_di[max_idx]
            # di              = floor(self.base_traj_length / (segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]))
            # print(cat([tile(tensor(self.base_traj_length).reshape(-1, 1).to(DEVICE), (batch_s, 1)), di, segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]], axis = 1))
            total_length_check = sign(clamp(total_limit - sampled_count, min = 0))
            segment_length_check = (segment_sampled_count[:, cur_segment_idx, :] < segment_limit[:, cur_segment_idx].reshape(-1, 1)).int()
            # index_check = abs(sign(remainder(i, di[:, cur_segment_idx].reshape(-1, 1))) - 1).reshape(batch_s, 1)
            index_check = ((ones(batch_s, 1).to(DEVICE) * i) - last_idx == di).int()
            total_check = total_length_check * segment_length_check * index_check

            new_idx = i * total_check
            max_last = last_idx < new_idx
            last_idx[max_last] = new_idx[max_last]
            # index_check = abs(sign(remainder(i, di)) - 1).reshape(batch_s, 1)

            # if not (total_length_check * segment_length_check * index_check).max().bool().item():
            #     print(i, '\n',
            #           last_idx, 'last_idx\n',
            #           di, 'di\n',
            #         #   new_di, 'new_di\n',
            #         #   segment_limit[:, cur_segment_idx].reshape(batch_s, 1), 'seg_lim\n',
            #           (self.base_traj_length - (last_idx - (cur_segment_idx * self.base_traj_length))), 'seg_left\n',
            #           (segment_limit[:, cur_segment_idx].reshape(batch_s, 1) - segment_sampled_count[:, cur_segment_idx, :]), 'limit_left\n',
            #         #   ((ones(batch_s, 1).to(DEVICE) * i - (cur_segment_idx * self.base_traj_length)) / di[:, cur_segment_idx].reshape(-1, 1)), '\n',
            #         #   segment_limit[:, cur_segment_idx].reshape(-1, 1), '\n',
            #           (ones(batch_s, 1).to(DEVICE) * i) - last_idx, '(ones(batch_s, 1).to(DEVICE) * i) - last_idx\n',
            #         #   sampled_count, 'sampled_count\n',
            #         #   total_length_check, 'total_length_check\n', 
            #         #   segment_length_check, 'segment_length_check\n', 
            #           index_check, 'index_check\n')
                # input()
                      
            # input()
            velocity_mask[:, i, :] = velocity_mask[:, i, :] + total_check
            sampled_count = sampled_count + total_check
            segment_sampled_count[:, cur_segment_idx, :] = segment_sampled_count[:, cur_segment_idx, :] + total_check
            # print(last_idx)
        velocity_mask = velocity_mask.bool()
        try:
            sampled_traj = original_traj[velocity_mask].reshape(batch_s, total_limit, self.dmp_param.dof)
        except RuntimeError:
            print(segment_limit)
            raise RuntimeError()
        return sampled_traj

class DynamicParameterDMPNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param
        self.model_param = train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.hidden_layer_sizes = self.model_param.layer_sizes
        self.max_segments = self.model_param.max_segments
        self.connect_segments = self.model_param.connect_segments
        self.dof = self.dmp_param.dof

        self.dynamical_model = self.model_param.dynamical_model(self.train_param)

        self.tanh = torch.nn.Tanh().to(DEVICE)

        self.conv1_1 = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=16, kernel_size=5).to(DEVICE)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5).to(DEVICE)
        # self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3).to(DEVICE)
        # self.conv1_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3).to(DEVICE)
        # self.conv1_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3).to(DEVICE)

        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=125).to(DEVICE)
        # self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=15).to(DEVICE)
        # self.conv2_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.num_segments_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        # output_num_segments_size        = self.max_segments
        output_num_segments_size        = 1
        output_y0_size                  = self.dof
        output_dmp_param_size           = ((2 if not self.connect_segments else 1) * self.dof) + (self.dmp_param.n_bf * self.dof)

        self.output_num_segments        = nn.Linear(self.hidden_layer_sizes[-1], output_num_segments_size).to(DEVICE)
        self.output_y0                  = nn.Linear(self.hidden_layer_sizes[-1], output_y0_size).to(DEVICE)
        self.output_dmp_param           = nn.Linear(self.hidden_layer_sizes[-1], output_dmp_param_size).to(DEVICE)

        # output_num_segments_size        = self.max_segments
        # output_segment_points_size      = self.dof * (self.max_segments + 1)
        # output_dmp_param_size           = self.dmp_param.n_bf * self.dof

        # self.output_num_segments        = nn.Linear(self.hidden_layer_sizes[-1], output_num_segments_size).to(DEVICE)
        # self.output_segment_points      = nn.Linear(self.hidden_layer_sizes[-1], output_segment_points_size).to(DEVICE)
        # self.output_dmp_param           = nn.Linear(self.hidden_layer_sizes[-1], output_dmp_param_size).to(DEVICE)

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x), 2), inplace=False)
        # x1 = F.relu(F.max_pool2d(self.conv1_2(x1), 2), inplace=False)
        # x1 = F.relu(F.max_pool2d(self.conv1_3(x1), 2), inplace=False)
        # x1 = F.relu(F.max_pool2d(self.conv1_4(x1), 2), inplace=False)
        # x1 = F.relu(F.max_pool2d(self.conv1_5(x1), 2), inplace=False)
        x1 = flatten(x1, 1) # flatten all dimensions except batch

        x2 = F.relu(F.max_pool2d(self.conv2_1(x), 2), inplace=False)
        # x2 = F.relu(F.max_pool2d(self.conv2_2(x2), 2), inplace=False)
        # x2 = F.relu(F.max_pool2d(self.conv2_3(x2), 2), inplace=False)
        x2 = flatten(x2, 1) # flatten all dimensions except batch

        x = torch.cat([x1, x2], dim = 1)
        # x = x1
        return x.cuda()

    def forward(self, x):
        self.batch_s = x.shape[0]
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            # x = self.tanh(fc(x))
            x = self.tanh(fc(self.dropout(x)))
        x = self.fc[-1](self.dropout(x))
        # x = self.fc[-1](x)

        # num_segments = self.num_segments_softmax(self.output_num_segments(x)).argmax(dim = 1).reshape(-1, 1).float() + 1
        num_segments = self.output_num_segments(x)
        y0 = self.output_y0(x).reshape(self.batch_s, self.dof)
        # segment_points = self.output_segment_points(self.dropout(x)).reshape(self.batch_s, self.max_segments + 1, self.dof)
        dmp_param = self.output_dmp_param(x).reshape(self.batch_s, 1, -1)
        # y0_goal_w[:, self.dof:] = y0_goal_w[:, self.dof:]

        for i in range(self.max_segments - 1):
            new_dmp_param = self.forwardState(x, dmp_param[:, -1, :], i)
            # print(goal_w.shape, new_goal_w.shape)
            dmp_param = cat([dmp_param, new_dmp_param.reshape(self.batch_s, 1, -1)], dim = 1)
            # y0_goal_w[:, self.dof:] = y0_goal_w[:, self.dof:]
        
        # print(y0)
        # print(dmp_param[:, :, :2])
        # y0 = segment_points[:,  0, :]
        # dmp_param = cat([segment_points[:,  1:, :], dmp_param], dim = 2)
        # print(y0.shape)
        # print(dmp_param.shape)

        if self.connect_segments:
            return [num_segments, y0, dmp_param]
        else:
            return [num_segments, dmp_param]

    def forwardState(self, context, dmp_param, seq_idx):
        if str(type(self.dynamical_model)).split('.')[-1].split("'")[0] == 'AutoEncoderNet':
            new_dmp_param = self.dynamical_model(context, dmp_param)
        elif str(type(self.dynamical_model)).split('.')[-1].split("'")[0] == 'LSTMNet':
            new_dmp_param = self.dynamical_model(context, dmp_param, (True if seq_idx == 0 else False))
        return new_dmp_param

class AutoEncoderNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = train_param.model_param
        self.connect_segments   = self.model_param.connect_segments
        self.dmp_param          = self.model_param.dmp_param
        self.hidden_layer_sizes = self.model_param.dynamical_model_hidden_layers
        self.dof                = self.dmp_param.dof
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.dropout            = nn.Dropout(p = self.model_param.dynamical_model_dropout_prob)

        output_context_size     = self.model_param.layer_sizes[-1]
        output_dmp_param_size   = ((2 if not self.connect_segments else 1) * self.dof) + (self.dmp_param.n_bf * self.dof)
        # output_dmp_param_size   = self.dmp_param.n_bf * self.dof

        self.input_layer        = nn.Bilinear(output_context_size, output_dmp_param_size, self.hidden_layer_sizes[0]).to(DEVICE)
        self.fc                 = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        self.output_dmp_param   = nn.Linear(self.hidden_layer_sizes[-1], output_dmp_param_size).to(DEVICE)

    def forward(self, context, dmp_param):
        x = self.input_layer(context, dmp_param)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
            # x = self.tanh(fc(self.dropout(x)))
        x = self.fc[-1](self.dropout(x))
        new_dmp_param = self.output_dmp_param(x)
        return new_dmp_param

class LSTMNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = train_param.model_param
        self.lstm_num_layers    = self.model_param.num_layers
        self.lstm_hidden_size   = self.model_param.hidden_size
        self.lstm_seq_length    = self.model_param.seq_length
        self.connect_segments   = self.model_param.connect_segments
        self.dmp_param          = self.model_param.dmp_param
        self.hidden_layer_sizes = self.model_param.pre_lstm_hidden_sizes
        self.dof                = self.dmp_param.dof
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.dropout            = nn.Dropout(p = self.model_param.dynamical_model_dropout_prob)

        output_context_size     = self.model_param.layer_sizes[-1]
        output_dmp_param_size   = ((2 if not self.connect_segments else 1) * self.dof) + (self.dmp_param.n_bf * self.dof)
        # output_dmp_param_size   = self.dmp_param.n_bf * self.dof

        self.input_layer        = nn.Bilinear(output_context_size, output_dmp_param_size, self.hidden_layer_sizes[0]).to(DEVICE)
        self.fc                 = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))
        self.lstm_layer         = LSTM(self.hidden_layer_sizes[-1], self.lstm_hidden_size, self.lstm_num_layers, batch_first = True).to(DEVICE)
        self.output_dmp_param   = nn.Linear(self.lstm_hidden_size, output_dmp_param_size).to(DEVICE)

    def reset_cell_hidden_states(self, batch_s):
        self.h_state = zeros(self.lstm_num_layers, batch_s, self.lstm_hidden_size).to(DEVICE)
        self.c_state = zeros(self.lstm_num_layers, batch_s, self.lstm_hidden_size).to(DEVICE)

    def forward(self, context, dmp_param, first_seq):
        batch_s = context.shape[0]
        if first_seq: self.reset_cell_hidden_states(batch_s)
        x = self.input_layer(context, dmp_param)
        for fc in self.fc:
            x = self.tanh(fc(x))
        x, (self.h_state, self.c_state) = self.lstm_layer(x.reshape(batch_s, 1, -1), (self.h_state, self.c_state))
        new_dmp_param = self.output_dmp_param(x[:, -1, :])
        return new_dmp_param

class FirstStageCNN(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.num_position       = self.max_segments + 1
        self.dof                = self.dmp_param.dof
        self.hidden_layer_sizes = self.model_param.layer_sizes

        self.conv_incline       = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=5).to(DEVICE)
        self.conv_width         = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(5, 49)).to(DEVICE)
        self.conv_height        = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(49, 5)).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.num_segments_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_num_segments_size    = 1
        # output_y0_size              = self.dof
        # output_goal_size            = self.dof
        output_pos_size             = self.num_position * self.dof
        output_w_size               = self.dof * self.dmp_param.n_bf

        self.output_num_segments    = nn.Linear(self.hidden_layer_sizes[-1], output_num_segments_size).to(DEVICE)
        # self.output_y0              = nn.Linear(self.hidden_layer_sizes[-1], output_y0_size).to(DEVICE)
        # self.output_goal            = nn.Linear(self.hidden_layer_sizes[-1], output_goal_size).to(DEVICE)
        self.output_pos             = nn.Linear(self.hidden_layer_sizes[-1], output_pos_size).to(DEVICE)
        self.output_w               = nn.Linear(self.hidden_layer_sizes[-1], output_w_size).to(DEVICE)

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv_incline(x), 2), inplace=False)
        x1 = flatten(x1, 1)

        x2 = F.relu(F.max_pool2d(self.conv_width(x), 2), inplace=False)
        x2 = flatten(x2, 1)

        x3 = F.relu(F.max_pool2d(self.conv_height(x), 2), inplace=False)
        x3 = flatten(x3, 1)

        x = torch.cat([x1, x2, x3], dim = 1)
        # x = x1
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)

        batch_s = x.shape[0]

        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
            # x = self.tanh(fc(self.dropout(x)))
        # x = self.fc[-1](x)
        x = self.fc[-1](self.dropout(x))

        num_segments    = self.output_num_segments(x)
        # y0              = self.output_y0(x)
        # goal            = self.output_goal(x)
        pos             = self.output_pos(x).reshape(batch_s, self.num_position, self.dof)
        w               = self.output_w(x)

        y0 = pos[:, 0]
        goals = pos[:, 1:] 

        return [num_segments, y0, goals, w, x]

class LSTMRemainingSegments(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param                    = train_param
        self.model_param                    = self.train_param.model_param
        self.dmp_param                      = self.model_param.dmp_param
        self.dof                            = self.dmp_param.dof
        self.cnn_model                      = self.model_param.cnn_model
        self.cnn_model_train_param          = self.model_param.cnn_model_train_param
        self.cnn_model_model_param          = self.cnn_model_train_param.model_param
        self.cnn_model_hidden_layer_sizes   = self.cnn_model_model_param.layer_sizes
        self.max_segments                   = self.model_param.max_segments
        
        self.lstm_goal_state_size           = self.model_param.lstm_goal_state_size
        self.lstm_w_state_size              = self.model_param.lstm_w_state_size
        self.lstm_goal_hidden_size          = self.model_param.lstm_goal_hidden_size
        self.lstm_w_hidden_size             = self.model_param.lstm_w_hidden_size
        self.lstm_goal_num_layer            = self.model_param.lstm_goal_num_layer
        self.lstm_w_num_layer               = self.model_param.lstm_w_num_layer
        self.pre_lstm_goal_hidden_size      = self.model_param.pre_lstm_goal_hidden_size
        self.pre_lstm_w_hidden_size         = self.model_param.pre_lstm_w_hidden_size

        self.dropout                        = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh                           = torch.nn.Tanh().to(DEVICE)

        # output_num_segments_size    = 1
        # output_y0_size              = self.dof
        # output_goal_size            = self.dof
        # output_w_size               = self.dof * self.dmp_param.n_bf
        output_new_goal_size        = self.dof
        output_new_w_size           = self.dof * self.dmp_param.n_bf

        # self.output_goal            = nn.Linear(self.hidden_layer_sizes[-1], output_goal_size).to(DEVICE)
        # self.output_w               = nn.Linear(self.hidden_layer_sizes[-1], output_w_size).to(DEVICE)

        self.input_pre_lstm_goal    = nn.Bilinear(self.cnn_model_hidden_layer_sizes[-1], self.lstm_goal_state_size, self.pre_lstm_goal_hidden_size[0]).to(DEVICE)
        self.input_pre_lstm_w       = nn.Bilinear(self.cnn_model_hidden_layer_sizes[-1], self.lstm_w_state_size, self.pre_lstm_w_hidden_size[0]).to(DEVICE)

        self.fc_goal = ModuleList()
        for idx in range(len(self.pre_lstm_goal_hidden_size[:-1])):
            self.fc_goal.append(nn.Linear(self.pre_lstm_goal_hidden_size[idx], self.pre_lstm_goal_hidden_size[idx+1]).to(DEVICE))

        self.fc_w = ModuleList()
        for idx in range(len(self.pre_lstm_w_hidden_size[:-1])):
            self.fc_w.append(nn.Linear(self.pre_lstm_w_hidden_size[idx], self.pre_lstm_w_hidden_size[idx+1]).to(DEVICE))

        self.lstm_goal              = nn.LSTM(self.pre_lstm_goal_hidden_size[-1], self.lstm_goal_hidden_size, self.lstm_goal_num_layer, batch_first = True).to(DEVICE)
        self.lstm_w                 = nn.LSTM(self.pre_lstm_w_hidden_size[-1], self.lstm_w_hidden_size, self.lstm_w_num_layer, batch_first = True).to(DEVICE)

        self.output_new_goal        = nn.Linear(self.lstm_goal_hidden_size, output_new_goal_size).to(DEVICE)
        self.output_new_w           = nn.Linear(self.lstm_w_hidden_size, output_new_w_size).to(DEVICE)

    def reset_cell_hidden_states(self, batch_s):
        self.goal_h_state = zeros(self.lstm_goal_num_layer, batch_s, self.lstm_goal_hidden_size).to(DEVICE)
        self.goal_c_state = zeros(self.lstm_goal_num_layer, batch_s, self.lstm_goal_hidden_size).to(DEVICE)

        self.w_h_state = zeros(self.lstm_w_num_layer, batch_s, self.lstm_w_hidden_size).to(DEVICE)
        self.w_c_state = zeros(self.lstm_w_num_layer, batch_s, self.lstm_w_hidden_size).to(DEVICE)

    def forward(self, x):
        if type(x) == dict:
            num_segments, y0, goal, w, context = self.cnn_model(x['image'])
        else:
            num_segments, y0, goal, w, context = self.cnn_model(x)

        batch_s         = context.shape[0]
        goals           = zeros(batch_s, self.max_segments, self.dof).to(DEVICE)
        ws              = zeros(batch_s, self.max_segments, self.dof * self.dmp_param.n_bf).to(DEVICE)

        goals[:, 0]     = goal
        ws[:, 0]        = w

        self.reset_cell_hidden_states(batch_s)
        for i in range(1, self.max_segments):
            prev_goal    = goals[:, i - 1].clone().to(DEVICE)
            prev_w       = ws[:, i - 1].clone().to(DEVICE)

            x           = self.input_pre_lstm_goal(context, prev_goal)
            y           = self.input_pre_lstm_w(context, cat([prev_goal, prev_w], dim = 1))

            for fc in self.fc_goal:
                x = self.tanh(fc(x))

            for fc in self.fc_w:
                y = self.tanh(fc(y))

            # print(self.goal_h_state.shape, self.goal_c_state.shape)
            x, (self.goal_h_state, self.goal_c_state)   = self.lstm_goal(x.reshape(batch_s, 1, -1), 
                                                                         (self.goal_h_state, 
                                                                          self.goal_c_state))
            y, (self.w_h_state, self.w_c_state)         = self.lstm_w(y.reshape(batch_s, 1, -1),
                                                                      (self.w_h_state, 
                                                                       self.w_c_state))

            new_goal    = self.output_new_goal(x[:, -1])
            new_w       = self.output_new_w(y[:, -1])

            goals[:, i] = new_goal
            ws[:, i]    = new_w

        return [num_segments, y0, goals, ws]

class SecondStageDMPWeightsLSTM(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param                    = train_param
        self.model_param                    = self.train_param.model_param
        self.dmp_param                      = self.model_param.dmp_param
        self.dof                            = self.dmp_param.dof
        self.cnn_model                      = self.model_param.cnn_model
        self.cnn_model_train_param          = self.model_param.cnn_model_train_param
        self.cnn_model_model_param          = self.cnn_model_train_param.model_param
        self.cnn_model_hidden_layer_sizes   = self.cnn_model_model_param.layer_sizes
        self.max_segments                   = self.model_param.max_segments
        
        self.dropout                        = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh                           = torch.nn.Tanh().to(DEVICE)

        self.pre_lstm_w_hidden_size         = self.model_param.pre_lstm_w_hidden_size
        
        self.lstm_w_state_size              = self.model_param.lstm_w_state_size
        self.lstm_w_hidden_size             = self.model_param.lstm_w_hidden_size
        self.lstm_w_num_layer               = self.model_param.lstm_w_num_layer

        output_new_w_size                   = self.dof * self.dmp_param.n_bf

        self.fc_w = ModuleList()
        for idx in range(len(self.pre_lstm_w_hidden_size[:-1])):
            self.fc_w.append(nn.Linear(self.pre_lstm_w_hidden_size[idx], self.pre_lstm_w_hidden_size[idx+1]).to(DEVICE))

        self.lstm_w                 = nn.LSTM(self.pre_lstm_w_hidden_size[-1], self.lstm_w_hidden_size, self.lstm_w_num_layer, batch_first = True).to(DEVICE)

        self.output_new_w           = nn.Linear(self.lstm_w_hidden_size, output_new_w_size).to(DEVICE)

    def reset_cell_hidden_states(self, batch_s):
        self.w_h_state = zeros(self.lstm_w_num_layer, batch_s, self.lstm_w_hidden_size).to(DEVICE)
        self.w_c_state = zeros(self.lstm_w_num_layer, batch_s, self.lstm_w_hidden_size).to(DEVICE)

    def forward(self, x):
        if type(x) == dict:
            num_segments, y0, goals, w, context = self.cnn_model(x['image'])
        else:
            num_segments, y0, goals, w, context = self.cnn_model(x)

        batch_s         = context.shape[0]
        ws              = zeros(batch_s, self.max_segments, self.dof * self.dmp_param.n_bf).to(DEVICE)
        ws[:, 0]        = w

        self.reset_cell_hidden_states(batch_s)
        for i in range(1, self.max_segments):
            prev_goal   = goals[:, i - 1].clone().to(DEVICE)
            prev_w      = ws[:, i - 1].clone().to(DEVICE)

            x           = cat([prev_goal, prev_w], dim = 1)

            for fc in self.fc_w:
                x = self.tanh(fc(x))

            x, (self.w_h_state, self.w_c_state) = self.lstm_w(x.reshape(batch_s, 1, -1),
                                                              (self.w_h_state, 
                                                               self.w_c_state))

            new_w       = self.output_new_w(x[:, -1])

            ws[:, i]    = new_w

        return [num_segments, y0, goals, ws]

class DeepKoopmanNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.hidden_layer_sizes = self.model_param.dynamical_model_hidden_layers
        self.dof                = self.dmp_param.dof
        raise NotImplementedError()

class SegmentPosNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.dof                = self.dmp_param.dof
        self.num_position       = self.model_param.max_segments + 1
        self.hidden_layer_sizes = self.model_param.layer_sizes

        self.conv_incline       = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=10).to(DEVICE)
        self.conv_width         = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(5, 49)).to(DEVICE)
        self.conv_height        = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(49, 5)).to(DEVICE)

        self.dropout            = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.num_segments_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes

        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_pos_size         = self.dof * self.num_position
        self.output_pos         = nn.Linear(self.hidden_layer_sizes[-1], output_pos_size).to(DEVICE)

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv_incline(x), 2), inplace=False)
        x1 = flatten(x1, 1)

        x2 = F.relu(F.max_pool2d(self.conv_width(x), 2), inplace=False)
        x2 = flatten(x2, 1)

        x3 = F.relu(F.max_pool2d(self.conv_height(x), 2), inplace=False)
        x3 = flatten(x3, 1)

        x = torch.cat([x1, x2, x3], dim = 1)
        # x = x1
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        
        batch_s = x.shape[0]
        for fc in self.fc:
            # x = self.tanh(fc(x))
            x = self.tanh(fc(self.dropout(x)))

        pos = self.output_pos(x).reshape(batch_s, self.num_position, self.dof)

        y0 = pos[:, 0]
        goals = pos[:, 1:] 

        return [y0, goals]



class SegmentWeightNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.dof                = self.dmp_param.dof
        self.max_segments       = self.model_param.max_segments
        self.hidden_layer_sizes = self.model_param.layer_sizes

        self.conv_incline       = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=10).to(DEVICE)
        self.conv_width         = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(5, 49)).to(DEVICE)
        self.conv_height        = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(49, 5)).to(DEVICE)

        self.dropout            = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.num_segments_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes

        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_segment_w_size   = self.max_segments * self.dof * self.dmp_param.n_bf
        self.output_segment_w   = nn.Linear(self.hidden_layer_sizes[-1], output_segment_w_size).to(DEVICE)

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv_incline(x), 2), inplace=False)
        x1 = flatten(x1, 1)

        x2 = F.relu(F.max_pool2d(self.conv_width(x), 2), inplace=False)
        x2 = flatten(x2, 1)

        x3 = F.relu(F.max_pool2d(self.conv_height(x), 2), inplace=False)
        x3 = flatten(x3, 1)

        x = torch.cat([x1, x2, x3], dim = 1)
        # x = x1
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        
        batch_s = x.shape[0]
        for fc in self.fc:
            # x = self.tanh(fc(x))
            x = self.tanh(fc(self.dropout(x)))

        w = self.output_segment_w(x).reshape(batch_s, self.max_segments, self.dof * self.dmp_param.n_bf)

        return [w]

class SegmentDMPCNN(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.num_position       = self.max_segments + 1
        self.dof                = self.dmp_param.dof
        self.hidden_layer_sizes = self.model_param.layer_sizes

        self.conv_1             = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=5).to(DEVICE)
        self.conv_2             = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=10).to(DEVICE)
        self.conv_width         = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=(5, 49)).to(DEVICE)
        self.conv_height        = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=(49, 5)).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.num_segments_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_num_segments_size    = 1
        # output_y0_size              = self.dof
        # output_goal_size            = self.dof
        output_pos_size             = self.num_position * self.dof
        output_w_size               = self.max_segments * self.dof * self.dmp_param.n_bf

        self.output_num_segments    = nn.Linear(self.hidden_layer_sizes[-1], output_num_segments_size).to(DEVICE)
        # self.output_y0              = nn.Linear(self.hidden_layer_sizes[-1], output_y0_size).to(DEVICE)
        # self.output_goal            = nn.Linear(self.hidden_layer_sizes[-1], output_goal_size).to(DEVICE)
        self.output_pos             = nn.Linear(self.hidden_layer_sizes[-1], output_pos_size).to(DEVICE)
        self.output_w               = nn.Linear(self.hidden_layer_sizes[-1], output_w_size).to(DEVICE)

    def forwardConv(self, x):
        x0 = F.relu(F.max_pool2d(self.conv_1(x), 2), inplace=False)
        x0 = flatten(x0, 1)

        x1 = F.relu(F.max_pool2d(self.conv_2(x), 2), inplace=False)
        x1 = flatten(x1, 1)

        x2 = F.relu(F.max_pool2d(self.conv_width(x), 2), inplace=False)
        x2 = flatten(x2, 1)

        x3 = F.relu(F.max_pool2d(self.conv_height(x), 2), inplace=False)
        x3 = flatten(x3, 1)

        x = torch.cat([x0, x1, x2, x3], dim = 1)
        # x = x1
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)

        batch_s = x.shape[0]

        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
            # x = self.tanh(fc(self.dropout(x)))
        # x = self.fc[-1](x)
        x = self.fc[-1](self.dropout(x))

        num_segments    = self.output_num_segments(x)
        # y0              = self.output_y0(x)
        # goal            = self.output_goal(x)
        pos             = self.output_pos(x).reshape(batch_s, self.num_position, self.dof)
        w               = self.output_w(x).reshape(batch_s, self.max_segments, self.dof, self.dmp_param.n_bf)

        y0 = pos[:, 0]
        goals = pos[:, 1:] 

        return [num_segments, y0, goals, w]

class CNNDeepDMP(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = 1
        self.num_position       = self.max_segments + 1
        self.dof                = self.dmp_param.dof
        self.hidden_layer_sizes = self.model_param.layer_sizes

        self.conv_incline       = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=256, kernel_size=5).to(DEVICE)
        self.conv_width         = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(5, 49)).to(DEVICE)
        self.conv_height        = nn.Conv2d(in_channels=self.model_param.image_dim[0], out_channels=128, kernel_size=(49, 5)).to(DEVICE)

        self.dropout = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        self.num_segments_softmax = nn.Softmax(dim = 1).to(DEVICE)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [conv_output_size] + self.hidden_layer_sizes
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

        output_pos_size             = self.num_position * self.dof
        output_w_size               = self.max_segments * self.dof * self.dmp_param.n_bf

        self.output_pos             = nn.Linear(self.hidden_layer_sizes[-1], output_pos_size).to(DEVICE)
        self.output_w               = nn.Linear(self.hidden_layer_sizes[-1], output_w_size).to(DEVICE)

    def forwardConv(self, x):
        x1 = F.relu(F.max_pool2d(self.conv_incline(x), 2), inplace=False)
        x1 = flatten(x1, 1)

        x2 = F.relu(F.max_pool2d(self.conv_width(x), 2), inplace=False)
        x2 = flatten(x2, 1)

        x3 = F.relu(F.max_pool2d(self.conv_height(x), 2), inplace=False)
        x3 = flatten(x3, 1)

        x = torch.cat([x1, x2, x3], dim = 1)
        # x = x1
        return x.cuda()

    def forward(self, x):
        if type(x) == dict:
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)

        batch_s = x.shape[0]

        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
            # x = self.tanh(fc(self.dropout(x)))
        # x = self.fc[-1](x)
        x = self.fc[-1](self.dropout(x))

        pos             = self.output_pos(x).reshape(batch_s, self.num_position, self.dof)
        w               = self.output_w(x).reshape(batch_s, self.max_segments, self.dof * self.dmp_param.n_bf)

        y0 = pos[:, 0]
        goals = pos[:, 1].reshape(batch_s, 1, self.dof)

        return [y0, goals, w]