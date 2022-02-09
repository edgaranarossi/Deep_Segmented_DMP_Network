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
        self.max_segments = self.dmp_param.segments
        self.traj_dict = self.dmp_param.traj_dict
        
        self.dmp_traj_length = self.traj_dict.shape[1]
        self.total_traj_length = self.traj_dict.shape[1] * self.max_segments

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
        traj_dict = torch.zeros(batch_s, self.traj_dict.shape[1], self.traj_dict.shape[2]).to(DEVICE)
        for i in range(self.traj_dict.shape[0]):
            multiplier = torch.tensor(i).to(DEVICE) - torch.clamp(torch.round(x * (self.traj_dict.shape[0] - 1)), min = 0, max = self.traj_dict.shape[0] - 1)
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
        traj_dict = torch.zeros(batch_s, self.traj_dict.shape[1], self.traj_dict.shape[2]).to(DEVICE)
        for i in range(self.traj_dict.shape[0]):
            multiplier = torch.tensor(i).to(DEVICE) - torch.clamp(torch.round(x * (self.traj_dict.shape[0] - 1)), min = 0, max = self.traj_dict.shape[0] - 1)
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