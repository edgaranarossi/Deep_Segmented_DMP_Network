from cmath import tau
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torch import nn, flatten, clone, ones, zeros, tensor, exp, linspace, sum, swapaxes, clamp, tile, abs, sign, round, zeros_like, cos, sin, cat, ceil, floor, remainder, tanh, sqrt, atan2, cat
from torch.nn import ModuleList, LSTM
import torch.nn.functional as F
import torch
import numpy as np
from .pydmps_torch import DMPs_discrete_torch
import pickle as pkl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageInputProcessor(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.conv_layer_params  = self.model_param.conv_layer_params
        self.hidden_layer_sizes = self.model_param.hidden_layer_sizes
        self.output_size        = self.hidden_layer_sizes[-1]
        
        if self.model_param.backbone_option is not None:
            if self.model_param.backbone_option == 'keypointrcnn_resnet50_fpn':
                self.backbone_model     = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(DEVICE)
                if self.model_param.backbone_eval:
                    self.backbone_model.eval()
        else:
            self.conv_pipelines = ModuleList()
            for conv_pipeline in self.conv_layer_params:
                convs = ModuleList()
                for i, conv_param in enumerate(conv_pipeline):
                    if i == 0: in_size = self.model_param.image_dim[0]
                    else: in_size = conv_pipeline[i-1].out_channels
                    convs.append(nn.Conv2d(in_channels=in_size, out_channels=conv_param.out_channels, kernel_size=conv_param.kernel_size).to(DEVICE))
                self.conv_pipelines.append(convs)

        _x = torch.ones(1, self.model_param.image_dim[0], self.model_param.image_dim[1], self.model_param.image_dim[2]).to(DEVICE)
        self.conv_output_size = self.forwardConv(_x).shape[1]
        self.hidden_layer_sizes = [self.conv_output_size] + self.hidden_layer_sizes

        self.dropout            = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        
        self.fc = ModuleList()
        for idx in range(len(self.hidden_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx+1]).to(DEVICE))

    def forwardConv(self, x):
        batch_s = x.shape[0]
        # print(x.shape)

        if self.model_param.backbone_option is not None:
            x = self.backbone_model.backbone(x)
            features = None
            # print(len(x))
            for key in x:
                if key not in []:
                    # print(x[key].reshape(batch_s, -1).shape)
                    if features is None:
                        features = x[key].reshape(batch_s, -1)
                    else:
                        features = cat((features, x[key].reshape(batch_s, -1)), axis = 1)
            # print(features.shape)
            return features
        else:
            conv_pipelines = []
            for conv_pipeline in self.conv_pipelines:
                input_x = x
                for conv in conv_pipeline:
                    input_x = F.relu(F.max_pool2d(conv(input_x), self.model_param.max_pool_size), inplace = False)
                conv_pipelines.append(flatten(input_x, start_dim = 1))
            # for conv_pipeline in conv_pipelines:
            #     print(conv_pipeline.shape)
            x = cat(conv_pipelines, dim = 1).to(DEVICE)
            return x

    def forward(self, x):
        if type(x) == dict:
            # print(x['image'].shape)
            x = self.forwardConv(x['image'])
        else:
            x = self.forwardConv(x)
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
            # x = self.tanh(fc(self.dropout(x)))
        # x = self.fc[-1](x)
        x = self.tanh(self.fc[-1](self.dropout(x)))
        return x

class DMPWeightDecoder(nn.Module):
    def __init__(self, train_param, input_size):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.decoder_layer_sizes = self.model_param.decoder_layer_sizes
        self.decoder_layer_sizes = [input_size] + self.decoder_layer_sizes
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.dof                = self.dmp_param.dof
        self.n_bf               = self.dmp_param.n_bf

        self.dropout            = nn.Dropout(p = self.model_param.dropout_prob)
        self.tanh               = torch.nn.Tanh().to(DEVICE)
        
        self.fc = ModuleList()
        for idx in range(len(self.decoder_layer_sizes[:-1])):
            self.fc.append(nn.Linear(self.decoder_layer_sizes[idx], self.decoder_layer_sizes[idx+1]).to(DEVICE))

        self.output_w           = nn.Linear(self.decoder_layer_sizes[-1], self.dof * self.n_bf).to(DEVICE)

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = self.tanh(fc(x))
            # x = self.tanh(fc(self.dropout(x)))
        x = self.fc[-1](x)
        # x = self.tanh(self.fc[-1](self.dropout(x)))

        output_w = self.output_w(x)
        return output_w

class DSDNetV0(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.dof                = self.dmp_param.dof
        self.n_bf               = self.dmp_param.n_bf

        if self.model_param.input_mode == ['image']:
            self.input_processor    = ImageInputProcessor(self.train_param)
        self.input_processor_output_size = self.input_processor.output_size

        output_y0_size          = self.max_segments * self.dof
        output_goal_size        = self.max_segments * self.dof
        output_w_size           = self.max_segments * self.dof * self.n_bf
        output_tau_size         = self.max_segments

        self.output_y0          = nn.Linear(self.input_processor_output_size, output_y0_size).to(DEVICE)
        self.output_goal        = nn.Linear(self.input_processor_output_size, output_goal_size).to(DEVICE)
        self.output_w           = nn.Linear(self.input_processor_output_size, output_w_size).to(DEVICE)
        self.output_tau         = nn.Linear(self.input_processor_output_size, output_tau_size).to(DEVICE)

    def forward(self, x):
        x           = self.input_processor(x)
        batch_s     = x.shape[0]

        dmp_y0      = self.output_y0(x).reshape(batch_s, self.max_segments, self.dof)
        dmp_goal    = self.output_goal(x).reshape(batch_s, self.max_segments, self.dof)
        dmp_weights = self.output_w(x).reshape(batch_s, self.max_segments, self.dof, self.n_bf)
        dmp_tau     = self.output_tau(x).reshape(batch_s, self.max_segments)

        return [dmp_y0, dmp_goal, dmp_weights, dmp_tau]

class DSDNetV1(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.dof                = self.dmp_param.dof
        self.n_bf               = self.dmp_param.n_bf
        self.latent_w_size      = self.model_param.latent_w_size

        if self.model_param.input_mode == ['image']:
            self.input_processor    = ImageInputProcessor(self.train_param)
        self.input_processor_output_size = self.input_processor.output_size
        self.dmp_weight_decoder = DMPWeightDecoder(self.train_param, input_size = self.latent_w_size)

        output_num_segments_size= 1
        output_y0_size          = self.max_segments * self.dof
        output_goal_size        = self.max_segments * self.dof
        # output_pos_size        = (self.max_segments + 1) * self.dof
        output_tau_size         = self.max_segments

        self.output_num_segments= nn.Linear(self.input_processor_output_size, output_num_segments_size).to(DEVICE)
        self.output_y0          = nn.Linear(self.input_processor_output_size, output_y0_size).to(DEVICE)
        self.output_goal        = nn.Linear(self.input_processor_output_size, output_goal_size).to(DEVICE)
        # self.output_pos         = nn.Linear(self.input_processor_output_size, output_pos_size).to(DEVICE)
        self.latent_w           = nn.Linear(self.input_processor_output_size, self.max_segments * self.latent_w_size).to(DEVICE)
        self.output_tau         = nn.Linear(self.input_processor_output_size, output_tau_size).to(DEVICE)

    def forward(self, x):
        x           = self.input_processor(x)
        batch_s     = x.shape[0]

        latent_w    = self.latent_w(x).reshape(batch_s * self.max_segments, self.latent_w_size)

        dmp_num_segments = self.output_num_segments(x).reshape(batch_s, 1)
        dmp_y0      = self.output_y0(x).reshape(batch_s, self.max_segments, self.dof)
        dmp_goal    = self.output_goal(x).reshape(batch_s, self.max_segments, self.dof)

        # dmp_pos     = self.output_pos(x).reshape(batch_s, self.max_segments + 1, self.dof)
        # dmp_y0      = dmp_pos[:, :-1]
        # dmp_goal    = dmp_pos[:, 1:]

        dmp_weights = self.dmp_weight_decoder(latent_w).reshape(batch_s, self.max_segments, self.dof, self.n_bf)
        dmp_tau     = self.output_tau(x).reshape(batch_s, self.max_segments)
        # dmp_tau     = torch.abs(self.output_tau(x).reshape(batch_s, self.max_segments))

        return [dmp_num_segments, dmp_y0, dmp_goal, dmp_weights, dmp_tau]

class DSDNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.dof                = self.dmp_param.dof
        self.n_bf               = self.dmp_param.n_bf
        self.latent_w_size      = self.model_param.latent_w_size

        if self.model_param.input_mode == ['image']:
            self.input_processor    = ImageInputProcessor(self.train_param)
        self.input_processor_output_size = self.input_processor.output_size
        self.dmp_weight_decoder = DMPWeightDecoder(self.train_param, input_size = self.latent_w_size)

        output_num_segments_size= 1
        output_y0_size          = self.max_segments * self.dof
        output_goal_size        = self.max_segments * self.dof
        # output_pos_size        = (self.max_segments + 1) * self.dof
        output_tau_size         = self.max_segments

        self.output_num_segments= nn.Linear(self.input_processor_output_size, output_num_segments_size).to(DEVICE)
        self.output_y0          = nn.Linear(self.input_processor_output_size, output_y0_size).to(DEVICE)
        self.output_goal        = nn.Linear(self.input_processor_output_size, output_goal_size).to(DEVICE)
        # self.output_pos         = nn.Linear(self.input_processor_output_size, output_pos_size).to(DEVICE)
        self.latent_w           = nn.Linear(self.input_processor_output_size, self.max_segments * self.latent_w_size).to(DEVICE)
        self.output_tau         = nn.Linear(self.input_processor_output_size, output_tau_size).to(DEVICE)

    def forward(self, x):
        x           = self.input_processor(x)
        batch_s     = x.shape[0]

        latent_w    = self.latent_w(x).reshape(batch_s * self.max_segments, self.latent_w_size)

        dmp_num_segments = self.output_num_segments(x).reshape(batch_s, 1)
        dmp_y0      = self.output_y0(x).reshape(batch_s, self.max_segments, self.dof)
        dmp_goal    = self.output_goal(x).reshape(batch_s, self.max_segments, self.dof)

        # dmp_pos     = self.output_pos(x).reshape(batch_s, self.max_segments + 1, self.dof)
        # dmp_y0      = dmp_pos[:, :-1]
        # dmp_goal    = dmp_pos[:, 1:]

        dmp_weights = self.dmp_weight_decoder(latent_w).reshape(batch_s, self.max_segments, self.dof, self.n_bf)
        dmp_tau     = self.output_tau(x).reshape(batch_s, self.max_segments)
        # dmp_tau     = torch.abs(self.output_tau(x).reshape(batch_s, self.max_segments))

        return [dmp_num_segments, dmp_y0, dmp_goal, dmp_weights, dmp_tau]

# class FineTuningDSDNet(nn.Module):
#     def __init__(self, train_param):
#         super().__init__()
#         self.train_param        = train_param
#         self.model_param        = self.train_param.model_param
#         self.dmp_param          = self.model_param.dmp_param

#         if self.model_param.backbone_option == 'keypointrcnn_resnet50_fpn':
#             self.backbone           = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
#             self.targets            = None

#         if self.model_param.eval:
#             self.backbone.eval()

#         self.max_segments       = self.model_param.max_segments
#         self.dof                = self.dmp_param.dof
#         self.n_bf               = self.dmp_param.n_bf
#         self.latent_w_size      = self.model_param.latent_w_size

#         if self.model_param.input_mode == ['image']:
#             self.input_processor    = ImageInputProcessor(self.train_param)
#         self.input_processor_output_size = self.input_processor.output_size
#         self.dmp_weight_decoder = DMPWeightDecoder(self.train_param, input_size = self.latent_w_size)

#         output_num_segments_size= 1
#         output_y0_size          = self.max_segments * self.dof
#         output_goal_size        = self.max_segments * self.dof
#         # output_pos_size        = (self.max_segments + 1) * self.dof
#         output_tau_size         = self.max_segments

#         self.output_num_segments= nn.Linear(self.input_processor_output_size, output_num_segments_size).to(DEVICE)
#         self.output_y0          = nn.Linear(self.input_processor_output_size, output_y0_size).to(DEVICE)
#         self.output_goal        = nn.Linear(self.input_processor_output_size, output_goal_size).to(DEVICE)
#         # self.output_pos         = nn.Linear(self.input_processor_output_size, output_pos_size).to(DEVICE)
#         self.latent_w           = nn.Linear(self.input_processor_output_size, self.max_segments * self.latent_w_size).to(DEVICE)
#         self.output_tau         = nn.Linear(self.input_processor_output_size, output_tau_size).to(DEVICE)

#     def forward(self, x):
#         # x           = self.input_processor(x)
#         images, targets = self.backbone(x, self.targets)
#         features        = self.backbone(images.tensors)
#         batch_s         = x.shape[0]

#         latent_w    = self.latent_w(x).reshape(batch_s * self.max_segments, self.latent_w_size)

#         dmp_num_segments = self.output_num_segments(x).reshape(batch_s, 1)
#         dmp_y0      = self.output_y0(x).reshape(batch_s, self.max_segments, self.dof)
#         dmp_goal    = self.output_goal(x).reshape(batch_s, self.max_segments, self.dof)

#         # dmp_pos     = self.output_pos(x).reshape(batch_s, self.max_segments + 1, self.dof)
#         # dmp_y0      = dmp_pos[:, :-1]
#         # dmp_goal    = dmp_pos[:, 1:]

#         dmp_weights = self.dmp_weight_decoder(latent_w).reshape(batch_s, self.max_segments, self.dof, self.n_bf)
#         dmp_tau     = self.output_tau(x).reshape(batch_s, self.max_segments)
#         # dmp_tau     = torch.abs(self.output_tau(x).reshape(batch_s, self.max_segments))

#         return [dmp_num_segments, dmp_y0, dmp_goal, dmp_weights, dmp_tau]

class DSDNetV2(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_observable_pos = self.model_param.max_observable_pos
        self.max_segments       = self.model_param.max_segments
        self.dof                = self.dmp_param.dof
        self.n_bf               = self.dmp_param.n_bf
        self.latent_w_size      = self.model_param.latent_w_size

        if self.model_param.input_mode == ['image']:
            self.input_processor    = ImageInputProcessor(self.train_param)
        self.input_processor_output_size = self.input_processor.output_size
        self.dmp_weight_decoder = DMPWeightDecoder(self.train_param, input_size = self.latent_w_size)

        output_num_segments_size= 1
        output_y0_size          = self.max_segments * self.dof
        output_goal_size        = self.max_segments * self.dof
        # output_pos_size        = (self.max_segments + 1) * self.dof
        output_tau_size         = self.max_segments

        self.latent_observable_pos     = nn.Linear(self.input_processor_output_size, self.max_observable_pos * 2).to(DEVICE)
        self.output_num_segments= nn.Linear(self.max_observable_pos * 2, output_num_segments_size).to(DEVICE)
        self.output_y0          = nn.Linear(self.max_observable_pos * 2, output_y0_size).to(DEVICE)
        self.output_goal        = nn.Linear(self.max_observable_pos * 2, output_goal_size).to(DEVICE)
        # self.output_pos         = nn.Linear(self.max_projected_points * 2, output_pos_size).to(DEVICE)

        self.latent_w           = nn.Linear(self.input_processor_output_size, self.max_segments * self.latent_w_size).to(DEVICE)
        self.output_tau         = nn.Linear(self.max_segments * self.latent_w_size, output_tau_size).to(DEVICE)

    def forward(self, x):
        x           = self.input_processor(x)
        batch_s     = x.shape[0]

        latent_observable_pos = self.latent_observable_pos(x)
        observable_pos = latent_observable_pos.reshape(batch_s, self.max_observable_pos, 2)
        dmp_num_segments = self.output_num_segments(latent_observable_pos).reshape(batch_s, 1)
        dmp_y0      = self.output_y0(latent_observable_pos).reshape(batch_s, self.max_segments, self.dof)
        dmp_goal    = self.output_goal(latent_observable_pos).reshape(batch_s, self.max_segments, self.dof)
        # dmp_pos     = self.output_pos(latent_img_pos).reshape(batch_s, self.max_segments + 1, self.dof)
        # dmp_y0      = dmp_pos[:, :-1]
        # dmp_goal    = dmp_pos[:, 1:]

        latent_w    = self.latent_w(x).reshape(batch_s * self.max_segments, self.latent_w_size)
        dmp_weights = self.dmp_weight_decoder(latent_w).reshape(batch_s, self.max_segments, self.dof, self.n_bf)
        dmp_tau     = self.output_tau(latent_w.reshape(batch_s, self.max_segments * self.latent_w_size))
        # dmp_tau     = torch.abs(self.output_tau(x).reshape(batch_s, self.max_segments))

        return [observable_pos, dmp_num_segments, dmp_y0, dmp_goal, dmp_weights, dmp_tau]

class CIMEDNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.dof                = self.dmp_param.dof
        self.n_bf               = self.dmp_param.n_bf

        if self.model_param.input_mode == ['image']:
            self.input_processor    = ImageInputProcessor(self.train_param)
        self.input_processor_output_size = self.input_processor.output_size

        output_y0_size          = self.dof
        output_goal_size        = self.dof
        output_w_size           = self.dof * self.n_bf
        output_tau_size         = 1

        self.output_y0          = nn.Linear(self.input_processor_output_size, output_y0_size).to(DEVICE)
        self.output_goal        = nn.Linear(self.input_processor_output_size, output_goal_size).to(DEVICE)
        self.output_w           = nn.Linear(self.input_processor_output_size, output_w_size).to(DEVICE)
        self.output_tau         = nn.Linear(self.input_processor_output_size, output_tau_size).to(DEVICE)
        
    def forward(self, x):
        x           = self.input_processor(x)
        batch_s     = x.shape[0]

        dmp_y0      = self.output_y0(x).reshape(batch_s, self.dof)
        dmp_goal    = self.output_goal(x).reshape(batch_s, self.dof)
        dmp_weights = self.output_w(x).reshape(batch_s, self.dof, self.n_bf)
        dmp_tau     = self.output_tau(x).reshape(batch_s, 1)

        return [dmp_y0, dmp_goal, dmp_weights, dmp_tau]

class PosNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.max_observable_pos = self.model_param.max_observable_pos

        if self.model_param.input_mode == ['image']:
            self.input_processor    = ImageInputProcessor(self.train_param)
        self.input_processor_output_size = self.input_processor.output_size

        self.latent_observable_pos     = nn.Linear(self.input_processor_output_size, self.max_observable_pos * 2).to(DEVICE)

    def forward(self, x):
        x           = self.input_processor(x)
        batch_s     = x.shape[0]
        latent_observable_pos = self.latent_observable_pos(x)
        observable_pos = latent_observable_pos.reshape(batch_s, self.max_observable_pos, 2)

        return [observable_pos]

class DSDPosNet(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        self.dmp_param          = self.model_param.dmp_param
        self.dof                = self.dmp_param.dof
        self.max_observable_pos = self.model_param.max_observable_pos
        self.max_segments       = self.model_param.max_segments

        output_y0_size          = self.max_segments * self.dof
        output_goal_size        = self.max_segments * self.dof

        self.pos_net            = self.model_param.pos_net

        self.output_y0          = nn.Linear(self.max_observable_pos * 2, output_y0_size).to(DEVICE)
        self.output_goal        = nn.Linear(self.max_observable_pos * 2, output_goal_size).to(DEVICE)

    def forward(self, x):
        batch_s                 = x[self.model_param.input_mode[0]].shape[0]
        x                       = self.pos_net(x)[0].reshape(batch_s, self.max_observable_pos * 2)

        dmp_y0                  = self.output_y0(x).reshape(batch_s, self.max_segments, self.dof)
        dmp_goal                = self.output_goal(x).reshape(batch_s, self.max_segments, self.dof)

        return [dmp_y0, dmp_goal]

class AutoEncoder(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        