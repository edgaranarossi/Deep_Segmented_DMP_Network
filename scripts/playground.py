#%%
from PIL import Image, ImageOps
from numpy import array, flipud, where
from matplotlib import pyplot as plt
from os import listdir
from os.path import isdir, join
from torch import nn, flatten, from_numpy, zeros, cat
from torch.nn import ModuleList
import torch.nn.functional as F
import torch
import pickle as pkl
from utils.pydmps_torch import DMPs_discrete_torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SegmentDMPCNN(nn.Module):
    def __init__(self, train_param):
        super().__init__()
        self.train_param        = train_param
        self.model_param        = self.train_param.model_param
        train_param.model_param.dmp_param          = self.model_param.dmp_param
        self.max_segments       = self.model_param.max_segments
        self.num_position       = self.max_segments + 1
        self.dof                = train_param.model_param.dmp_param.dof
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
        output_w_size               = self.max_segments * self.dof * train_param.model_param.dmp_param.n_bf

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
        w               = self.output_w(x).reshape(batch_s, self.max_segments, self.dof * train_param.model_param.dmp_param.n_bf)

        y0 = pos[:, 0]
        goals = pos[:, 1:] 

        return [num_segments, y0, goals, w]
    


img_path = 'D:\\rllab\\scripts\\dmp\\scripts'
train_param_path = join(img_path, 'train-model-dmp_param.pkl')
best_param_path = join(img_path, 'best_net_parameters')
train_param = pkl.load(open(train_param_path, 'rb'))
model = SegmentDMPCNN(train_param)
model.load_state_dict(torch.load(best_param_path))
model.eval()
#%%
img_names = [i for i in listdir(img_path) if not isdir(join(img_path, i)) and i[-3:] == 'jpg' and i[:4] == 'test']
# img_name = 'test_0'
for img_name in img_names:
    original_img = array(Image.open(join(img_path, img_name)))
    # img_1 = array(Image.open(join(img_path, img_names[0])))
    # img_2 = array(Image.open(join(img_path, img_names[1])))
    
    # img = flipud(img_0) / 255
    img = original_img / 255
    threshold = 0.7
    img = img.mean(axis = 2)
    img = where(img > threshold, 0., 1.)
    
    img = array(Image.fromarray(img).resize((50, 50), Image.ANTIALIAS))
    original_img_resized = array(Image.fromarray(original_img).resize((50, 50), Image.ANTIALIAS))
    
    preds = model(from_numpy(img.reshape(1, 1, 50, 50)).to(DEVICE).float())
    
    rescaled_pred = []
    for idx, key in enumerate(train_param.model_param.keys_to_normalize):
        rescaled_pred.append(train_param.model_param.dmp_param.scale[key].denormalize(preds[idx][0]))
    
    num_segments_pred = int(torch.round(rescaled_pred[0]).reshape(1).item())
    # num_segments_pred = 15
    
    y_pred = zeros(num_segments_pred, int(1 / train_param.model_param.dmp_param.dt), train_param.model_param.dmp_param.dof).to(DEVICE)
    
    all_pos_pred = cat([rescaled_pred[1].reshape(1, train_param.model_param.dmp_param.dof, 1), rescaled_pred[2].reshape(-1, train_param.model_param.dmp_param.dof, 1)], dim = 0)
    
    y0s_pred = all_pos_pred[:-1]
    goals_pred = all_pos_pred[1:]
    
    dmp_pred = DMPs_discrete_torch(n_dmps = train_param.model_param.dmp_param.dof, 
                                   n_bfs = train_param.model_param.dmp_param.n_bf, 
                                   ay = train_param.model_param.dmp_param.ay, 
                                   dt = train_param.model_param.dmp_param.dt)
    # dmp_pred.y0 = rescaled_pred[1].reshape(1, train_param.model_param.dmp_param.dof, 1)
    dmp_pred.y0         = y0s_pred[:num_segments_pred]
    dmp_pred.goal       = goals_pred[:num_segments_pred]
    dmp_pred.w          = rescaled_pred[3][:num_segments_pred].reshape(num_segments_pred, train_param.model_param.dmp_param.dof, train_param.model_param.dmp_param.n_bf)
    y_track_pred, _, _  = dmp_pred.rollout()
    
    y_pred = y_track_pred.reshape(-1, train_param.model_param.dmp_param.dof)
    #%
    padding_x = 2
    padding_y = 1
    padding = array([[padding_x, padding_y]])
    
    multiplier = 28
    y_pred_np = ((y_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, train_param.model_param.dmp_param.dof)
    all_pos_pred_np = ((all_pos_pred.detach().cpu().numpy() * multiplier).reshape(-1, train_param.model_param.dmp_param.dof) + padding)
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(flipud(original_img_resized), cmap='Greys_r', origin = 'lower')
    plt.scatter(all_pos_pred_np[:num_segments_pred + 1, 0], all_pos_pred_np[:num_segments_pred + 1, 1], c = 'r', zorder = 6)
    plt.plot(y_pred_np[:,0], y_pred_np[:,1], c = 'r')
    plt.show()