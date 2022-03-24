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
from utils.networks import SegmentDMPCNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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