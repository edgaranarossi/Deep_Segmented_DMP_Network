#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:44:42 2022

@author: edgar
"""
import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from numpy.random import randint
from numpy import array
from utils.networks import *
from matplotlib import pyplot as plt
from pydmps import DMPs_discrete

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester:
    def __init__(self, dataloaders: [DataLoader, DataLoader, DataLoader]):
        self.train_loaders = dataloaders[0]
        self.val_loaders = dataloaders[1]
        self.test_loaders = dataloaders[2]
        self.train_dataset = self.train_loaders.dataset
        self.val_dataset = self.val_loaders.dataset
        self.test_dataset = self.test_loaders.dataset
        
        self.read_dataset()
    
    def read_dataset(self):
        print('Dataset lengths:')
        print('- Train dataset: {}'.format(len(self.train_dataset)))
        print('- Val dataset  : {}'.format(len(self.val_dataset)))
        print('- Test dataset : {}'.format(len(self.test_dataset)))
        print('Total          : {}\n'.format(len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)))
        print('Available inputs : {}'.format([i for i in self.train_dataset[0][0]]))
        print('Available outputs: {}'.format([i for i in self.train_dataset[0][1]]))
        
        if len([i for i in self.train_dataset[0][0]]) == 1:
            self.input_mode = [i for i in self.train_dataset[0][0]][0]
        else:
            print('Set input mode with set_input_mode')
            
    def set_input_mode(self, mode):
        self.input_mode = mode
        
    def predict_test(self, model, idx = None):
        scaler = model.train_param.scaler
        if idx == None: idx = randint(len(self.test_dataset))
        input_data, output_data = self.test_dataset[idx]
        data = input_data[self.input_mode].to(DEVICE)
        data = torch.unsqueeze(data, 0).to(DEVICE)
        pred = model(data)
        
        rescaled_pred = []
        for i, key in enumerate(model.model_param.output_mode):
            rescaled_pred.append(scaler[key].denormalize(pred[i]))
            
        return rescaled_pred, input_data, output_data, pred
        
    def predict_val(self, model, idx = None):
        scaler = model.train_param.scaler
        if idx == None: idx = randint(len(self.val_dataset))
        input_data, output_data = self.val_dataset[idx]
        data = input_data[self.input_mode].to(DEVICE)
        data = torch.unsqueeze(data, 0).to(DEVICE)
        pred = model(data)
        
        rescaled_pred = []
        for i, key in enumerate(model.model_param.output_mode):
            rescaled_pred.append(scaler[key].denormalize(pred[i]))
            
        return rescaled_pred, input_data, output_data, pred
        
    def predict_train(self, model, idx = None):
        scaler = model.train_param.scaler
        if idx == None: idx = randint(len(self.train_dataset))
        input_data, output_data = self.train_dataset[idx]
        data = input_data[self.input_mode].to(DEVICE)
        data = torch.unsqueeze(data, 0).to(DEVICE)
        pred = model(data)
        
        rescaled_pred = []
        for i, key in enumerate(model.model_param.output_mode):
            rescaled_pred.append(scaler[key].denormalize(pred[i]))
            
        return rescaled_pred, input_data, output_data, pred

def str_to_ndarray(s):
    arr = []
    for line in s.split(';'):
        if len(line) > 0:
            l = []
            for w in line.split(','):
                l.append(float(w))
            arr.append(l)
    return array(arr)

def plot_data_separately(data, names, titles):
    assert len(data) == len(names), "Number of data to plot doesn't match number of names"
    fig, axs = plt.subplots(len(data), 1)
    for title in range(len(data)):
        for name in range(len(data[title])):
            axs[title].scatter(range(len(data[title][name])), data[title][name])
            axs[title].legend(names)
            axs[title].set_title(titles[title])

def tensor_to_ndarray(t_list):
    return [t.detach().cpu().numpy()[0] for t in t_list]

def generate_dmp_traj(y0, goal, w, ay, dt, tau = 1.0):
    # print(w.shape[1])
    dmp = DMPs_discrete(n_dmps = w.shape[0], 
                        n_bfs = w.shape[1], 
                        ay = np.ones(w.shape[0]) * ay, 
                        dt = dt)
    dmp.w = w
    dmp.y0 = y0
    dmp.goal = goal
    y_track, dy_track, ddy_track = dmp.rollout(tau = tau)
    return y_track, dy_track, ddy_track
#%
if __name__=='__main__':
    MODEL_NAME = 'Model_SegmentedDMPNetwork_2022-05-12_02-36-34'
    MODEL_NAME_2 = 'Model_NormalDMPJoinedNetwork_2022-05-12_02-37-30'
    
    MODEL_ROOT = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/'
    MODEL_SUB_ROOT = MODEL_NAME.split('_')[1]
    MODEL_SUB_ROOT_2 = MODEL_NAME_2.split('_')[1]
    
    assert exists(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME))
    data_loaders = pkl.load(open(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME, 'data_loaders.pkl'), 'rb'))    
    train_param_segment = pkl.load(open(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME, 'train-model-dmp_param.pkl'), 'rb'))
    train_param_deep_dmp = pkl.load(open(join(MODEL_ROOT, MODEL_SUB_ROOT_2, MODEL_NAME_2, 'train-model-dmp_param.pkl'), 'rb'))
    
    tester = Tester(data_loaders)
    model_segmented = DSDNetV1(train_param_segment).to(DEVICE)
    model_segmented.load_state_dict(torch.load(join(MODEL_ROOT, MODEL_SUB_ROOT, MODEL_NAME, 'best_net_parameters')))
    
    model_deep_dmp = NormalDMPJoinedNetwork(train_param_deep_dmp).to(DEVICE)
    model_deep_dmp.load_state_dict(torch.load(join(MODEL_ROOT, MODEL_SUB_ROOT_2, MODEL_NAME_2, 'best_net_parameters')))
#%%    
    pred_segmented, input_data, output_data, unscaled_pred_segmented = tester.predict_test(model_segmented, idx = 0)
    pred_deep_dmp, _, _, _ = tester.predict_test(model_deep_dmp, idx = 0)
    
    original_trajectory = str_to_ndarray(output_data['original_trajectory'])
    
    pred_segmented_np = tensor_to_ndarray(pred_segmented)
    pred_deep_dmp_np = tensor_to_ndarray(pred_deep_dmp)
#%%    
    pred_segmented, input_data, output_data, unscaled_pred_segmented = tester.predict_val(model_segmented, idx = 0)
    pred_deep_dmp, _, _, _ = tester.predict_val(model_deep_dmp, idx = 0)
    
    original_trajectory = str_to_ndarray(output_data['original_trajectory'])
    
    pred_segmented_np = tensor_to_ndarray(pred_segmented)
    pred_deep_dmp_np = tensor_to_ndarray(pred_deep_dmp)

#%%    
    pred_segmented, input_data, output_data, unscaled_pred_segmented = tester.predict_train(model_segmented, idx = 0)
    pred_deep_dmp, _, _, _ = tester.predict_train(model_deep_dmp, idx = 0)
    
    original_trajectory = str_to_ndarray(output_data['original_trajectory'])
    
    pred_segmented_np = tensor_to_ndarray(pred_segmented)
    pred_deep_dmp_np = tensor_to_ndarray(pred_deep_dmp)
#%%    
    titles = ['Axis-X', 'Axis-Y', 'Axis-Z', 'Vel-X', 'Vel-Y', 'Vel-Z']
    names = ['Original', 'Deep-DMP', 'Segmented-Deep-DMP']
    track_deep_dmp = generate_dmp_traj(y0 = pred_deep_dmp_np[0],
                                       goal = pred_deep_dmp_np[1],
                                       w = pred_deep_dmp_np[2],
                                       ay = 25,
                                       dt = 0.001,
                                       tau = 1.0)
    tracks_segmented = []
    for i in range(pred_segmented_np[0].shape[0]):
        tracks_segmented.append(generate_dmp_traj(y0 = pred_segmented_np[0][i],
                                                  goal = pred_segmented_np[1][i],
                                                  w = pred_segmented_np[2][i],
                                                  ay = 10,
                                                  dt = 0.015,
                                                  # tau = pred_segmented_np[3][i]))
                                                  tau = 1.0))
    y_track_segmented = []
    dy_track_segmented = []
    ddy_track_segmented = []
    for i in tracks_segmented:
        y_track_segmented.append(i[0])
        dy_track_segmented.append(i[1])
        ddy_track_segmented.append(i[2])
    y_track_segmented = array(y_track_segmented).reshape(-1, train_param_segment.model_param.dmp_param.dof)
    dy_track_segmented = array(dy_track_segmented).reshape(-1, train_param_segment.model_param.dmp_param.dof)
    ddy_track_segmented = array(ddy_track_segmented).reshape(-1, train_param_segment.model_param.dmp_param.dof)
    track_segmented = [y_track_segmented, dy_track_segmented, ddy_track_segmented]
#%%
axis = 0
#%%
plt.scatter(range(original_trajectory[61:143, axis].shape[0]), original_trajectory[61:143, axis] - original_trajectory[61:143, axis].min())
#%%
plt.scatter(range(original_trajectory[98:163, axis].shape[0]), original_trajectory[98:163, axis] - original_trajectory[98:163, axis].min())
#%%
plt.scatter(range(original_trajectory[143:202, axis].shape[0]), original_trajectory[143:202, axis] - original_trajectory[143:202, axis].min())
#%%
plt.plot(track_segmented[0][:,0])
#%%
plt.plot(track_deep_dmp[0][:,0])

#%%
from pydmd import DMD

dmd_1 = DMD(svd_rank=1.0, tlsq_rank=0, exact=True, opt=True)
dmd_2 = DMD(svd_rank=1.0, tlsq_rank=0, exact=True, opt=True)
dmd_3 = DMD(svd_rank=1.0, tlsq_rank=0, exact=True, opt=True)

axis = 2
traj_1 = original_trajectory[61:143, axis] - original_trajectory[61:143, axis].min()
traj_2 = original_trajectory[98:163, axis] - original_trajectory[98:163, axis].min()
traj_3 = original_trajectory[143:202, axis] - original_trajectory[143:202, axis].min()

dmd_1.fit(list(traj_1.reshape(-1,1)))
dmd_2.fit(list(traj_2.reshape(-1,1)))
dmd_3.fit(list(traj_3.reshape(-1,1)))

# plt.figure()
# plt.title('Axis {}'.format(axis))
# plt.scatter(range(len(traj_1)), traj_1)
# plt.scatter(range(len(traj_2)), traj_2)
# plt.scatter(range(len(traj_3)), traj_3)
# plt.legend(['1', '2', '3'])
# plt.show()

mse_1_3 = ((dmd_1.modes - dmd_3.modes)**2).mean()
mse_1_2 = ((dmd_1.modes - dmd_2.modes)**2).mean()
mse_2_3 = ((dmd_2.modes - dmd_3.modes)**2).mean()

print(mse_1_3)
print(mse_1_2)
print(mse_2_3)
print(mse_1_3 < mse_1_2)
print(mse_1_3 < mse_2_3)
print(mse_1_3 < mse_1_2 and mse_1_3 < mse_2_3)

print()

mse_1_3 = ((dmd_1.atilde - dmd_3.atilde)**2).mean()
mse_1_2 = ((dmd_1.atilde - dmd_2.atilde)**2).mean()
mse_2_3 = ((dmd_2.atilde - dmd_3.atilde)**2).mean()

print(mse_1_3)
print(mse_1_2)
print(mse_2_3)
print(mse_1_3 < mse_1_2)
print(mse_1_3 < mse_2_3)
print(mse_1_3 < mse_1_2 and mse_1_3 < mse_2_3)