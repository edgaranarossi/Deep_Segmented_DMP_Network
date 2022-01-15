from __future__ import print_function, division
from torch._C import Value
from torch.utils.data import Dataset, DataLoader
from torch import tensor, from_numpy
import torch
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sys
sys.path.append('/home/edgar/rllab/tools/DMP/imednet')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Mapping:
    y_max = 1
    y_min = -1
    x_max = []
    x_min = []

class MinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max

class Scale:
    def __init__(self, y_old, w_old, y_new = [-1.0, 1.0], w_new = [-1.0, 1.0]):
        self.y_old = MinMax(y_old[0], y_old[1])
        self.y_new = MinMax(y_new[0], y_new[1])
        self.w_old = MinMax(w_old[0], w_old[1])
        self.w_new = MinMax(w_new[0], w_new[1])

    def normalize(self, X):
        X_np = np.array(X)
        X_normalized = np.zeros_like(X_np)
        X_normalized[:, :4] = (self.y_new.max - self.y_new.min) * \
                              (X_np[:, :4] - self.y_old.min) / \
                              (self.y_old.max - self.y_old.min) + \
                              self.y_new.min
        X_normalized[:, 4:] = (self.w_new.max - self.w_new.min) * \
                              (X_np[:, 4:] - self.w_old.min) / \
                              (self.w_old.max - self.w_old.min) + \
                              self.w_new.min
        return X_normalized

    def denormalize(self, X):
        X_np = np.array(X)
        X_denormalized = np.zeros_like(X_np)
        X_denormalized[:, :4] = (X_np[:, :4] - self.y_new.min) / \
                                (self.y_new.max - self.y_new.min) * \
                                (self.y_old.max - self.y_old.min) + \
                                self.y_old.min
        X_denormalized[:, 4:] = (X_np[:, 4:] - self.w_new.min) / \
                                (self.w_new.max - self.w_new.min) * \
                                (self.w_old.max - self.w_old.min) + \
                                self.w_old.min
        return X_denormalized

class MatDataLoader:
    def __init__(self, mat_path, data_limit = None, include_tau = False):
        data                            = sio.loadmat(mat_path)

        if data_limit is None:
            self.images                 = data['imageArray']
            self.dmp_outputs            = data['outputs']
            self.traj                   = data['trajArray']
            self.captions               = data['caption']
        else:
            self.images                 = data['imageArray'][:data_limit]
            self.dmp_outputs            = data['outputs'][:data_limit]
            self.traj                   = data['trajArray'][:data_limit]
            self.captions               = data['text'][:data_limit]

        self.dmp_scaling                = Mapping()
        self.dmp_scaling.x_max          = data['scaling']['x_max'][0,0][0]
        self.dmp_scaling.x_min          = data['scaling']['x_min'][0,0][0]
        self.dmp_scaling.y_max          = data['scaling']['y_max'][0,0][0,0]
        self.dmp_scaling.y_min          = data['scaling']['y_min'][0,0][0,0]

        self.combined_inputs            = []
        self.combined_outputs           = []
        if include_tau:
            begin_idx = None
        else:
            begin_idx = 1
        self.tau = self.dmp_outputs[0][0]
        for idx in range(len(self.images)):
            self.combined_inputs.append({
                'image'                 : torch.from_numpy(self.images[idx]).float().to(DEVICE),
                'caption'               : self.captions[idx],
            })
            self.combined_outputs.append({
                'outputs'               : torch.from_numpy(self.dmp_outputs[idx][begin_idx:]).float().to(DEVICE),
                'trajectory'            : torch.from_numpy(self.traj[idx]).float().to(DEVICE),
            })

    def getData(self):
        return self.combined_inputs, self.combined_outputs

    def getTau(self):
        return self.tau

    def getDataLoader(self, data_ratio = [7, 2, 1], batch_size = 50, input_mode = 'image',output_mode = 'dmp'):
        X_train, X_val, Y_train, Y_val  = train_test_split(
                                                        self.combined_inputs,
                                                        self.combined_outputs,
                                                        test_size=(data_ratio[1]+data_ratio[2])/sum(data_ratio))
        X_val, X_test, Y_val, Y_test    = train_test_split(
                                                        X_val,
                                                        Y_val, 
                                                        test_size=data_ratio[2]/(data_ratio[1]+data_ratio[2]))

        train_dataset                   = DMPDataset(X = X_train, Y = Y_train, input_mode = input_mode, output_mode = output_mode)
        val_dataset                     = DMPDataset(X = X_val, Y = Y_val, input_mode = input_mode, output_mode = output_mode)
        test_dataset                    = DMPDataset(X = X_test, Y = Y_test, input_mode = input_mode, output_mode = output_mode)

        train_loader                    = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle = True)
        val_loader                      = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle = True)
        test_loader                     = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle = True)

        return [train_loader, val_loader, test_loader], self.dmp_scaling

class PickleDataLoader:
    def __init__(self, pkl_path, data_limit = None, include_tau = False):
        begin_idx                           = 1 if include_tau else None
        self.data                           = pickle.load(open(pkl_path, 'rb'))

        if 'dmp_outputs_unscaled' in self.data:
            self.dmp_outputs                = self.data['dmp_outputs_unscaled'][:data_limit, begin_idx:]
            self.tau                        = self.dmp_outputs[0][0] if include_tau else 1
        if 'dmp_outputs_scaled' in self.data:
            self.dmp_outputs_scaled         = self.data['dmp_outputs_scaled'][:data_limit, begin_idx:]
            self.tau                        = self.dmp_outputs_scaled[0][0] if include_tau else 1
        if 'segmented_dmp_outputs_unscaled' in self.data:
            self.segment_dmp_outputs        = self.data['segmented_dmp_outputs_unscaled'][:data_limit, begin_idx:]
        if 'segmented_dmp_outputs_scaled' in self.data:
            self.segment_dmp_outputs_scaled = self.data['segmented_dmp_outputs_scaled'][:data_limit, begin_idx:]

        if 'image' in self.data:
            self.images                     = self.data['image'][:data_limit]
        if 'caption' in self.data:
            self.captions                   = self.data['caption'][:data_limit]
        if 'cut_distance' in self.data:
            self.cut_distance               = self.data['cut_distance'][:data_limit]
        if 'segmented_dmp_segments' in self.data:
            self.segments                   = self.data['segmented_dmp_segments']
        if 'traj' in self.data:
            self.traj                       = self.data['traj'][:data_limit]
        if 'dmp_traj' in self.data:
            self.dmp_traj                       = self.data['dmp_traj'][:data_limit]
        if 'segmented_dmp_traj' in self.data:
            self.segmented_dmp_traj         = self.data['segmented_dmp_traj'][:data_limit]

        if 'dmp_scaling' in self.data:
            self.dmp_scaling                = self.data['dmp_scaling']
            # self.dmp_scaling[0]             = self.dmp_scaling[0][begin_idx:]
            # self.dmp_scaling[1]             = self.dmp_scaling[1][begin_idx:]
        if 'segmented_dmp_scaling' in self.data:
            self.segmented_dmp_scaling      = self.data['segmented_dmp_scaling']
            # self.segmented_dmp_scaling[0]   = self.segmented_dmp_scaling[0][begin_idx:]
            # self.segmented_dmp_scaling[1]   = self.segmented_dmp_scaling[1][begin_idx:]

        # self.dmp_scaling.x_max          = from_numpy(self.dmp_scaling.x_max[begin_idx:]).to(DEVICE)
        # self.dmp_scaling.x_min          = from_numpy(self.dmp_scaling.x_min[begin_idx:]).to(DEVICE)
        # self.dmp_scaling.y_max          = tensor(self.dmp_scaling.y_max).to(DEVICE)
        # self.dmp_scaling.y_min          = tensor(self.dmp_scaling.y_min).to(DEVICE)            

        self.combined_inputs            = []
        self.combined_outputs           = []
        
        for idx in range(len(self.dmp_outputs)):
            inputs = {}
            if 'image' in self.data:
                inputs['image']                         = torch.from_numpy(self.images[idx]).float().to(DEVICE)
            if 'caption' in self.data:
                inputs['caption']                       = self.captions[idx]
            if 'dmp_outputs_unscaled' in self.data:
                inputs['dmp_param']                     = torch.from_numpy(self.dmp_outputs[idx][begin_idx:]).float().to(DEVICE)
            if 'dmp_outputs_scaled' in self.data:
                inputs['dmp_param_scaled']              = torch.from_numpy(self.dmp_outputs_scaled[idx][begin_idx:]).float().to(DEVICE)
            if 'segmented_dmp_outputs_unscaled' in self.data:
                inputs['segmented_dmp_param']           = torch.from_numpy(self.segment_dmp_outputs[idx][begin_idx:]).float().to(DEVICE)
            if 'segmented_dmp_outputs_scaled' in self.data:
                inputs['segmented_dmp_param_scaled']    = torch.from_numpy(self.segment_dmp_outputs_scaled[idx][begin_idx:]).float().to(DEVICE)
            self.combined_inputs.append(inputs)

            outputs = {}
            if 'dmp_outputs_unscaled' in self.data:
                outputs['dmp_param']                    = torch.from_numpy(self.dmp_outputs[idx][begin_idx:]).float().to(DEVICE)
            if 'dmp_outputs_scaled' in self.data:
                outputs['dmp_param_scaled']             = torch.from_numpy(self.dmp_outputs_scaled[idx][begin_idx:]).float().to(DEVICE)
            if 'segmented_dmp_outputs_unscaled' in self.data:
                outputs['segmented_dmp_param']          = torch.from_numpy(self.segment_dmp_outputs[idx][begin_idx:]).float().to(DEVICE)
            if 'segmented_dmp_outputs_scaled' in self.data:
                outputs['segmented_dmp_param_scaled']   = torch.from_numpy(self.segment_dmp_outputs_scaled[idx][begin_idx:]).float().to(DEVICE)
            if 'traj' in self.data:
                outputs['traj']                         = torch.from_numpy(self.traj[idx]).float().to(DEVICE)
            if 'dmp_traj' in self.data:
                outputs['dmp_traj']                         = torch.from_numpy(self.dmp_traj[idx]).float().to(DEVICE)
            if 'segmented_dmp_traj' in self.data:
                outputs['segmented_dmp_traj']           = torch.from_numpy(self.segmented_dmp_traj[idx]).float().to(DEVICE)
            self.combined_outputs.append(outputs)

    def getData(self):
        return self.combined_inputs, self.combined_outputs

    def getTau(self):
        return self.tau

    def getDataLoader(self, input_mode, output_mode, data_ratio = [7, 2, 1], batch_size = 50):
        """
        Input mode = ['image',
                      'caption',
                      'dmp_param',
                      'dmp_param_scaled',
                      'segmented_dmp_param',
                      'segmented_dmp_param_scaled']
        Output mode = ['dmp_param',
                       'dmp_param_scaled',
                       'segmented_dmp_param',
                       'segmented_dmp_param_scaled',
                       'traj',
                       'dmp_traj',
                       'segmented_dmp_traj']
        """
        X_train, X_val, Y_train, Y_val  = train_test_split(
                                                        self.combined_inputs,
                                                        self.combined_outputs,
                                                        test_size=(data_ratio[1]+data_ratio[2])/sum(data_ratio))
        X_val, X_test, Y_val, Y_test    = train_test_split(
                                                        X_val,
                                                        Y_val, 
                                                        test_size=data_ratio[2]/(data_ratio[1]+data_ratio[2]))

        train_dataset                   = DMPDataset(X = X_train, Y = Y_train, input_mode = input_mode, output_mode = output_mode)
        val_dataset                     = DMPDataset(X = X_val, Y = Y_val, input_mode = input_mode, output_mode = output_mode)
        test_dataset                    = DMPDataset(X = X_test, Y = Y_test, input_mode = input_mode, output_mode = output_mode)

        train_loader                    = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader                      = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)
        test_loader                     = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

        if 'segmented' in output_mode:
            if 'segmented_dmp_scaling' in self.data:
                scaling = self.segmented_dmp_scaling
        else:
            if 'dmp_scaling' in self.data:
                scaling = self.dmp_scaling

        if 'segmented_dmp_scaling' in self.data or 'dmp_scaling' in self.data:
            return [train_loader, val_loader, test_loader], scaling
        else:
            return [train_loader, val_loader, test_loader], ''
    
class DMPDataset(Dataset):
    def __init__(self, X, input_mode, output_mode, Y = None):
        self.X                          = X
        self.Y                          = Y
        self.input_mode                 = input_mode
        self.output_mode                = output_mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.input_mode == 'image':
            X                           = self.X[idx]['image']
        elif self.input_mode == 'caption':
            X                           = self.X[idx]['caption']
        elif self.input_mode == 'dmp_param':
            X                           = self.X[idx]['dmp_param']
        elif self.input_mode == 'dmp_param_scaled':
            X                           = self.X[idx]['dmp_param_scaled']
        elif self.input_mode == 'segmented_dmp_param':
            X                           = self.X[idx]['segmented_dmp_param']
        elif self.input_mode == 'segmented_dmp_param_scaled':
            X                           = self.X[idx]['segmented_dmp_param_scaled']

        if self.Y != None:
            if self.output_mode == 'dmp_param':
                Y                       = self.Y[idx]['dmp_param']
            elif self.output_mode == 'dmp_param_scaled':
                Y                       = self.Y[idx]['dmp_param_scaled']
            elif self.output_mode == 'segmented_dmp_param':
                Y                       = self.Y[idx]['segmented_dmp_param']
            elif self.output_mode == 'segmented_dmp_param_scaled':
                Y                       = self.Y[idx]['segmented_dmp_param_scaled']
            elif self.output_mode == 'traj':
                Y                       = self.Y[idx]['traj']
            elif self.output_mode == 'dmp_traj':
                Y                       = self.Y[idx]['dmp_traj']
            elif self.output_mode == 'segmented_dmp_traj':
                Y                       = self.Y[idx]['segmented_dmp_traj']
            return (X, Y)
        
        return X