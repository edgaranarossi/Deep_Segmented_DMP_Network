from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sys
sys.path.append('/home/edgar/rllab/tools/DMP/imednet')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class PickleDataLoader:
    def __init__(self, pkl_path, data_limit = None, include_tau = False):
        begin_idx                           = 1 if include_tau else None
        self.data                           = pickle.load(open(pkl_path, 'rb'))
        data_length                         = None
        self.with_scaling                   = False

        if 'dmp_outputs_unscaled' in self.data:
            print('dmp_outputs_unscaled')
            self.dmp_outputs                    = self.data['dmp_outputs_unscaled'][:data_limit, begin_idx:]
            self.tau                            = self.dmp_outputs[0][0] if include_tau else 1
            if data_length == None: data_length = len(self.dmp_outputs)
        if 'dmp_outputs_scaled' in self.data:
            self.dmp_outputs_scaled             = self.data['dmp_outputs_scaled'][:data_limit, begin_idx:]
            self.tau                            = self.dmp_outputs_scaled[0][0] if include_tau else 1
            if data_length == None: data_length = len(self.dmp_outputs_scaled)
        if 'segmented_dmp_outputs_unscaled' in self.data:
            self.segment_dmp_outputs            = self.data['segmented_dmp_outputs_unscaled'][:data_limit, begin_idx:]
            if data_length == None: data_length = len(self.segment_dmp_outputs)
        if 'segmented_dmp_outputs_scaled' in self.data:
            self.segment_dmp_outputs_scaled     = self.data['segmented_dmp_outputs_scaled'][:data_limit, begin_idx:]
            if data_length == None: data_length = len(self.segment_dmp_outputs_scaled)

        if 'segmented_dict_dmp_outputs' in self.data:
            self.segmented_dict_dmp_outputs     = self.data['segmented_dict_dmp_outputs'][:data_limit]
            if data_length == None: data_length = len(self.segmented_dict_dmp_outputs)
        if 'segmented_dict_dmp_types' in self.data:
            self.segmented_dict_dmp_types       = self.data['segmented_dict_dmp_types'][:data_limit]
            if data_length == None: data_length = len(self.segmented_dict_dmp_types)
        if 'image' in self.data:
            self.images                         = self.data['image'][:data_limit] / 255
            if data_length == None: data_length = len(self.images)
        if 'image_name' in self.data:
            self.image_names                    = self.data['image_name'][:data_limit]
            if data_length == None: data_length = len(self.image_names)
        if 'caption' in self.data:
            self.captions                       = self.data['caption'][:data_limit]
            if data_length == None: data_length = len(self.captions)
        if 'cut_distance' in self.data:
            self.cut_distance                   = self.data['cut_distance'][:data_limit]
            if data_length == None: data_length = len(self.cut_distance)
        if 'num_segments' in self.data:
            self.num_segments                   = np.array(self.data['num_segments']).reshape(-1, 1)
            # print(self.num_segments.shape)
            if data_length == None: data_length = len(self.num_segments)
        if 'traj' in self.data:
            self.traj                           = self.data['traj'][:data_limit] * 100
            if data_length == None: data_length = len(self.traj)
        if 'dmp_traj' in self.data:
            self.dmp_traj                       = self.data['dmp_traj'][:data_limit]
            if data_length == None: data_length = len(self.dmp_traj)
        if 'dmp_traj_padded' in self.data:
            self.dmp_traj_padded                = self.data['dmp_traj_padded'][:data_limit]
            if data_length == None: data_length = len(self.dmp_traj_padded)
        if 'dmp_traj_interpolated' in self.data:
            self.dmp_traj_interpolated          = self.data['dmp_traj_interpolated'][:data_limit]
            if data_length == None: data_length = len(self.dmp_traj_interpolated)
        if 'segmented_dmp_traj' in self.data:
            self.segmented_dmp_traj             = self.data['segmented_dmp_traj'][:data_limit]
            if data_length == None: data_length = len(self.segmented_dmp_traj)

        if 'dmp_scaling' in self.data:
            self.dmp_scaling                = self.data['dmp_scaling']
            self.with_scaling               = True
            # self.dmp_scaling[0]             = self.dmp_scaling[0][begin_idx:]
            # self.dmp_scaling[1]             = self.dmp_scaling[1][begin_idx:]
        if 'segmented_dmp_scaling' in self.data:
            self.segmented_dmp_scaling      = self.data['segmented_dmp_scaling']
            self.with_scaling               = True
            # self.segmented_dmp_scaling[0]   = self.segmented_dmp_scaling[0][begin_idx:]
            # self.segmented_dmp_scaling[1]   = self.segmented_dmp_scaling[1][begin_idx:]

        # self.dmp_scaling.x_max          = from_numpy(self.dmp_scaling.x_max[begin_idx:]).to(DEVICE)
        # self.dmp_scaling.x_min          = from_numpy(self.dmp_scaling.x_min[begin_idx:]).to(DEVICE)
        # self.dmp_scaling.y_max          = tensor(self.dmp_scaling.y_max).to(DEVICE)
        # self.dmp_scaling.y_min          = tensor(self.dmp_scaling.y_min).to(DEVICE)            

        self.combined_inputs            = []
        self.combined_outputs           = []
        
        for idx in range(data_length):
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
            
            if 'segmented_dict_dmp_outputs' in self.data:
                outputs['segmented_dict_dmp_outputs']   = torch.from_numpy(self.segmented_dict_dmp_outputs[idx]).float().to(DEVICE)
            if 'segmented_dict_dmp_types' in self.data:
                outputs['segmented_dict_dmp_types']     = torch.from_numpy(self.segmented_dict_dmp_types[idx]).float().to(DEVICE)
            if 'num_segments' in self.data:
                outputs['num_segments']                 = torch.from_numpy(self.num_segments[idx]).float().to(DEVICE)
            if 'traj' in self.data:
                outputs['traj']                         = torch.from_numpy(self.traj[idx]).float().to(DEVICE)
            if 'dmp_traj' in self.data:
                outputs['dmp_traj']                     = torch.from_numpy(self.dmp_traj[idx]).float().to(DEVICE)
            if 'dmp_traj_padded' in self.data:
                outputs['dmp_traj_padded']              = torch.from_numpy(self.dmp_traj_padded[idx]).float().to(DEVICE)
            if 'dmp_traj_interpolated' in self.data:
                outputs['dmp_traj_interpolated']        = torch.from_numpy(self.dmp_traj_interpolated[idx]).float().to(DEVICE)
            if 'segmented_dmp_traj' in self.data:
                outputs['segmented_dmp_traj']           = torch.from_numpy(self.segmented_dmp_traj[idx]).float().to(DEVICE)
            self.combined_outputs.append(outputs)

    def getData(self):
        return self.combined_inputs, self.combined_outputs

    def getTau(self):
        return self.tau

    def getDataLoader(self, data_ratio = [7, 2, 1], batch_size = 50):
        
        X_train, X_val, Y_train, Y_val  = train_test_split(
                                                        self.combined_inputs,
                                                        self.combined_outputs,
                                                        test_size=(data_ratio[1]+data_ratio[2])/sum(data_ratio))
        X_val, X_test, Y_val, Y_test    = train_test_split(
                                                        X_val,
                                                        Y_val, 
                                                        test_size=data_ratio[2]/(data_ratio[1]+data_ratio[2]))

        train_dataset                   = DMPDataset(X = X_train, Y = Y_train)
        val_dataset                     = DMPDataset(X = X_val, Y = Y_val)
        test_dataset                    = DMPDataset(X = X_test, Y = Y_test)

        train_loader                    = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader                      = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)
        test_loader                     = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

        if self.with_scaling:
            if 'segmented_dmp_scaling' in self.data:
                scaling = self.segmented_dmp_scaling
            elif 'dmp_scaling' in self.data:
                scaling = self.dmp_scaling
        else:
            scaling = None

        return [train_loader, val_loader, test_loader], scaling
    
class DMPDataset(Dataset):
    def __init__(self, X, Y = None):
        self.X                          = X
        self.Y                          = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = self.X[idx]
        if self.Y != None: 
            labels = self.Y[idx]
            return (inputs, labels)
        return inputs