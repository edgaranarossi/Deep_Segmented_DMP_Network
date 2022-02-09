from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sys
import psutil
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

    def denormalize_np(self, X):
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

    def denormalize_torch(self, X):
        X_denormalized = torch.zeros_like(X)
        X_denormalized[:, :4] = (X[:, :4] - self.y_new.min) / \
                                (self.y_new.max - self.y_new.min) * \
                                (self.y_old.max - self.y_old.min) + \
                                self.y_old.min
        X_denormalized[:, 4:] = (X[:, 4:] - self.w_new.min) / \
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

        if 'dmp_y0_goal_w' in self.data:
            self.dmp_y0_goal_w                  = self.data['dmp_y0_goal_w'][:data_limit, begin_idx:]
            self.tau                            = self.dmp_y0_goal_w[0][0] if include_tau else 1
            if data_length == None: data_length = len(self.dmp_y0_goal_w)
        if 'dmp_y0_goal_w_scaled' in self.data:
            self.dmp_y0_goal_w_scaled           = self.data['dmp_y0_goal_w_scaled'][:data_limit, begin_idx:]
            self.tau                            = self.dmp_y0_goal_w_scaled[0][0] if include_tau else 1
            if data_length == None: data_length = len(self.dmp_y0_goal_w_scaled)

        if 'points_padded' in self.data:
            self.points_padded                  = self.data['points_padded'][:data_limit]
            if data_length == None: data_length = len(self.points_padded)
        if 'segment_types_padded' in self.data:
            self.segment_types_padded           = self.data['segment_types_padded'][:data_limit]
            if data_length == None: data_length = len(self.segment_types_padded)

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
            if data_length == None: data_length = len(self.num_segments)
        if 'traj' in self.data:
            self.traj                           = self.data['traj'][:data_limit]# * 100
            if data_length == None: data_length = len(self.traj)
        if 'normal_dmp_traj' in self.data:
            if len(self.data['normal_dmp_traj'].shape) == 4:
                self.normal_dmp_traj                = self.data['normal_dmp_traj'][:data_limit, 0]
            elif len(self.data['normal_dmp_traj'].shape) == 3:
                self.normal_dmp_traj                = self.data['normal_dmp_traj']
            if data_length == None: data_length = len(self.normal_dmp_traj)
        if 'segmented_dmp_traj' in self.data:
            self.segmented_dmp_traj             = self.data['segmented_dmp_traj'][:data_limit]
            if data_length == None: data_length = len(self.segmented_dmp_traj)

        if 'dmp_scaling' in self.data:
            self.dmp_scaling                = self.data['dmp_scaling']
            self.with_scaling               = True
            # self.dmp_scaling[0]             = self.dmp_scaling[0][begin_idx:]
            # self.dmp_scaling[1]             = self.dmp_scaling[1][begin_idx:]

        # self.dmp_scaling.x_max          = from_numpy(self.dmp_scaling.x_max[begin_idx:]).to(DEVICE)
        # self.dmp_scaling.x_min          = from_numpy(self.dmp_scaling.x_min[begin_idx:]).to(DEVICE)
        # self.dmp_scaling.y_max          = tensor(self.dmp_scaling.y_max).to(DEVICE)
        # self.dmp_scaling.y_min          = tensor(self.dmp_scaling.y_min).to(DEVICE)            

        self.combined_inputs            = []
        self.combined_outputs           = []
        
        for idx in range(data_length):
            if psutil.virtual_memory().percent > 90:
                raise MemoryError("Out of Memory")
            inputs = {}
            if 'image' in self.data:
                inputs['image']                         = torch.from_numpy(self.images[idx]).float().to(DEVICE)
            if 'caption' in self.data:
                inputs['caption']                       = self.captions[idx]
            if 'dmp_y0_goal_w' in self.data:
                inputs['dmp_y0_goal_w']        = torch.from_numpy(self.dmp_y0_goal_w[idx][begin_idx:]).float().to(DEVICE)
            if 'dmp_y0_goal_w_scaled' in self.data:
                inputs['dmp_y0_goal_w_scaled']          = torch.from_numpy(self.dmp_y0_goal_w_scaled[idx][begin_idx:]).float().to(DEVICE)
            self.combined_inputs.append(inputs)

            outputs = {}
            if 'dmp_y0_goal_w' in self.data:
                outputs['dmp_y0_goal_w']                = torch.from_numpy(self.dmp_y0_goal_w[idx][begin_idx:]).float().to(DEVICE)
            if 'dmp_y0_goal_w_scaled' in self.data:
                outputs['dmp_y0_goal_w_scaled']         = torch.from_numpy(self.dmp_y0_goal_w_scaled[idx][begin_idx:]).float().to(DEVICE)
            if 'points_padded' in self.data:
                outputs['points_padded']                = torch.from_numpy(self.points_padded[idx]).float().to(DEVICE)
            if 'segment_types_padded' in self.data:
                outputs['segment_types_padded']         = torch.from_numpy(self.segment_types_padded[idx]).float().to(DEVICE)
            if 'num_segments' in self.data:
                outputs['num_segments']                 = torch.from_numpy(self.num_segments[idx]).float().to(DEVICE)
            if 'traj' in self.data:
                outputs['traj']                         = torch.from_numpy(self.traj[idx]).float().to(DEVICE)
            if 'dmp_traj_interpolated' in self.data:
                outputs['traj_interpolated']            = torch.from_numpy(self.traj_interpolated[idx]).float().to(DEVICE)
            if 'normal_dmp_traj' in self.data:
                outputs['normal_dmp_traj']              = torch.from_numpy(self.normal_dmp_traj[idx]).float().to(DEVICE)
            if 'dmp_traj_padded' in self.data:
                outputs['dmp_traj_padded']              = torch.from_numpy(self.dmp_traj_padded[idx]).float().to(DEVICE)
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
            if 'dmp_scaling' in self.data:
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