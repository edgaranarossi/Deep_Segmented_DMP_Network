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

class DMPParamScale:
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

class Scaler:
    def __init__(self, data, range_new = [-1.0, 1.0]):
        self.range_old = MinMax(min = data.min(), max = data.max())
        self.range_new = MinMax(min = range_new[0], max = range_new[1])
        self.old_data = data
        self.normalized_data = self.normalize()

    def normalize(self):
        normalized_data = (self.range_new.max - self.range_new.min) * \
                          (self.old_data      - self.range_old.min) / \
                          (self.range_old.max - self.range_old.min) + \
                           self.range_new.min
        return normalized_data

    def denormalize(self, X):
        denormalized_data = (X - self.range_new.min) / \
                            (self.range_new.max - self.range_new.min) * \
                            (self.range_old.max - self.range_old.min) + \
                             self.range_old.min
        return denormalized_data

class PickleDataLoader:
    def __init__(self, pkl_path, data_limit = None, keys_to_normalize = []):
        self.data                           = pickle.load(open(pkl_path, 'rb'))
        self.with_scaling                   = False
        self.data_limit                     = data_limit
        self.scaler                         = {}
        self.normalized_keys                = keys_to_normalize
        self.max_segment                    = None

        for key in keys_to_normalize:
            self.scaler[key] = Scaler(self.data[key])
            self.data[key] = self.scaler[key].normalized_data
        
        self.buildDataset()          

    def buildDataset(self):
        if 'dmp_y0_goal_w' in self.data:
            self.dmp_y0_goal_w                  = self.data['dmp_y0_goal_w'][:self.data_limit]
        if 'dmp_y0_goal_w_scaled' in self.data:
            self.dmp_y0_goal_w_scaled           = self.data['dmp_y0_goal_w_scaled'][:self.data_limit]
        if 'dmp_scaling' in self.data:
            self.dmp_scaling                    = self.data['dmp_scaling']
            self.with_scaling                   = True

        if 'points_padded' in self.data:
            self.points_padded                  = self.data['points_padded'][:self.data_limit]
        if 'segment_types_padded' in self.data:
            self.segment_types_padded           = self.data['segment_types_padded'][:self.data_limit]

        if 'image' in self.data:
            self.images                         = self.data['image'][:self.data_limit] / 255
        if 'image_name' in self.data:
            self.image_names                    = self.data['image_name'][:self.data_limit]
        if 'caption' in self.data:
            self.captions                       = self.data['caption'][:self.data_limit]
        if 'cut_distance' in self.data:
            self.cut_distance                   = self.data['cut_distance'][:self.data_limit]
        if 'traj' in self.data:
            self.traj                           = self.data['traj'][:self.data_limit]# * 100
        if 'traj_interpolated' in self.data:
            self.traj_interpolated              = self.data['traj_interpolated'][:self.data_limit]# * 100
        if 'normal_dmp_y_track' in self.data:
            if len(self.data['normal_dmp_y_track'].shape) == 4:
                self.normal_dmp_y_track         = self.data['normal_dmp_y_track'][:self.data_limit, 0]
            elif len(self.data['normal_dmp_y_track'].shape) == 3:
                self.normal_dmp_y_track         = self.data['normal_dmp_y_track']
        if 'segmented_dmp_traj' in self.data:
            self.segmented_dmp_traj             = self.data['segmented_dmp_traj'][:self.data_limit]

        if 'num_segments' in self.data:
            self.num_segments                   = self.data['num_segments'].reshape(-1, 1)[:self.data_limit]
        if 'dmp_y0_segments' in self.data:
            self.dmp_y0_segments                = self.data['dmp_y0_segments'][:self.data_limit]
        if 'dmp_goal_segments' in self.data:
            self.dmp_goal_segments              = self.data['dmp_goal_segments'][:self.data_limit]
            if self.max_segment == None: 
                self.max_segment                = self.data['dmp_goal_segments'].shape[1]
        if 'dmp_w_segments' in self.data:
            self.dmp_w_segments                 = self.data['dmp_w_segments'][:self.data_limit]
            if self.max_segment == None: 
                self.max_segment                = self.data['dmp_w_segments'].shape[1]
        
        if 'dmp_y0_normal' in self.data:
            self.dmp_y0_normal                  = self.data['dmp_y0_normal'][:self.data_limit]
        if 'dmp_goal_normal' in self.data:
            self.dmp_goal_normal                = self.data['dmp_goal_normal'][:self.data_limit]
        if 'dmp_w_normal' in self.data:
            self.dmp_w_normal                   = self.data['dmp_w_normal'][:self.data_limit]

        self.combined_inputs            = []
        self.combined_outputs           = []
        data_length = len(self.data[[key for key in self.data][0]])
        
        for idx in range(data_length):
            if psutil.virtual_memory().percent > 90:
                raise MemoryError("Out of Memory")
            inputs = {}
            if 'image' in self.data:
                inputs['image']                         = torch.from_numpy(self.images[idx]).float().to(DEVICE)
            if 'caption' in self.data:
                inputs['caption']                       = self.captions[idx]
            if 'dmp_y0_goal_w' in self.data:
                inputs['dmp_y0_goal_w']                 = torch.from_numpy(self.dmp_y0_goal_w[idx]).float().to(DEVICE)
            if 'dmp_y0_goal_w_scaled' in self.data:
                inputs['dmp_y0_goal_w_scaled']          = torch.from_numpy(self.dmp_y0_goal_w_scaled[idx]).float().to(DEVICE)
            self.combined_inputs.append(inputs)

            outputs = {}
            if 'dmp_y0_goal_w' in self.data:
                outputs['dmp_y0_goal_w']                = torch.from_numpy(self.dmp_y0_goal_w[idx]).float().to(DEVICE)
            if 'dmp_y0_goal_w_scaled' in self.data:
                outputs['dmp_y0_goal_w_scaled']         = torch.from_numpy(self.dmp_y0_goal_w_scaled[idx]).float().to(DEVICE)
            if 'points_padded' in self.data:
                outputs['points_padded']                = torch.from_numpy(self.points_padded[idx]).float().to(DEVICE)
            if 'segment_types_padded' in self.data:
                outputs['segment_types_padded']         = torch.from_numpy(self.segment_types_padded[idx]).float().to(DEVICE)
            if 'traj' in self.data:
                outputs['traj']                         = torch.from_numpy(self.traj[idx]).float().to(DEVICE)
            if 'traj_interpolated' in self.data:
                outputs['traj_interpolated']            = torch.from_numpy(self.traj_interpolated[idx]).float().to(DEVICE)
            if 'normal_dmp_y_track' in self.data:
                outputs['normal_dmp_y_track']           = torch.from_numpy(self.normal_dmp_y_track[idx]).float().to(DEVICE)
            # if 'dmp_traj_padded' in self.data:
            #     outputs['dmp_traj_padded']              = torch.from_numpy(self.dmp_traj_padded[idx]).float().to(DEVICE)
            if 'segmented_dmp_traj' in self.data:
                outputs['segmented_dmp_traj']           = torch.from_numpy(self.segmented_dmp_traj[idx]).float().to(DEVICE)

            if 'num_segments' in self.data:
                outputs['num_segments']                 = torch.from_numpy(self.num_segments[idx]).float().to(DEVICE)
            if 'dmp_y0_segments' in self.data:
                outputs['dmp_y0_segments']              = torch.from_numpy(self.dmp_y0_segments[idx]).float().to(DEVICE)
            if 'dmp_goal_segments' in self.data:
                outputs['dmp_goal_segments']            = torch.from_numpy(self.dmp_goal_segments[idx]).float().to(DEVICE)
                outputs['first_segment_goal']           = outputs['dmp_goal_segments'][0]
            if 'dmp_w_segments' in self.data:
                outputs['dmp_w_segments']               = torch.from_numpy(self.dmp_w_segments[idx]).float().reshape(self.max_segment, -1).to(DEVICE)
                outputs['first_segment_w']              = outputs['dmp_w_segments'][0]

            if 'dmp_y0_normal' in self.data:
                outputs['dmp_y0_normal']              = torch.from_numpy(self.dmp_y0_normal[idx]).float().to(DEVICE)
            if 'dmp_goal_normal' in self.data:
                outputs['dmp_goal_normal']            = torch.from_numpy(self.dmp_goal_normal[idx]).float().reshape(1, -1).to(DEVICE)
            if 'dmp_w_normal' in self.data:
                outputs['dmp_w_normal']               = torch.from_numpy(self.dmp_w_normal[idx]).float().reshape(1, -1).to(DEVICE)

            self.combined_outputs.append(outputs)

    def getData(self):
        return self.combined_inputs, self.combined_outputs

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
        elif self.scaler != {}:
            scaling = self.scaler
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