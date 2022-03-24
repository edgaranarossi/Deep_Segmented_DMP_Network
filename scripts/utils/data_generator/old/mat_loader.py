import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Mapping:
    y_max = 1
    y_min = -1
    x_max = []
    x_min = []

class MatDataLoader:
    def __init__(self, mat_path, data_limit = None):
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
        begin_idx = None
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

        train_loader                    = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle = True)
        val_loader                      = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle = True)
        test_loader                     = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle = True)

        return [train_loader, val_loader, test_loader], self.dmp_scaling

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