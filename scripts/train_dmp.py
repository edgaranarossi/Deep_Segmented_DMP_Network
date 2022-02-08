from utils.dataset_importer import PickleDataLoader
from utils.trainer import Trainer
from parameters import TrainingParameters

from os.path import join
from torchsummary import summary
import torch
import pickle as pkl

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    train_param = TrainingParameters()
    model_param = train_param.model_param
    dmp_param   = model_param.dmp_param

    open(train_param.log_writer_path, 'w')
    writer = SummaryWriter(train_param.model_save_path+'/log')

    train_param.writeLog('Importing ' + train_param.dataset_name + ' ...')
    file_data_loader = PickleDataLoader(train_param.dataset_path,
                                        # data_limit = 1000,
                                        include_tau = (True if train_param.includes_tau and model_param.dmp_param.tau == None else False))
    train_param.writeLog(train_param.dataset_name + ' imported')

    train_param.writeLog('Splitting dataset')
    data_loaders, scale = \
        file_data_loader.getDataLoader(data_ratio = train_param.data_ratio,
                                      batch_size = train_param.batch_size)
    dmp_param.scale = scale
    train_param.writeLog('Dataset split')

    train_param.writeLog('Saving training parameters')
    pkl.dump(train_param, open(join(train_param.model_save_path, "train-model-dmp_param.pkl"),"wb"))
    train_param.writeLog('Saved as ' + join(train_param.model_save_path, "train-model-dmp_param.pkl"))

    model = model_param.model(train_param)
    train_param.writeInitLog(model)
    summary(model, model_param.image_dim, batch_size = train_param.batch_size)
    
    trainer = Trainer(model, train_param, train_param.model_save_path, train_param.log_writer_path, writer)
#%%
    trainer.train(data_loaders)
    print('\nModel saved in '+ train_param.model_save_path)