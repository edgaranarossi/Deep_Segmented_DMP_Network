from utils.dataset_importer import PickleDataLoader
from utils.trainer import Trainer
from parameters import TrainingParameters
from os import chdir, getcwd
from os.path import join
from torchsummary import summary
import torch
import pickle as pkl
from tensorboardX import SummaryWriter
chdir(getcwd())

if __name__ == '__main__':
    train_param = TrainingParameters()
    model_param = train_param.model_param
    dmp_param   = model_param.dmp_param

    open(train_param.log_writer_path, 'w')
    writer = SummaryWriter(train_param.model_save_path+'/log')
#%%
    if train_param.data_loaders_model_name == None:
        train_param.writeLog('Importing ' + train_param.dataset_name + ' ...')
        file_data_loader = PickleDataLoader(train_param)
        train_param.writeLog('Dataset imported')
    #%%
        train_param.writeLog('Splitting dataset')
        data_loaders, scaler = \
            file_data_loader.getDataLoader(data_ratio = train_param.data_ratio,
                                          batch_size = train_param.batch_size)
        train_param.writeLog('Dataset split')
    else:
        train_param.writeLog('Reusing {} data loader'.format(train_param.data_loaders_model_name))
        data_loaders_model_dir = join(train_param.root_dir, 
                                      'models', 
                                      train_param.data_loaders_model_name.split('_')[1],
                                      train_param.data_loaders_model_name)
        data_loaders_train_param = pkl.load(open(join(data_loaders_model_dir, 'train-model-dmp_param.pkl'), 'rb'))
        scaler = data_loaders_train_param.scaler
        train_param.dataset_path = data_loaders_train_param.dataset_path
        train_param.scaler = data_loaders_train_param.scaler
        data_loaders = pkl.load(open(join(data_loaders_model_dir, 'data_loaders.pkl'), 'rb'))
        # unscaled_data = [i for i in model_param.output_mode if i not in data_loaders_train_param.model_param.output_mode]
    
    pkl.dump(data_loaders, open(join(train_param.model_save_path, "data_loaders.pkl"),"wb"))
    train_param.scaler = scaler
#%%
    if train_param.load_model_name == None:
        model = model_param.model(train_param)
    else:
        train_param.writeLog('Loading {} weights'.format(train_param.load_model_name))
        load_train_param = pkl.load(open(join(train_param.root_dir, 'models', train_param.load_model_name.split('_')[1], train_param.load_model_name, 'train-model-dmp_param.pkl'), 'rb'))
        load_model_param = load_train_param.model_param
        model_param.hidden_layer_sizes = load_model_param.hidden_layer_sizes
        if 'DSDNet' in train_param.model_type:
            model_param.decoder_layer_sizes = load_model_param.decoder_layer_sizes
        # train_param.model_param = load_model_param
        # model_param = train_param.model_param

        model = model_param.model(train_param)
        model.load_state_dict(torch.load(join(train_param.root_dir, 'models', train_param.load_model_name.split('_')[1], train_param.load_model_name, 'best_net_parameters')))
    train_param.writeInitLog(model)
    # summary(model, model_param.image_dim, batch_size = train_param.batch_size)

    train_param.writeLog('Saving training parameters')
    pkl.dump(train_param, open(join(train_param.model_save_path, "train-model-dmp_param.pkl"),"wb"))
    train_param.writeLog('Saved as ' + join(train_param.model_save_path, "train-model-dmp_param.pkl"))
    
    trainer = Trainer(model, train_param, train_param.model_save_path, train_param.log_writer_path, writer)
#%%
    trainer.train(data_loaders)
    print('\nModel saved in '+ train_param.model_save_path)