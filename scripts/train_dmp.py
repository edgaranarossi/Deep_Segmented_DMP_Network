from utils.dataset_importer import MatDataLoader, PickleDataLoader
from utils.networks import CNNDMPNet, NewCNNDMPNet, SegmentedDMPNet, DMPIntegratorNet
from utils.trainer import Trainer
from parameters import TrainingParameters, ModelParameters

from os.path import join, isdir, dirname
from os import makedirs, getcwd, chdir
from torchsummary import summary
import torch
import pickle as pkl
from datetime import datetime
init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT_DIR = '/home/edgar/rllab/scripts/dmp/Custom_Deep-DMP'
chdir(ROOT_DIR)

from tensorboardX import SummaryWriter

FILE_NAME = 'carrot_50-grayscale_distance_10-orientation_0-height_25-traj_length_1000-DMP-BF_500-dt_0.001-SegmentDMP-segments_25-BF_20-dt_0.025.pkl'
ext = FILE_NAME[-3:]
FILE_DIR = join(dirname(getcwd()), 'data', ext)
FILE_PATH = join(FILE_DIR,  FILE_NAME)
MODEL_SAVE_PATH = join(getcwd(), 'models', 'Model_' + init_time)
if not isdir(MODEL_SAVE_PATH): makedirs(MODEL_SAVE_PATH)
LOG_WRITER_PATH = join(MODEL_SAVE_PATH, 'network_description.txt')
open(LOG_WRITER_PATH, 'w')

MODEL_NAME = 'Model_DMP_Integrator_2022-01-07_20-51-40'
MODEL_PATH = join(ROOT_DIR, 'models', MODEL_NAME)
PARAMETER_TO_LOAD = 'best' # or 'final'
SURROGATE_PARAM_PKL_PATH = join(MODEL_PATH, 'train-model-dmp_param.pkl')

def writeLog(log):
    print(log, '\n')
    if LOG_WRITER_PATH != None:
        LOG_FILE = open(LOG_WRITER_PATH, 'a')
        LOG_FILE.write('\n' + log)
        LOG_FILE.close()

def writeInitLog():
    writeLog('Network created: ' + init_time)
    if model_param.output_mode == 'dmp':
        writeLog('Model : Custom_Deep-DMP.scripts.utils.networks.CNNDMPNet')
    elif model_param.output_mode == 'traj' and model_param.dmp_param.segments == None:
        writeLog('Model : Custom_Deep-DMP.scripts.utils.networks.NewCNNDMPNet')
    elif model_param.output_mode == 'traj' and model_param.dmp_param.segments != None:
        writeLog('Model : Custom_Deep-DMP.scripts.utils.networks.SegmentedDMPNet')
    writeLog('Data Path : ' + FILE_PATH)
    writeLog('Model Save Path : ' + MODEL_SAVE_PATH)
    writeLog('Layer Sizes : ' + str(model_param.layer_sizes))

if __name__ == '__main__':
    writer = SummaryWriter(MODEL_SAVE_PATH+'/log')

    surrogate_train_param = pkl.load(open(SURROGATE_PARAM_PKL_PATH, 'rb'))
    surrogate_model_param = surrogate_train_param.model_param
    surrogate_dmp_param   = surrogate_model_param.dmp_param
    surrogate_input_size = int((2 * surrogate_dmp_param.dof) + (surrogate_dmp_param.n_bf * surrogate_dmp_param.dof))
    surrogate_output_size = int(surrogate_dmp_param.dof / surrogate_dmp_param.dt)
    surrogate_model = DMPIntegratorNet(surrogate_train_param, surrogate_input_size, surrogate_output_size, surrogate_model_param.layer_sizes)
    surrogate_model.load_state_dict(torch.load(join(MODEL_PATH, PARAMETER_TO_LOAD + '_net_parameters')))
    
    train_param = TrainingParameters()
    model_param = train_param.model_param
    if model_param.output_mode == 'dmp':
        model = CNNDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = 'MSE'
    elif model_param.output_mode == 'traj' and model_param.dmp_param.segments == None:
        model = NewCNNDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = 'MSE'
    elif model_param.output_mode == 'traj' and model_param.dmp_param.segments != None:
        model = SegmentedDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = 'MSE'
    model.surrogate_model = surrogate_model

    writeInitLog()
    # file_data_loader = MatDataLoader(FILE_PATH, include_tau = False if model_param.dmp_param.tau == None else True)
    writeLog('Importing ' + FILE_NAME + ' ...')
    file_data_loader = PickleDataLoader(FILE_PATH, 'dmp_param-traj', include_tau = True if train_param.includes_tau and model_param.dmp_param.tau == None else False)
    writeLog(FILE_NAME + ' imported')
    writeLog('Splitting dataset')
    data_loaders, scale = \
        file_data_loader.getDataLoader(data_ratio = train_param.data_ratio,
                                      batch_size = train_param.batch_size,
                                      input_mode = model_param.input_mode,
                                      output_mode = model_param.output_mode)
    model_param.scale = scale
    writeLog('Saving training parameters')
    pkl.dump(train_param, open(join(MODEL_SAVE_PATH, "train-model-dmp_param.pkl"),"wb"))
    summary(model, model_param.image_dim, batch_size = train_param.batch_size)
    trainer = Trainer(model, train_param, MODEL_SAVE_PATH, LOG_WRITER_PATH, writer)
#%%
    trainer.train(data_loaders)