from utils.dataset_importer import PickleDataLoader
from utils.networks import CNNDMPNet, NewCNNDMPNet, SegmentedDMPNet, DMPIntegratorNet, FixedSegmentDictDMPNet, DynamicSegmentDictDMPNet, SegmentNumCNN
from utils.trainer import Trainer
from parameters import TrainingParameters, ModelParameters

from os.path import join, isdir, dirname
from os import makedirs, getcwd, chdir
from torchsummary import summary
import torch
import pickle as pkl
from datetime import datetime
init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT_DIR = '/home/edgar/rllab/scripts/dmp/Segmented-Deep-DMPs'
# chdir(ROOT_DIR)

from tensorboardX import SummaryWriter

FILE_NAME = 'image-dict_output-traj_N_10000_n-bf_5_ay_4_dt_0_2022-01-30_18-45-47.pkl'
ext = FILE_NAME[-3:]
FILE_DIR = join(ROOT_DIR, 'data/pkl/shapes-8')
FILE_PATH = join(FILE_DIR,  FILE_NAME)
# MODEL_SAVE_PATH = join(ROOT_DIR, 'models', 'Model_' + init_time)
MODEL_SAVE_PATH = ROOT_DIR
MODEL_SAVE_PATH += '/models/Model_'
MODEL_SAVE_PATH += 'DynamicSegmentDictDMPNet'
MODEL_SAVE_PATH += '_' + init_time
if not isdir(MODEL_SAVE_PATH): makedirs(MODEL_SAVE_PATH)
LOG_WRITER_PATH = join(MODEL_SAVE_PATH, 'network_description.txt')
open(LOG_WRITER_PATH, 'w')

# SURROGATE_MODEL_NAME = 'Model_DMP_Integrator_2022-01-15_22-47-17'
# SURROGATE_MODEL_PATH = join(ROOT_DIR, 'models', SURROGATE_MODEL_NAME)
# PARAMETER_TO_LOAD = 'best' # or 'final'
# SURROGATE_PARAM_PKL_PATH = join(SURROGATE_MODEL_PATH, 'train-model-dmp_param.pkl')

def writeLog(log):
    print(log, '\n')
    if LOG_WRITER_PATH != None:
        LOG_FILE = open(LOG_WRITER_PATH, 'a')
        LOG_FILE.write('\n' + log)
        LOG_FILE.close()

def writeInitLog():
    writeLog('Network created: ' + init_time)
    if 'dmp' in model_param.output_mode:
        writeLog('Model : Segmented-Deep-DMPs.scripts.utils.networks.CNNDMPNet')
    elif 'traj' in model_param.output_mode and model_param.dmp_param.segments == None:
        writeLog('Model : Segmented-Deep-DMPs.scripts.utils.networks.NewCNNDMPNet')
    elif 'traj' in model_param.output_mode and model_param.dmp_param.segments != None:
        writeLog('Model : Segmented-Deep-DMPs.scripts.utils.networks.SegmentedDMPNet')
    elif 'dmp_traj_interpolated' in model_param.output_mode and model_param.dmp_param.segments != None:
        writeLog('Model : Segmented-Deep-DMPs.scripts.utils.networks.FixedSegmentDictDMPNet')
    elif 'segmented_dict_dmp_outputs' in model_param.output_mode and \
         'num_segments' in model_param.output_mode and \
         'segmented_dict_dmp_types' in model_param.output_mode and \
         model_param.dmp_param.segments != None:
        writeLog('Model : Segmented-Deep-DMPs.scripts.utils.networks.DynamicSegmentDictDMPNet')
    elif 'num_segments' in model_param.output_mode and \
         model_param.dmp_param.segments != None:
        writeLog('Model : Segmented-Deep-DMPs.scripts.utils.networks.SegmentNumCNN')
    writeLog(model.__str__())
    writeLog('Data Path : ' + FILE_PATH)
    writeLog('Model Save Path : ' + MODEL_SAVE_PATH)
    writeLog('Layer Sizes : ' + str(model_param.layer_sizes))

if __name__ == '__main__':
    writer = SummaryWriter(MODEL_SAVE_PATH+'/log')

    # surrogate_train_param = pkl.load(open(SURROGATE_PARAM_PKL_PATH, 'rb'))
    # surrogate_model_param = surrogate_train_param.model_param
    # surrogate_dmp_param   = surrogate_model_param.dmp_param
    # surrogate_input_size = int((2 * surrogate_dmp_param.dof) + (surrogate_dmp_param.n_bf * surrogate_dmp_param.dof))
    # surrogate_output_size = int(surrogate_dmp_param.dof / surrogate_dmp_param.dt)
    # surrogate_model = DMPIntegratorNet(surrogate_train_param, surrogate_input_size, surrogate_output_size, surrogate_model_param.layer_sizes)
    # surrogate_model.load_state_dict(torch.load(join(SURROGATE_MODEL_PATH, PARAMETER_TO_LOAD + '_net_parameters')))
    # for param in surrogate_model.parameters():
    #         param.requires_grad = False
    
    train_param = TrainingParameters()
    model_param = train_param.model_param
    dmp_param = model_param.dmp_param
    # output_size = [surrogate_model.layer_sizes[0] * dmp_param.segments]
    if 'dmp' in model_param.output_mode:
        model = CNNDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = 'MSE'
    elif 'traj' in model_param.output_mode and model_param.dmp_param.segments == None:
        model = NewCNNDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = 'MSE'
    # elif model_param.output_mode == 'traj' and model_param.dmp_param.segments != None:
    #     model = SegmentedDMPNet(train_param, output_size, surrogate_model)
    #     if train_param.loss_type == None: train_param.loss_type = 'MSE'
    elif 'segmented_dict_dmp_outputs' in model_param.output_mode and \
         'num_segments' in model_param.output_mode and \
         'segmented_dict_dmp_types' in model_param.output_mode and \
         model_param.dmp_param.segments != None:
        model = DynamicSegmentDictDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = ['MSE', 'MSE', 'MSE']
    elif 'num_segments' in model_param.output_mode and \
         model_param.dmp_param.segments != None:
        model = SegmentNumCNN(train_param)
        if train_param.loss_type == None: train_param.loss_type = ['MSE']
    elif 'dmp_traj_interpolated' in model_param.output_mode and \
         model_param.dmp_param.segments != None:
        model = FixedSegmentDictDMPNet(train_param)
        if train_param.loss_type == None: train_param.loss_type = 'SDTW'

    writeInitLog()
    # file_data_loader = MatDataLoader(FILE_PATH, include_tau = False if model_param.dmp_param.tau == None else True)
    writeLog('Importing ' + FILE_NAME + ' ...')
    file_data_loader = PickleDataLoader(FILE_PATH,
                                        data_limit = 1000,
                                        include_tau = (True if train_param.includes_tau and model_param.dmp_param.tau == None else False))
    writeLog(FILE_NAME + ' imported')
    writeLog('Splitting dataset')
    data_loaders, scale = \
        file_data_loader.getDataLoader(data_ratio = train_param.data_ratio,
                                      batch_size = train_param.batch_size)
    model_param.scale = scale
    writeLog('Saving training parameters')
    pkl.dump(train_param, open(join(MODEL_SAVE_PATH, "train-model-dmp_param.pkl"),"wb"))
    summary(model, model_param.image_dim, batch_size = train_param.batch_size)
    trainer = Trainer(model, train_param, MODEL_SAVE_PATH, LOG_WRITER_PATH, writer)
#%%
    trainer.train(data_loaders)
    print('\nModel saved in '+ MODEL_SAVE_PATH)