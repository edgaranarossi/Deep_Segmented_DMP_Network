from utils.dataset_importer import MatDataLoader, PickleDataLoader
from utils.networks import CNNDMPNet, NewCNNDMPNet, SegmentedDMPNet, DMPIntegratorNet
from utils.trainer import Trainer
from os import makedirs, getcwd, chdir
from os.path import join, isdir, dirname
import pickle as pkl
import torch
ROOT_DIR = '/home/edgar/rllab/scripts/dmp/Custom_Deep-DMP'
chdir(ROOT_DIR)

MODEL_NAME = 'Model_DMP_Integrator_2022-01-07_20-51-40'
MODEL_PATH = join(ROOT_DIR, 'models', MODEL_NAME)
PARAMETER_TO_LOAD = 'best' # or 'final'
PARAM_PKL_PATH = join(MODEL_PATH, 'train-model-dmp_param.pkl')

FILE_NAME = 'dmp_parameter-traj_N_5000000_n-bf_20_ay_25.pkl'
ext = FILE_NAME[-3:]
FILE_DIR = 'scripts/dataset'
FILE_PATH = join(ROOT_DIR, FILE_DIR,  FILE_NAME)

if __name__ == '__main__':
    train_param = pkl.load(open(PARAM_PKL_PATH, 'rb'))
    model_param = train_param.model_param
    dmp_param   = model_param.dmp_param
    input_size = int((2 * dmp_param.dof) + (dmp_param.n_bf * dmp_param.dof))
    output_size = int(dmp_param.dof / dmp_param.dt)
    model = DMPIntegratorNet(train_param, input_size, output_size, model_param.layer_sizes)
    model.load_state_dict(torch.load(join(MODEL_PATH, PARAMETER_TO_LOAD + '_net_parameters')))
    print(('Importing ' + FILE_NAME + ' ...'))
    file_data_loader = PickleDataLoader(FILE_PATH)
    print(FILE_NAME + ' imported')
    print('Splitting dataset')
    data_loaders, _ = file_data_loader.getDataLoader(input_mode  = model_param.input_mode,
                                                     output_mode = model_param.output_mode,
                                                     data_ratio  = train_param.data_ratio,
                                                     batch_size  = train_param.batch_size)
    trainer = Trainer(model, train_param)
    trainer.test(data_loaders, plot_comparison_idx = 30)
