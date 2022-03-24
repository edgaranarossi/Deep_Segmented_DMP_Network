from utils.dataset_importer import MatDataLoader, PickleDataLoader, DMPParamScale
from utils.networks import CNNDMPNet, NewCNNDMPNet, SegmentedDMPNet, DMPIntegratorNet
from utils.trainer import Trainer
from parameters import TrainingParameters, ModelParameters
from os.path import join, isdir, dirname
from os import makedirs, getcwd, chdir
from torchinfo import summary
import pickle as pkl
from datetime import datetime
from tensorboardX import SummaryWriter
from os import listdir
from time import sleep

init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs'
chdir(ROOT_DIR)

FILE_NAME = 'dmp_parameter-traj_N_2000000_random-line-curves-scale_50-pos_randomized__n-bf_15_ay_15_dt_0_scale-pos_1_scale-w_1_lim-w_1e8.pkl'
ext = FILE_NAME[-3:]
FILE_DIR = 'scripts/dataset'
FILE_PATH = join(ROOT_DIR, FILE_DIR,  FILE_NAME)

while FILE_NAME not in listdir(join(ROOT_DIR, FILE_DIR)):
    print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), ':: Waiting for dataset', FILE_NAME)
    sleep(60)

MODEL_SAVE_PATH = join(getcwd(), 'models', 'Model_DMP_Integrator_' + init_time)
if not isdir(MODEL_SAVE_PATH): makedirs(MODEL_SAVE_PATH)
LOG_WRITER_PATH = join(MODEL_SAVE_PATH, 'network_description.txt')
open(LOG_WRITER_PATH, 'w')

def writeLog(log):
    if log[0] == '\n':
        print('\n', datetime.now().strftime("%Y-%m-%d_%H-%M-%S ::"), log[1:], '\n')
    else:
        print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S ::"), log, '\n')
    if LOG_WRITER_PATH != None:
        LOG_FILE = open(LOG_WRITER_PATH, 'a')
        LOG_FILE.write('\n' + log)
        LOG_FILE.close()

def writeInitLog():
    writeLog('Network created: ' + init_time)
    writeLog('Model : SegmentedDeepDMPs.scripts.utils.networks.DMPIntegratorNet')
    writeLog('Data Path : ' + FILE_PATH)
    writeLog('Model Save Path : ' + MODEL_SAVE_PATH)
    writeLog('Layer Sizes : ' + str(model.layer_sizes))

def saveParams():
    writeLog('Saving training parameters')
    pkl.dump(train_param, open(join(MODEL_SAVE_PATH, "train-model-dmp_param.pkl"),"wb"))

if __name__ == '__main__':
    writer = SummaryWriter(MODEL_SAVE_PATH+'/log')
    train_param = TrainingParameters()
    model_param = train_param.model_param
    dmp_param   = model_param.dmp_param
    input_size = int((2 * dmp_param.dof) + (dmp_param.n_bf * dmp_param.dof))
    output_size = int(dmp_param.dof / dmp_param.dt)
    model = DMPIntegratorNet(train_param, input_size, output_size, model_param.layer_sizes)
    train_param.loss_type = 'MSE'
    writeInitLog()
    writeLog('Importing ' + FILE_NAME + ' ...')
    file_data_loader = PickleDataLoader(FILE_PATH, data_limit = None)
    writeLog(FILE_NAME + ' imported')
    writeLog('Splitting dataset')
    data_loaders, _ = file_data_loader.getDataLoader(data_ratio  = train_param.data_ratio,
                                                     batch_size  = train_param.batch_size)
    # model_param.scale = scale
    saveParams()
    summary(model, input_size=(train_param.batch_size, input_size,))
    trainer = Trainer(model, train_param, MODEL_SAVE_PATH, LOG_WRITER_PATH, writer)
#%%
    trainer.train(data_loaders)
#%%
    trainer.test(data_loaders, plot_comparison_idx = 0)