from os import listdir
from os.path import join
from shutil import rmtree

model_path = '/home/edgar/rllab/scripts/dmp/Segmented-Deep-DMPs/models'
dirs = listdir(model_path)
fails = [i for i in dirs if 'best_net_parameters' not in listdir(join(model_path, i))]
for fail in fails:
	rmtree(join(model_path, fail))