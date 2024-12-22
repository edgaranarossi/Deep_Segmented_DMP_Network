from os import listdir
from os.path import join
from shutil import rmtree

def delete_failed_trains(model_path):
    """
    Delete failed training directories that do not contain 'best_net_parameters'.
    """
    dirs = listdir(model_path)
    fails = [i for i in dirs if 'best_net_parameters' not in listdir(join(model_path, i))]
    for fail in fails:
        rmtree(join(model_path, fail))

if __name__ == '__main__':
    model_path = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'
    delete_failed_trains(model_path)