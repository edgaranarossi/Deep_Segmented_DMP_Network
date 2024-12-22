#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 02:29:01 2022

@author: edgar
"""

from os import listdir
from os.path import join, isdir
import pandas as pd
import pickle as pkl

def analyze_models(model_dir, net_desc_name):
    """
    Analyze models and extract relevant information.

    Parameters:
    model_dir (str): Directory containing the models.
    net_desc_name (str): Name of the network description file.

    Returns:
    pd.DataFrame: DataFrame containing parsed model information.
    """
    models = [i for i in listdir(model_dir) if isdir(join(model_dir, i)) and
                                               'DSDNet' in i and
                                               net_desc_name in listdir(join(model_dir, i))]
    models = sorted(models)

    descs = []
    for model in models:
        with open(join(model_dir, model, net_desc_name)) as f:
            lines = f.readlines()
        descs.append([model, lines])
    
    valid_models = [i for i in descs if len(i[1]) > 0 and
                                        'Final Validation Loss' in i[1][-2] and
                                        ' :: Epoch : ' not in i[1][-1] and
                                        'Network created' in i[1][1]]

    parsed = []
    for model in valid_models:
        parsed.append([model[0], model[1][1].split(': ')[-1][:-1], float(model[1][-1].split(' : ')[-1])])
    return pd.DataFrame(parsed)

def analyze_models_with_images(model_dir, net_desc_name):
    """
    Analyze models with images and extract relevant information.

    Parameters:
    model_dir (str): Directory containing the models.
    net_desc_name (str): Name of the network description file.

    Returns:
    pd.DataFrame: DataFrame containing parsed model information.
    """
    models_with_img = [i for i in listdir(model_dir) if isdir(join(model_dir, i)) and
                                                        'Integrator' in i and
                                                        net_desc_name in listdir(join(model_dir, i)) and
                                                        len([j for j in listdir(join(model_dir, i)) if 'png' in j]) > 0]
    models_with_img = sorted(models_with_img)

    descs = []
    for model in models_with_img:
        with open(join(model_dir, model, net_desc_name)) as f:
            lines = f.readlines()
        descs.append(lines)

    valid_models = [i for i in descs if 'Best Validation Loss' in i[-1] and
                                        'tensor' not in i[-1] and
                                        'Network created' in i[1]]

    parsed = []
    for model in valid_models:
        parsed.append([model[1].split(': ')[-1][:-1], 
                       model[3].split(' : ')[-1].split('/')[-1].replace('generated\\', ''), 
                       float(model[-1].split(' : ')[-1])])
    return pd.DataFrame(parsed)

def load_train_params(model_dir, model_name):
    """
    Load training parameters from a model.

    Parameters:
    model_dir (str): Directory containing the model.
    model_name (str): Name of the model.

    Returns:
    dict: Training parameters.
    """
    return pkl.load(open(join(model_dir, model_name, 'train-model-dmp_param.pkl'), 'rb'))

def load_data_loaders(model_root_dir, model_name):
    """
    Load data loaders from a model.

    Parameters:
    model_root_dir (str): Root directory containing the models.
    model_name (str): Name of the model.

    Returns:
    tuple: Training, validation, and test data loaders.
    """
    data_path = join(model_root_dir, model_name.split('_')[1], model_name, 'data_loaders.pkl')
    data_loaders = pkl.load(open(data_path, 'rb'))
    return data_loaders

def analyze_dataset(data_root_dir, data_name):
    """
    Analyze a dataset and print relevant information.

    Parameters:
    data_root_dir (str): Root directory containing the dataset.
    data_name (str): Name of the dataset.

    Returns:
    None
    """
    data_path = join(data_root_dir, data_name)
    dataset = pkl.load(open(data_path, 'rb'))

    print('Dataset name\n{}\n'.format(data_name))

    print('Normal DMP data')
    print('y0:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_y0'].min(),
                                                     dataset['normal_dmp_y0'].max(),
                                                     np.abs(dataset['normal_dmp_y0'].max() - dataset['normal_dmp_y0'].min())))
    print('goal:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_goal'].min(),
                                                       dataset['normal_dmp_goal'].max(),
                                                       np.abs(dataset['normal_dmp_goal'].max() - dataset['normal_dmp_goal'].min())))
    print('w:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_w'].min(),
                                                    dataset['normal_dmp_w'].max(),
                                                    np.abs(dataset['normal_dmp_w'].max() - dataset['normal_dmp_w'].min())))

    if 'normal_dmp_L_y0' in dataset:
        print('\nNormal DMP L data')
        print('y0:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_L_y0'].min(),
                                                         dataset['normal_dmp_L_y0'].max(),
                                                         np.abs(dataset['normal_dmp_L_y0'].max() - dataset['normal_dmp_L_y0'].min())))
        print('goal:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_L_goal'].min(),
                                                           dataset['normal_dmp_L_goal'].max(),
                                                           np.abs(dataset['normal_dmp_L_goal'].max() - dataset['normal_dmp_L_goal'].min())))
        print('w:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['normal_dmp_L_w'].min(),
                                                        dataset['normal_dmp_L_w'].max(),
                                                        np.abs(dataset['normal_dmp_L_w'].max() - dataset['normal_dmp_L_w'].min())))

    print('\nSegmented DMP data')
    print('y0:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['segmented_dmp_y0'].min(),
                                                     dataset['segmented_dmp_y0'].max(),
                                                     np.abs(dataset['segmented_dmp_y0'].max() - dataset['segmented_dmp_y0'].min())))
    print('goal:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['segmented_dmp_goal'].min(),
                                                       dataset['segmented_dmp_goal'].max(),
                                                       np.abs(dataset['segmented_dmp_goal'].max() - dataset['segmented_dmp_goal'].min())))
    print('w:\n    min = {}\n    max = {}\n    dist = {}'.format(dataset['segmented_dmp_w'].min(),
                                                    dataset['segmented_dmp_w'].max(),
                                                    np.abs(dataset['segmented_dmp_w'].max() - dataset['segmented_dmp_w'].min())))

def save_dataset(dataset, data_root_dir, data_name):
    """
    Save the dataset with a new name.

    Parameters:
    dataset (dict): The dataset to save.
    data_root_dir (str): Root directory to save the dataset.
    data_name (str): Original name of the dataset.

    Returns:
    None
    """
    from datetime import datetime

    new_data_name = data_name.split('.')
    new_data_name[-2] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    new_data_name = '.'.join(new_data_name)

    pkl.dump(dataset, open(join(data_root_dir, new_data_name), 'wb'))
    print('Saved as {}'.format(new_data_name))

if __name__ == '__main__':
    net_desc_name = 'network_description.txt'
    model_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'
    df = analyze_models(model_dir, net_desc_name)
    print(df)

    df_with_images = analyze_models_with_images(model_dir, net_desc_name)
    print(df_with_images)

    model_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models/DSDNet'
    model_name = 'Model_SegmentDMPCNN_2022-04-06_13-42-39'
    train_param = load_train_params(model_dir, model_name)
    print(train_param)

    model_root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/models'
    model_name = 'Model_DSDNet_2022-07-28_13-22-12'
    train_loader, val_loader, test_loader = load_data_loaders(model_root_dir, model_name)
    print(train_loader, val_loader, test_loader)

    data_root_dir = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/cutting'
    data_name = 'cutting_5000.num_data_5000_num_seg_15.normal_dmp_bf_300_ay_100_dt_0.001.seg_dmp_bf_20_ay_7_dt_0.2022-09-12_03-15-37.pkl'
    analyze_dataset(data_root_dir, data_name)

    dataset = pkl.load(open(join(data_root_dir, data_name), 'rb'))
    save_dataset(dataset, data_root_dir, data_name)