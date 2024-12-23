"""
Playground Script

This script contains functions to test and visualize the Deep Segmented DMP Network.
"""

from PIL import Image
import numpy as np
from numpy import array, flipud, where
from matplotlib import pyplot as plt
from os import listdir
from os.path import join
from torch import from_numpy, cat
import torch
import pickle as pkl
from utils.pydmps_torch import DMPs_discrete_torch
from utils.networks import SegmentDMPCNN, CNNDeepDMP
from copy import deepcopy
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preProcessImage(img_np, segment_threshold=0.7, size=50):
    """
    Preprocess the input image.

    Parameters:
    img_np (array): The input image array.
    segment_threshold (float): The threshold for segmentation.
    size (int): The size to resize the image.

    Returns:
    array: The processed image.
    """
    assert len(img_np.shape) == 3
    assert img_np.shape[2] <= 3
    
    processed_img = deepcopy(img_np)
    if processed_img.max() > 1:
        processed_img = processed_img / 255
    if img_np.shape[2] == 3:
        processed_img = processed_img.mean(axis = 2)
    
    if segment_threshold == 'auto':
        segment_threshold = processed_img[:10, -10:].mean() * 0.7
    processed_img = where(processed_img > segment_threshold, 0., 1.)
    processed_img = array(Image.fromarray(processed_img).resize((size, size), Image.ANTIALIAS))
    
    return processed_img

def generateDMPSegments(train_param, preds, image=None, plot=True, dt=None, deep_dmp=False):
    """
    Generate DMP segments from predictions.

    Parameters:
    train_param (TrainingParameters): The training parameters.
    preds (tensor): The predictions from the network.
    image (array): The input image.
    plot (bool): Whether to plot the trajectory.
    dt (float): The time step for DMP.
    deep_dmp (bool): Whether to use deep DMP.

    Returns:
    array: The generated DMP segments.
    """
    if deep_dmp == False:
        idx_modifier = 0
        num_segments = int(torch.clamp(torch.round(preds[0]).reshape(1), max = train_param.model_param.max_segments).item())
    else: idx_modifier = -1
    y0 = preds[1 + idx_modifier].reshape(1, train_param.model_param.dmp_param.dof, 1)
    segment_goals = preds[2 + idx_modifier].reshape(-1, train_param.model_param.dmp_param.dof, 1)
    segment_weights = preds[3 + idx_modifier].reshape(-1, train_param.model_param.dmp_param.dof, train_param.model_param.dmp_param.n_bf)
    
    all_pos_pred = cat([y0, segment_goals], dim = 0)
    
    y0s_pred = all_pos_pred[:-1]
    goals_pred = all_pos_pred[1:]
    
    dmp_pred = DMPs_discrete_torch(n_dmps = train_param.model_param.dmp_param.dof, 
                                   n_bfs = train_param.model_param.dmp_param.n_bf, 
                                   ay = train_param.model_param.dmp_param.ay, 
                                   dt = train_param.model_param.dmp_param.dt if dt == None else dt)
    if deep_dmp == False:
        dmp_pred.y0         = y0s_pred[:num_segments]
        dmp_pred.goal       = goals_pred[:num_segments]
        dmp_pred.w          = segment_weights[:num_segments].reshape(num_segments, train_param.model_param.dmp_param.dof, train_param.model_param.dmp_param.n_bf)
    else:
        dmp_pred.y0         = y0s_pred
        dmp_pred.goal       = goals_pred
        dmp_pred.w          = segment_weights.reshape(1, train_param.model_param.dmp_param.dof, train_param.model_param.dmp_param.n_bf)
    y_track_pred, _, _  = dmp_pred.rollout()
    
    y_pred = y_track_pred.reshape(-1, train_param.model_param.dmp_param.dof)
    padding_x = 2
    padding_y = 1
    padding = array([[padding_x, padding_y]])
    
    multiplier = 28
    y_pred_np = ((y_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, train_param.model_param.dmp_param.dof)
    
    if plot: plotTrajectory(y_pred_np, image)
    
    return y_pred_np

def plotTrajectory(y_pred, image=None):
    """
    Plot the trajectory on the image.

    Parameters:
    y_pred (array): The predicted trajectory.
    image (array): The input image.
    """
    print(":: Plotting trajectory")
    y_pred_color = np.append(y_pred[:, 1], y_pred[-1, 1])
    y_pred_color = np.diff(y_pred_color)
    y_pred_color = (y_pred_color - y_pred_color.min()) / (y_pred_color.max() - y_pred_color.min())
    
    plt.figure(figsize=(6,6))
    plt.imshow(flipud(image), cmap='Greys_r', origin = 'lower')
    for i in range(y_pred.shape[0]):
        plt.scatter(y_pred[i,0], y_pred[i,1], c = (y_pred_color[i], 0, 1 - y_pred_color[i]))
    plt.show()
    
def rescaleOutput(preds, keys_to_normalize, scaler):
    """
    Rescale the output predictions.

    Parameters:
    preds (tensor): The predictions from the network.
    keys_to_normalize (list): The keys to normalize.
    scaler (Scaler): The scaler object.

    Returns:
    list: The rescaled predictions.
    """
    rescaled_pred = []
    for idx, key in enumerate(keys_to_normalize):
        rescaled_pred.append(scaler[key].denormalize(preds[idx][0]))
    return rescaled_pred

if __name__ == '__main__':
    """
    Main function to run the playground script.
    """
    print("\n:: Loading segment model")
    model_path_segment = '/home/robot-ll172/Documents/edgar/Segmented_Deep_DMPs/test_model/SegmentDMPCNN'
    train_param_segment_path = join(model_path_segment, 'train-model-dmp_param.pkl')
    best_param_segment_path = join(model_path_segment, 'best_net_parameters')
    train_param_segment = pkl.load(open(train_param_segment_path, 'rb'))
    model_segment = SegmentDMPCNN(train_param_segment)
    model_segment.load_state_dict(torch.load(best_param_segment_path))
    model_segment.eval()
    print(":: Model loaded")

    print("\n:: Loading Deep DMP model")
    model_path_deep_dmp = '/home/robot-ll172/Documents/edgar/Segmented_Deep_DMPs/test_model/CNNDeepDMP'
    train_param_deep_dmp_path = join(model_path_deep_dmp, 'train-model-dmp_param.pkl')
    best_param_deep_dmp_path = join(model_path_deep_dmp, 'best_net_parameters')
    train_param_deep_dmp = pkl.load(open(train_param_deep_dmp_path, 'rb'))
    model_deep_dmp = CNNDeepDMP(train_param_deep_dmp)
    model_deep_dmp.load_state_dict(torch.load(best_param_deep_dmp_path))
    model_deep_dmp.eval()
    print(":: Model loaded")

    DT_DEEP_DMP = train_param_deep_dmp.model_param.dmp_param.dt
    DT_DEEP_DMP = 0.01
    DT = DT_DEEP_DMP

    FRAME_SIZE = 480
    IMG_SIZE = 50

    vid = cv2.VideoCapture(0)
    
    while(True):
        key = cv2.waitKey(1)
        ret, frame = vid.read()

        if frame is None: frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3))
        frame_cropped = np.array(frame[:, :frame.shape[0]])
        frame_cropped_resized = array(Image.fromarray(frame_cropped).resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS))
        processed_img = preProcessImage(frame_cropped, segment_threshold='auto')
        
        preds_segment = model_segment(from_numpy(processed_img.reshape(1, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE).float())
        preds_deep_dmp = model_deep_dmp(from_numpy(processed_img.reshape(1, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE).float())
        rescaled_pred_segment = rescaleOutput(preds_segment, train_param_segment.model_param.keys_to_normalize, train_param_segment.scale)
        rescaled_pred_deep_dmp = rescaleOutput(preds_deep_dmp, train_param_deep_dmp.model_param.keys_to_normalize, train_param_deep_dmp.scale)
        y_pred_segment = generateDMPSegments(train_param_segment, rescaled_pred_segment, image=processed_img, plot=False, dt=DT)
        y_pred_deep_dmp = generateDMPSegments(train_param_deep_dmp, rescaled_pred_deep_dmp, image=processed_img, plot=False, dt=DT_DEEP_DMP, deep_dmp=True)

        to_plot = np.fliplr(np.rot90(np.rot90(np.array(frame_cropped)))).astype(np.uint8).copy()
        y_pred_color_segment = np.append(y_pred_segment[:, 1], y_pred_segment[-1, 1])
        y_pred_color_segment = np.diff(y_pred_color_segment)
        y_pred_color_segment = (y_pred_color_segment - y_pred_color_segment.min()) / (y_pred_color_segment.max() - y_pred_color_segment.min())

        y_pred_color_deep_dmp = np.append(y_pred_deep_dmp[:, 1], y_pred_deep_dmp[-1, 1])
        y_pred_color_deep_dmp = np.diff(y_pred_color_deep_dmp)
        y_pred_color_deep_dmp = (y_pred_color_deep_dmp - y_pred_color_deep_dmp.min()) / (y_pred_color_deep_dmp.max() - y_pred_color_deep_dmp.min())
        
        to_plot_2 = deepcopy(to_plot)
        
        for i, p in enumerate(y_pred_segment):
            to_plot = cv2.circle(img=to_plot, center=(int(np.round(p[0] / 50 * FRAME_SIZE)), int(np.round(p[1] / 50 * FRAME_SIZE))), radius=1, color=[255 - (y_pred_color_segment[i] * 255), 0, y_pred_color_segment[i] * 255], thickness=2)
        
        for i, p in enumerate(y_pred_deep_dmp):
            to_plot_2 = cv2.circle(img=to_plot_2, center=(int(np.round(p[0] / 50 * FRAME_SIZE)), int(np.round(p[1] / 50 * FRAME_SIZE))), radius=1, color=[0, 255 - (y_pred_color_deep_dmp[i] * 255), y_pred_color_deep_dmp[i] * 255], thickness=2)
        
        to_plot = array(Image.fromarray(to_plot).resize((FRAME_SIZE*2, FRAME_SIZE*2), Image.ANTIALIAS))
        to_plot_2 = array(Image.fromarray(to_plot_2).resize((FRAME_SIZE*2, FRAME_SIZE*2), Image.ANTIALIAS))
        processed_img = array(Image.fromarray(processed_img).resize((FRAME_SIZE*2, FRAME_SIZE*2), Image.ANTIALIAS))
        processed_img = np.tile(processed_img.reshape(FRAME_SIZE*2, FRAME_SIZE*2, 1), (1, 1, 3)) / processed_img.max() * 255
        processed_img = np.where(processed_img < 50, 0, processed_img)
        processed_img = processed_img.astype(np.uint8).copy()
        
        to_plot = np.fliplr(np.rot90(np.rot90(to_plot)))
        to_plot_2 = np.fliplr(np.rot90(np.rot90(to_plot_2)))
        cv2.imshow('frame', np.concatenate((processed_img.astype(np.uint8), to_plot, to_plot_2), axis = 1))
        
        if key == 27:
            break
        
    vid.release() 
    cv2.destroyAllWindows()

    import cv2
    import numpy as np

    vid = cv2.VideoCapture(0)
    frame_size = 480

    while(True):
        k = cv2.waitKey(1)
        ret, frame = vid.read()
        if frame is None: frame = np.zeros((frame_size, frame_size, 3))
        frame_cropped = np.array(frame[:, :frame.shape[0]])
        cv2.imshow('frame', frame_cropped)
        if k == 27:
            break
        
    vid.release() 
    cv2.destroyAllWindows()