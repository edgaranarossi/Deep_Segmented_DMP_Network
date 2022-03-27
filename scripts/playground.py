#%%
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
from utils.networks import SegmentDMPCNN
from copy import deepcopy
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 50

model_path = '/home/robot-ll172/Documents/edgar/Segmented_Deep_DMPs/test_model/SegmentDMPCNN'
# img_path = '/home/robot-ll172/Documents/edgar/Segmented_Deep_DMPs/test_model'
train_param_path = join(model_path, 'train-model-dmp_param.pkl')
best_param_path = join(model_path, 'best_net_parameters')
train_param = pkl.load(open(train_param_path, 'rb'))
model = SegmentDMPCNN(train_param)
model.load_state_dict(torch.load(best_param_path))
model.eval()

def preProcessImage(img_np, segment_threshold = 0.7, size = 50):
    assert len(img_np.shape) == 3
    assert img_np.shape[2] <= 3
    
    processed_img = deepcopy(img_np)
    if processed_img.max() > 1:
        processed_img = processed_img / 255
    if img_np.shape[2] == 3:
        processed_img = processed_img.mean(axis = 2)
    
    if segment_threshold == 'auto':
        segment_threshold = processed_img[:10, -10:].mean() * 0.7
        # print(segment_threshold, processed_img[:10, -10:].mean())
    processed_img = where(processed_img > segment_threshold, 0., 1.)
    processed_img = array(Image.fromarray(processed_img).resize((size, size), Image.ANTIALIAS))
    
    return processed_img

def plotDMPSegments(preds, image = None, plot = True):
    num_segments = int(torch.clamp(torch.round(preds[0]).reshape(1), max = train_param.model_param.max_segments).item())
    y0 = preds[1].reshape(1, train_param.model_param.dmp_param.dof, 1)
    segment_goals = preds[2].reshape(-1, train_param.model_param.dmp_param.dof, 1)
    segment_weights = preds[3].reshape(-1, train_param.model_param.dmp_param.dof, train_param.model_param.dmp_param.n_bf)
    
    all_pos_pred = cat([y0, segment_goals], dim = 0)
    
    y0s_pred = all_pos_pred[:-1]
    goals_pred = all_pos_pred[1:]
    
    dmp_pred = DMPs_discrete_torch(n_dmps = train_param.model_param.dmp_param.dof, 
                                   n_bfs = train_param.model_param.dmp_param.n_bf, 
                                   ay = train_param.model_param.dmp_param.ay, 
                                   dt = train_param.model_param.dmp_param.dt)
    # dmp_pred.y0 = rescaled_pred[1].reshape(1, train_param.model_param.dmp_param.dof, 1)
    dmp_pred.y0         = y0s_pred[:num_segments]
    dmp_pred.goal       = goals_pred[:num_segments]
    dmp_pred.w          = segment_weights[:num_segments].reshape(num_segments, train_param.model_param.dmp_param.dof, train_param.model_param.dmp_param.n_bf)
    y_track_pred, _, _  = dmp_pred.rollout()
    
    y_pred = y_track_pred.reshape(-1, train_param.model_param.dmp_param.dof)
    #%
    padding_x = 2
    padding_y = 1
    padding = array([[padding_x, padding_y]])
    
    multiplier = 28
    y_pred_np = ((y_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, train_param.model_param.dmp_param.dof)
    all_pos_pred_np = ((all_pos_pred.detach().cpu().numpy() * multiplier).reshape(-1, train_param.model_param.dmp_param.dof) + padding)
    
    if plot:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(flipud(image), cmap='Greys_r', origin = 'lower')
        plt.scatter(all_pos_pred_np[:num_segments + 1, 0], all_pos_pred_np[:num_segments + 1, 1], c = 'r', zorder = 6)
        plt.plot(y_pred_np[:,0], y_pred_np[:,1], c = 'r')
        plt.show()
    
    return y_pred_np
    
def rescaleOutput(preds, keys_to_normalize, scaler):
    rescaled_pred = []
    for idx, key in enumerate(keys_to_normalize):
        rescaled_pred.append(scaler[key].denormalize(preds[idx][0]))
    return rescaled_pred
# #%%
# img_names = [i for i in listdir(img_path) if not isdir(join(img_path, i)) and i[-3:] == 'jpg' and i[:4] == 'test']
# # img_name = 'test_0'
# for img_name in img_names:
#     original_img = array(Image.open(join(img_path, img_name)))
#     original_img_resized = array(Image.fromarray(original_img).resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS))
#     processed_img = preProcessImage(original_img, image = original_img_resized)
    
#     preds = model(from_numpy(processed_img.reshape(1, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE).float())
    
#     rescaled_pred = []
#     for idx, key in enumerate(train_param.model_param.keys_to_normalize):
#         rescaled_pred.append(train_param.model_param.dmp_param.scale[key].denormalize(preds[idx][0]))
    
#     num_segments_pred = int(torch.round(rescaled_pred[0]).reshape(1).item())
#     y0_pred = rescaled_pred[1]
#     goals_pred = rescaled_pred[2]
#     weights_pred = rescaled_pred[3]
    
#     plotDMPSegments(num_segments_pred, y0_pred, goals_pred, weights_pred)

# #%%
# from time import sleep
# from datetime import datetime
 
# time_between_inference = 5
# prev_infer_time = datetime.now()
frame_size = 480

# overlay_img = np.zeros()
vid = cv2.VideoCapture(0)
  
while(True):
    k = cv2.waitKey(1)
    # print(k)
    # cur_time = datetime.now()
    # Capture the video frame
    # by frame  
    ret, frame = vid.read()
    if frame is None: frame = np.zeros((frame_size, frame_size, 3))
    frame_cropped = np.array(frame[:, :frame.shape[0]])
    frame_cropped_resized = array(Image.fromarray(frame_cropped).resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS))
    processed_img = preProcessImage(frame_cropped, segment_threshold = 'auto')
    
    preds = model(from_numpy(processed_img.reshape(1, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE).float())
    rescaled_pred = rescaleOutput(preds, train_param.model_param.keys_to_normalize, train_param.model_param.dmp_param.scale)
    y_pred = plotDMPSegments(rescaled_pred, image = processed_img, plot = False)
    # prev_infer_time = cur_time
    # if (cur_time - prev_infer_time).seconds >= time_between_inference:
    if k == 32:
        y_pred = plotDMPSegments(rescaled_pred, image = processed_img)
    
    # Display the resulting frame
    # to_plot = (np.tile(processed_img.reshape(frame_size, frame_size, 1), (1, 1, 3)) * 255).astype(np.uint8)
    to_plot = np.fliplr(np.rot90(np.rot90(frame_cropped))).astype(np.uint8).copy()
    for p in y_pred:
        # to_plot[int(np.round(p[0] / 50 * frame_size)), int(np.round(p[1]) / 50 * frame_size)] = [0, 0, 255]
        to_plot = cv2.circle(img = to_plot, center = (int(np.round(p[0] / 50 * frame_size)), int(np.round(p[1]) / 50 * frame_size)), radius = 1, color = [0, 0, 255], thickness = 2)
    # to_plot[40, 40] = [0, 0, 255]
    to_plot = array(Image.fromarray(to_plot).resize((frame_size*2, frame_size*2), Image.ANTIALIAS))
    to_plot = np.fliplr(np.rot90(np.rot90(to_plot)))
    cv2.imshow('frame', to_plot)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if k == 27:
        break
    
vid.release() 
# Destroy all the windows
cv2.destroyAllWindows()

#%%
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
# Destroy all the windows
cv2.destroyAllWindows()