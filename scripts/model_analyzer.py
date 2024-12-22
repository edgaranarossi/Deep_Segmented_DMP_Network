#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:24:25 2022

@author: edgar
"""

from torchviz import make_dot
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
import torch

def analyze_model(model, images):
    """
    Analyze the model by printing the shapes of the features.
    
    Parameters:
    model (torch.nn.Module): The model to analyze.
    images (torch.Tensor): The input images.
    """
    features = model.backbone(images)
    for i in features:
        print(features[i].reshape(-1).shape)

if __name__ == '__main__':
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    images = torch.randn(1, 3, 150, 150)
    analyze_model(model, images)