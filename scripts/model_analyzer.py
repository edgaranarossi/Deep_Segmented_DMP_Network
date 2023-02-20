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

model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
# model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# model.eval()

images = torch.randn(1, 3, 150, 150)
targets = None
original_image_sizes = [(images.shape[-2], images.shape[-1])]

# images, targets = model.transform(images, targets)
features = model.backbone(images)
[print(features[i].reshape(-1).shape) for i in features]
    

# proposals, proposal_losses = model.rpn(images, features, targets)
# detections, detector_losses, keypoint_features = model.roi_heads(features, proposals, images.image_sizes, targets)

# post_detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)