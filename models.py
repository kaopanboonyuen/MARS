"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: This module defines the MARS model, incorporating a ResNet-50 backbone 
             and a quadtree-based attention mechanism for car damage instance segmentation.
License: MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class QuadtreeAttention(nn.Module):
    def __init__(self, in_channels):
        super(QuadtreeAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class MARSModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MARSModel, self).__init__()
        
        # Backbone: ResNet50 with ViT-like transformer layers
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the final layers
        
        # Quadtree-based Attention
        self.quadtree_attention = QuadtreeAttention(in_channels=2048)
        
        # Upsampling and Final Convolution for Segmentation
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.quadtree_attention(x)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        
        x = self.upsample(x)
        return x