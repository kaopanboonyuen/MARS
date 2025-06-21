"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: Flexible MARS instance segmentation model with dynamic backbone selection
             (Mask R-CNN, Mask2Former, PointRend, etc.) and quadtree attention.
License: MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# For Mask2Former, PointRend, etc. you may need external libs like detectron2 or timm
# For demo, we'll use torchvision models only (Mask R-CNN, and backbone alternatives)

# Placeholder imports for demonstration, replace with actual implementations if available
try:
    from mask2former import Mask2Former # hypothetical import
except ImportError:
    Mask2Former = None

try:
    from pointrend import PointRend # hypothetical import
except ImportError:
    PointRend = None


class QuadtreeAttention(nn.Module):
    """
    Quadtree Attention Module for refining feature maps via self-attention.
    
    Args:
        in_channels (int): Number of input feature channels.
        
    Inputs:
        x (Tensor): Input feature map of shape (B, C, H, W).
        
    Returns:
        Tensor: Attention-refined feature map of same shape (B, C, H, W).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        
        # Compute query, key, and value feature maps
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x N x C'
        key = self.key_conv(x).view(B, -1, H * W)                       # B x C' x N
        value = self.value_conv(x).view(B, -1, H * W)                   # B x C x N
        
        # Calculate attention map (B x N x N)
        attention = self.softmax(torch.bmm(query, key))  # dot-product attention
        
        # Apply attention map to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        
        # Reshape back to feature map shape
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable scaling
        out = self.gamma * out + x
        
        return out

class MARSModel(nn.Module):
    def __init__(self, 
                 num_classes=2, 
                 backbone_name='resnet50', 
                 pretrained=True,
                 use_quadtree_attention=True):
        """
        Args:
            num_classes (int): Number of segmentation classes.
            backbone_name (str): Choice of backbone. Supported: 'resnet50', 'maskrcnn', 'mask2former', 'pointrend'
            pretrained (bool): Load pretrained weights if available.
            use_quadtree_attention (bool): Whether to add quadtree attention module.
        """
        super(MARSModel, self).__init__()

        self.use_quadtree_attention = use_quadtree_attention

        if backbone_name == 'resnet50':
            # Use ResNet50 backbone from torchvision without classification head
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_out_channels = 2048

        elif backbone_name == 'maskrcnn':
            # Load Mask R-CNN pretrained on COCO, replace classifier for num_classes
            model = maskrcnn_resnet50_fpn(pretrained=pretrained)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, num_classes)
            self.backbone = model.backbone
            backbone_out_channels = 256  # FPN output channels

            # Store full model for forward (override forward)
            self.detection_model = model

        elif backbone_name == 'mask2former' and Mask2Former is not None:
            # Hypothetical: instantiate Mask2Former model here, pretrained if possible
            self.detection_model = Mask2Former(num_classes=num_classes, pretrained=pretrained)
            backbone_out_channels = self.detection_model.backbone_out_channels

        elif backbone_name == 'pointrend' and PointRend is not None:
            # Hypothetical: instantiate PointRend model here
            self.detection_model = PointRend(num_classes=num_classes, pretrained=pretrained)
            backbone_out_channels = self.detection_model.backbone_out_channels

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # If using maskrcnn or similar full model, skip quadtree and custom head
        if backbone_name in ['maskrcnn', 'mask2former', 'pointrend']:
            self.use_custom_head = False
        else:
            self.use_custom_head = True
            if self.use_quadtree_attention:
                self.quadtree_attention = QuadtreeAttention(in_channels=backbone_out_channels)

            # Custom segmentation head
            self.conv1 = nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        if hasattr(self, 'detection_model'):
            # For full detection models (maskrcnn, mask2former, pointrend),
            # just pass input through their own forward method
            return self.detection_model(x)

        else:
            # Backbone + quadtree + segmentation head
            features = self.backbone(x)

            if self.use_quadtree_attention:
                features = self.quadtree_attention(features)

            x = F.relu(self.conv1(features))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.conv4(x)
            x = self.upsample(x)
            return x