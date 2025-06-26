"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: Flexible MARS instance segmentation model with dynamic backbone selection
             supporting Mask R-CNN, Mask2Former, PointRend, etc., enhanced with
             quadtree attention for high-quality mask refinement.
License: MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class QuadtreeAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # B x N x C'
        key = self.key_conv(x).view(B, -1, H*W)                       # B x C' x N
        value = self.value_conv(x).view(B, -1, H*W)                   # B x C x N
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))            # B x C x N
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class MaskTransfinerModule(nn.Module):
    """
    Simplified Mask Transfiner refinement module.
    Refines coarse mask logits at selected quadtree nodes via transformer attention.
    """
    def __init__(self, feature_channels, num_classes, transformer_dim=256, num_layers=6, nhead=8):
        super().__init__()
        self.num_classes = num_classes
        self.feature_proj = nn.Conv2d(feature_channels, transformer_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, transformer_dim))  # max 1000 nodes

        # MLP to update mask logits at nodes
        self.mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, num_classes)
        )

    def select_quadtree_nodes(self, masks, max_nodes=1000):
        """
        Select points on coarse mask boundaries to refine.
        Here we use a simple heuristic: pixels where mask logits gradient is high.

        Args:
            masks: Tensor (B, num_classes, H, W) coarse mask logits
            max_nodes: max number of nodes to select per image

        Returns:
            List of (B) LongTensors with selected node indices (flattened H*W indices)
        """
        B, C, H, W = masks.shape
        nodes_indices = []
        for b in range(B):
            mask_logits = masks[b]  # C x H x W

            # Compute gradient magnitude of max class logits as proxy for boundary
            max_logits, _ = mask_logits.max(dim=0)  # H x W
            gx = torch.abs(max_logits[:, 1:] - max_logits[:, :-1])  # H x (W-1)
            gy = torch.abs(max_logits[1:, :] - max_logits[:-1, :])  # (H-1) x W
            grad_mag = torch.zeros_like(max_logits)
            grad_mag[:, :-1] += gx
            grad_mag[:-1, :] += gy

            grad_flat = grad_mag.view(-1)
            topk = torch.topk(grad_flat, min(max_nodes, grad_flat.size(0)), largest=True).indices
            nodes_indices.append(topk)
        return nodes_indices

    def forward(self, features, coarse_masks):
        """
        Args:
            features: (B, C, H, W) feature map from backbone
            coarse_masks: (B, num_classes, H, W) coarse mask logits

        Returns:
            refined_masks: (B, num_classes, H, W) refined mask logits
        """
        B, C, H, W = features.shape
        device = features.device

        # Project features to transformer_dim
        feat_proj = self.feature_proj(features)  # B x D x H x W
        D = feat_proj.shape[1]

        # Select quadtree nodes (pixel indices) to refine
        nodes_indices = self.select_quadtree_nodes(coarse_masks)  # list of length B

        refined_masks = coarse_masks.clone()

        for b in range(B):
            node_idx = nodes_indices[b]  # (N,) indices in flattened H*W
            if node_idx.numel() == 0:
                continue
            N = node_idx.shape[0]

            # Extract features at node positions
            feat_b = feat_proj[b].view(D, H*W).permute(1,0)  # (H*W, D)
            node_feats = feat_b[node_idx, :]  # (N, D)

            # Add positional encoding (slice from learned pos_embed)
            pos_embed = self.pos_embed[:, :N, :].squeeze(0)  # (N, D)
            node_feats = node_feats + pos_embed.to(device)

            # Transformer expects input (N, batch=1, D) or (S, B, E) format
            node_feats = node_feats.unsqueeze(1)  # (N,1,D)
            refined_feats = self.transformer(node_feats)  # (N,1,D)
            refined_feats = refined_feats.squeeze(1)  # (N, D)

            # MLP to predict mask update logits at selected nodes
            delta_logits = self.mlp(refined_feats)  # (N, num_classes)

            # Update coarse masks logits at selected nodes
            mask_b = refined_masks[b].view(self.num_classes, -1).permute(1, 0)  # (H*W, num_classes)
            mask_b[node_idx, :] = mask_b[node_idx, :] + delta_logits
            refined_masks[b] = mask_b.permute(1, 0).view(self.num_classes, H, W)

        return refined_masks


class MARSModel(nn.Module):
    def __init__(self, 
                 num_classes=2, 
                 backbone_name='resnet50', 
                 pretrained=True,
                 use_quadtree_attention=True,
                 use_mask_transfiner=False):
        super(MARSModel, self).__init__()

        self.use_quadtree_attention = use_quadtree_attention
        self.use_mask_transfiner = use_mask_transfiner

        if backbone_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_out_channels = 2048

        elif backbone_name == 'maskrcnn':
            model = maskrcnn_resnet50_fpn(pretrained=pretrained)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, num_classes)
            self.backbone = model.backbone
            backbone_out_channels = 256

            self.detection_model = model

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if backbone_name in ['maskrcnn']:
            self.use_custom_head = False
        else:
            self.use_custom_head = True
            if self.use_quadtree_attention:
                self.quadtree_attention = QuadtreeAttention(in_channels=backbone_out_channels)

            self.conv1 = nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        if self.use_mask_transfiner:
            self.mask_transfiner = MaskTransfinerModule(
                feature_channels=backbone_out_channels,
                num_classes=num_classes
            )

    def forward(self, x):
        if hasattr(self, 'detection_model'):
            # For Mask R-CNN full model, output masks directly
            outputs = self.detection_model(x)
            if self.use_mask_transfiner:
                # Extract backbone features and coarse mask logits to refine
                # This is simplified: actual Mask R-CNN outputs list of dicts
                # Here we assume batch size 1 for demo
                features = self.backbone(x)
                # Assume output mask logits are available in outputs[0]['masks']
                coarse_masks = outputs[0]['masks']  # (N, 1, H, W)
                # We simplify: pick first N masks and squeeze channel dim
                coarse_masks = coarse_masks.squeeze(1)  # (N, H, W)
                # Pad or reduce to batch dimension
                # Here we just take first mask as batch=1, num_classes=1 for demo
                coarse_masks = coarse_masks.unsqueeze(1)  # (N,1,H,W) treat as batch?
                refined_masks = self.mask_transfiner(features, coarse_masks)
                return refined_masks
            else:
                return outputs
        else:
            features = self.backbone(x)
            if self.use_quadtree_attention:
                features = self.quadtree_attention(features)
            x = F.relu(self.conv1(features))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.conv4(x)
            x = self.upsample(x)
            if self.use_mask_transfiner:
                x = self.mask_transfiner(features, x)
            return x