# models/transformer_backbone.py

import torch.nn as nn
from torchvision.models import vit_b_16

class TransformerBackboneModel(nn.Module):
    def __init__(self, num_classes):
        super(TransformerBackboneModel, self).__init__()
        self.backbone = vit_b_16(pretrained=True)  # You can customize this
        self.backbone.heads = nn.Linear(self.backbone.heads.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)