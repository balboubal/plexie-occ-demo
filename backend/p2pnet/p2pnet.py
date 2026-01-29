"""
P2PNet Model - Matches Official TencentYoutuResearch Implementation

Key architecture:
1. VGG16-BN backbone -> multi-scale features (C3, C4, C5)
2. FPN to create P3, P4, P5 feature maps (all 256 channels)
3. Uses P3 (finest) for prediction
4. Regression head: predicts (x,y) offsets from anchor points
5. Classification head: predicts confidence for person class

Anchor configuration: row=2, line=2 -> 4 anchors per spatial location
Output: 8 channels for both heads (4 anchors * 2 values)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone


class AnchorPoints(nn.Module):
    """Generate anchor points for each spatial location."""
    
    def __init__(self, space=8, row=2, line=2):
        super().__init__()
        self.space = space
        self.row = row
        self.line = line
        
    def forward(self, x):
        """Generate anchor points based on input feature map size."""
        b, c, h, w = x.shape
        
        # Create grid of anchor centers
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=x.device) * self.space + self.space // 2
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=x.device) * self.space + self.space // 2
        
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # Generate sub-anchors within each cell
        anchors = []
        for i in range(self.row):
            for j in range(self.line):
                # Offset from center
                offset_x = (j - (self.line - 1) / 2) * self.space / self.line
                offset_y = (i - (self.row - 1) / 2) * self.space / self.row
                
                anchor_x = shift_x + offset_x
                anchor_y = shift_y + offset_y
                anchors.append(torch.stack([anchor_x, anchor_y], dim=-1))
        
        # Stack all anchors: (H, W, num_anchors, 2)
        anchors = torch.stack(anchors, dim=2)
        # Flatten: (H*W*num_anchors, 2)
        anchors = anchors.view(-1, 2)
        
        return anchors


class P2PNet(nn.Module):
    """P2PNet crowd counting model."""
    
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        
        self.backbone = backbone
        self.row = row
        self.line = line
        self.num_anchors = row * line  # 4
        
        # FPN layers - reduce channels and merge scales
        # Input channels from backbone: C3=256, C4=512, C5=512
        self.fpn = nn.ModuleDict({
            'P5_1': nn.Conv2d(512, 256, 1),
            'P5_2': nn.Conv2d(256, 256, 3, padding=1),
            'P4_1': nn.Conv2d(512, 256, 1),
            'P4_2': nn.Conv2d(256, 256, 3, padding=1),
            'P3_1': nn.Conv2d(256, 256, 1),
            'P3_2': nn.Conv2d(256, 256, 3, padding=1),
        })
        
        # Regression head: predicts (x, y) offsets
        # 4 conv layers + output layer
        self.regression = nn.ModuleDict({
            'conv1': nn.Conv2d(256, 256, 3, padding=1),
            'conv2': nn.Conv2d(256, 256, 3, padding=1),
            'conv3': nn.Conv2d(256, 256, 3, padding=1),
            'conv4': nn.Conv2d(256, 256, 3, padding=1),
            'output': nn.Conv2d(256, self.num_anchors * 2, 3, padding=1),  # 8 channels
        })
        
        # Classification head: predicts class scores
        # 4 conv layers + output layer
        self.classification = nn.ModuleDict({
            'conv1': nn.Conv2d(256, 256, 3, padding=1),
            'conv2': nn.Conv2d(256, 256, 3, padding=1),
            'conv3': nn.Conv2d(256, 256, 3, padding=1),
            'conv4': nn.Conv2d(256, 256, 3, padding=1),
            'output': nn.Conv2d(256, self.num_anchors * 2, 3, padding=1),  # 8 = 4 anchors * 2 classes
        })
        
        # Anchor generator
        self.anchor_gen = AnchorPoints(space=8, row=row, line=line)
        
    def forward(self, x):
        # Get backbone features
        c3, c4, c5 = self.backbone(x)  # 256, 512, 512 channels
        
        # FPN forward
        # Top-down pathway
        p5 = self.fpn['P5_1'](c5)
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        
        p4 = self.fpn['P4_1'](c4) + p5_up
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        
        p3 = self.fpn['P3_1'](c3) + p4_up
        
        # Smooth
        p5 = self.fpn['P5_2'](p5)
        p4 = self.fpn['P4_2'](p4)
        p3 = self.fpn['P3_2'](p3)
        
        # Use P3 (finest resolution) for predictions
        feat = p3
        
        # Regression head
        reg = F.relu(self.regression['conv1'](feat))
        reg = F.relu(self.regression['conv2'](reg))
        reg = F.relu(self.regression['conv3'](reg))
        reg = F.relu(self.regression['conv4'](reg))
        reg = self.regression['output'](reg)  # (B, 8, H, W)
        
        # Classification head
        cls = F.relu(self.classification['conv1'](feat))
        cls = F.relu(self.classification['conv2'](cls))
        cls = F.relu(self.classification['conv3'](cls))
        cls = F.relu(self.classification['conv4'](cls))
        cls = self.classification['output'](cls)  # (B, 8, H, W)
        
        # Reshape outputs
        b, _, h, w = reg.shape
        
        # reg: (B, 8, H, W) -> (B, H*W*4, 2)
        reg = reg.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 8)
        reg = reg.view(b, -1, 2)  # (B, H*W*4, 2)
        
        # cls: (B, 8, H, W) -> (B, H*W*4, 2)
        cls = cls.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 8)
        cls = cls.view(b, -1, 2)  # (B, H*W*4, 2)
        
        # Generate anchor points
        anchors = self.anchor_gen(feat)  # (H*W*4, 2)
        
        # Add offsets to anchors to get predicted points
        pred_points = anchors.unsqueeze(0) + reg
        
        return {
            'pred_logits': cls,
            'pred_points': pred_points,
        }


def build_model(row=2, line=2, pretrained_backbone=False):
    """Build P2PNet model."""
    backbone = build_backbone('vgg16_bn', pretrained=pretrained_backbone)
    model = P2PNet(backbone, row=row, line=line)
    return model
