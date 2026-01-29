"""
VGG Backbone for P2PNet - Matches Official TencentYoutuResearch Weights EXACTLY

Weight structure from checkpoint:
- body1: indices 0,1,3,4,7,8,10,11 (Conv+BN pairs)
- body2: indices 1,2,4,5,7,8 (Conv+BN pairs, starting at 1)
- body3: indices 1,2,4,5,7,8 (Conv+BN pairs, starting at 1)
- body4: indices 1,2,4,5,7,8 (Conv+BN pairs, starting at 1)

Key insight: body2/3/4 skip index 0 (MaxPool), 3, 6 (ReLU).
We use add_module to create non-contiguous indices, then apply ReLU/MaxPool manually.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Backbone_VGG(nn.Module):
    """
    VGG16-BN backbone that outputs multi-scale features for FPN.
    Matches the official P2PNet weight structure exactly.
    """
    
    def __init__(self, name='vgg16_bn', pretrained=True):
        super().__init__()
        
        # Load VGG16 with batch norm to get the pretrained layers
        if pretrained:
            try:
                vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            except:
                vgg = models.vgg16_bn(pretrained=True)
        else:
            try:
                vgg = models.vgg16_bn(weights=None)
            except:
                vgg = models.vgg16_bn(pretrained=False)
        
        features = list(vgg.features.children())
        
        # VGG16_bn structure (44 layers total):
        # Block 1 (0-6): Conv64, BN, ReLU, Conv64, BN, ReLU, MaxPool
        # Block 2 (7-13): Conv128, BN, ReLU, Conv128, BN, ReLU, MaxPool
        # Block 3 (14-23): Conv256, BN, ReLU, Conv256, BN, ReLU, Conv256, BN, ReLU, MaxPool
        # Block 4 (24-33): Conv512, BN, ReLU, Conv512, BN, ReLU, Conv512, BN, ReLU, MaxPool
        # Block 5 (34-43): Conv512, BN, ReLU, Conv512, BN, ReLU, Conv512, BN, ReLU, MaxPool
        
        # ============================================
        # BODY1: Block 1 + Block 2 (indices 0,1,3,4,7,8,10,11)
        # ============================================
        # Contains: Conv64, BN, Conv64, BN, Conv128, BN, Conv128, BN
        # Skips: ReLU (2,5,9,12), MaxPool (6,13)
        self.body1 = nn.Sequential()
        self.body1.add_module('0', features[0])   # Conv(3->64)
        self.body1.add_module('1', features[1])   # BN(64)
        # index 2 skipped (ReLU)
        self.body1.add_module('3', features[3])   # Conv(64->64)
        self.body1.add_module('4', features[4])   # BN(64)
        # index 5 skipped (ReLU)
        # index 6 skipped (MaxPool)
        self.body1.add_module('7', features[7])   # Conv(64->128)
        self.body1.add_module('8', features[8])   # BN(128)
        # index 9 skipped (ReLU)
        self.body1.add_module('10', features[10]) # Conv(128->128)
        self.body1.add_module('11', features[11]) # BN(128)
        # index 12 skipped (ReLU)
        # index 13 skipped (MaxPool)
        
        # ============================================
        # BODY2: Block 3 (indices 1,2,4,5,7,8)
        # ============================================
        # Contains: Conv256, BN, Conv256, BN, Conv256, BN
        # Starts at 1 (index 0 is MaxPool from previous block)
        self.body2 = nn.Sequential()
        self.body2.add_module('1', features[14])  # Conv(128->256)
        self.body2.add_module('2', features[15])  # BN(256)
        # index 3 skipped (ReLU)
        self.body2.add_module('4', features[17])  # Conv(256->256)
        self.body2.add_module('5', features[18])  # BN(256)
        # index 6 skipped (ReLU)
        self.body2.add_module('7', features[20])  # Conv(256->256)
        self.body2.add_module('8', features[21])  # BN(256)
        # index 9 skipped (ReLU + MaxPool)
        
        # ============================================
        # BODY3: Block 4 (indices 1,2,4,5,7,8)
        # ============================================
        self.body3 = nn.Sequential()
        self.body3.add_module('1', features[24])  # Conv(256->512)
        self.body3.add_module('2', features[25])  # BN(512)
        self.body3.add_module('4', features[27])  # Conv(512->512)
        self.body3.add_module('5', features[28])  # BN(512)
        self.body3.add_module('7', features[30])  # Conv(512->512)
        self.body3.add_module('8', features[31])  # BN(512)
        
        # ============================================
        # BODY4: Block 5 (indices 1,2,4,5,7,8)
        # ============================================
        self.body4 = nn.Sequential()
        self.body4.add_module('1', features[34])  # Conv(512->512)
        self.body4.add_module('2', features[35])  # BN(512)
        self.body4.add_module('4', features[37])  # Conv(512->512)
        self.body4.add_module('5', features[38])  # BN(512)
        self.body4.add_module('7', features[40])  # Conv(512->512)
        self.body4.add_module('8', features[41])  # BN(512)
        
        # MaxPool layer for downsampling between blocks
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Output channels for FPN
        self.num_channels = [256, 512, 512]  # C3, C4, C5

    def forward(self, x):
        # BODY1: Block 1 + Block 2
        # Conv+BN+ReLU, Conv+BN+ReLU, MaxPool, Conv+BN+ReLU, Conv+BN+ReLU, MaxPool
        x = F.relu(self.body1._modules['1'](self.body1._modules['0'](x)))  # Conv64+BN+ReLU
        x = F.relu(self.body1._modules['4'](self.body1._modules['3'](x)))  # Conv64+BN+ReLU
        x = self.maxpool(x)  # H/2
        x = F.relu(self.body1._modules['8'](self.body1._modules['7'](x)))  # Conv128+BN+ReLU
        x = F.relu(self.body1._modules['11'](self.body1._modules['10'](x)))  # Conv128+BN+ReLU
        x = self.maxpool(x)  # H/4, 128 channels
        
        # BODY2: Block 3 -> C3 (256 channels, H/8)
        x = F.relu(self.body2._modules['2'](self.body2._modules['1'](x)))  # Conv256+BN+ReLU
        x = F.relu(self.body2._modules['5'](self.body2._modules['4'](x)))  # Conv256+BN+ReLU
        x = F.relu(self.body2._modules['8'](self.body2._modules['7'](x)))  # Conv256+BN+ReLU
        c3 = self.maxpool(x)  # H/8, 256 channels
        
        # BODY3: Block 4 -> C4 (512 channels, H/16)
        x = F.relu(self.body3._modules['2'](self.body3._modules['1'](c3)))
        x = F.relu(self.body3._modules['5'](self.body3._modules['4'](x)))
        x = F.relu(self.body3._modules['8'](self.body3._modules['7'](x)))
        c4 = self.maxpool(x)  # H/16, 512 channels
        
        # BODY4: Block 5 -> C5 (512 channels, H/32)
        x = F.relu(self.body4._modules['2'](self.body4._modules['1'](c4)))
        x = F.relu(self.body4._modules['5'](self.body4._modules['4'](x)))
        x = F.relu(self.body4._modules['8'](self.body4._modules['7'](x)))
        c5 = self.maxpool(x)  # H/32, 512 channels
        
        return c3, c4, c5


def build_backbone(name='vgg16_bn', pretrained=True):
    """Build backbone with given name."""
    return Backbone_VGG(name, pretrained)
