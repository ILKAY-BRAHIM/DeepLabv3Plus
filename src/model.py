import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        # Image-level features
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        
        # Global features
        h, w = x.shape[2:]
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        res.append(global_feat)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(DeepLabv3Plus, self).__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            low_level_features = 64
            high_level_features = 2048
        elif backbone == 'resnet101':
            self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            low_level_features = 64
            high_level_features = 2048
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # ASPP module with atrous rates
        self.aspp = ASPP(high_level_features, 256, atrous_rates=[6, 12, 18])
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_features, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Split input into RGB and edge channels and ensure float32
        rgb_input = x[:, :3, :, :].float()  # RGB channels
        edge_input = x[:, 3:, :, :].float()  # Edge channel
        
        # Forward pass through backbone
        # Get low-level features
        low_level_features = self.backbone.conv1(rgb_input)
        low_level_features = self.backbone.bn1(low_level_features)
        low_level_features = self.backbone.relu(low_level_features)
        low_level_features = self.backbone.maxpool(low_level_features)
        
        # Get high-level features
        high_level_features = self.backbone.layer1(low_level_features)
        high_level_features = self.backbone.layer2(high_level_features)
        high_level_features = self.backbone.layer3(high_level_features)
        high_level_features = self.backbone.layer4(high_level_features)
        
        # ASPP
        aspp_features = self.aspp(high_level_features)
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        
        # Upsample ASPP features to match low-level features size
        aspp_features = F.interpolate(aspp_features, size=low_level_features.shape[2:], 
                                   mode='bilinear', align_corners=True)
        
        # Concatenate features
        combined_features = torch.cat([aspp_features, low_level_features], dim=1)
        
        # Decoder
        output = self.decoder(combined_features)
        
        # Upsample to input size
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return output